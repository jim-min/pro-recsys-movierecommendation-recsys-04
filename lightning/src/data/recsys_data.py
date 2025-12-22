import os
import logging
import random
import numpy as np
import pandas as pd
import lightning as L
import torch
from scipy import sparse
from torch.utils.data import DataLoader, TensorDataset

log = logging.getLogger(__name__)


class RecSysDataModule(L.LightningDataModule):
    """
    RecSys 데이터 모듈 (Collaborative Filtering)

    Hydra 설정 사용:
        data:
            data_dir: "~/data/train/"
            batch_size: 512
            valid_ratio: 0.1
            min_interactions: 5
            split_strategy: "random"  # "random", "leave_one_out", "temporal_user", "temporal_global"
            temporal_split_ratio: 0.8  # temporal split용 (0.8 = 80% train, 20% valid)

    Split Strategies:
        - random: 각 유저별로 랜덤하게 valid_ratio 비율만큼 분할
        - leave_one_out: 각 유저별로 랜덤하게 1개의 아이템만 validation
        - temporal_user: 각 유저별로 시간순 정렬 후 temporal_split_ratio 기준 분할 (time 컬럼 필요)
        - temporal_global: 전체 데이터를 시간순 정렬 후 temporal_split_ratio 기준 분할 (time 컬럼 필요)
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 512,
        valid_ratio: float = 0.1,
        min_interactions: int = 5,
        seed: int = 42,
        data_file="train_ratings.csv",
        split_strategy: str = "random",
        temporal_split_ratio: float = 0.8,
    ):
        super().__init__()
        self.data_dir = os.path.expanduser(data_dir)
        self.batch_size = batch_size
        self.valid_ratio = valid_ratio
        self.min_interactions = min_interactions
        self.seed = seed
        self.data_file = data_file
        self.split_strategy = split_strategy
        self.temporal_split_ratio = temporal_split_ratio

        # 인코딩 매핑 (setup 후 사용 가능)
        self.user2idx = None  # user_id -> user_idx
        self.idx2user = None  # user_idx -> user_id
        self.item2idx = None  # item_id -> item_idx
        self.idx2item = None  # item_idx -> item_id

        self.num_users = None
        self.num_items = None

        self.train_mat = None  # CSR sparse matrix (num_users, num_items)
        self.valid_gt = None  # dict: {user_idx: [item_idx, ...]}

    def prepare_data(self):
        """데이터 다운로드 및 전처리 (단일 프로세스에서만 실행)"""
        # 이미 로컬에 있는 데이터 사용
        pass

    def setup(self, stage: str = None):
        """데이터 로드 및 분할"""
        log.info("=" * 60)
        log.info("Setting up RecSys DataModule...")

        if self.num_users:  # 이미 초기화 되어 있으면 skip
            log.info("Skip as RecSysDataModule is already initialized")
        else:
            # 1. 상호작용 데이터 읽기
            log.info("Step 1/4: Reading interaction data...")
            df = self._read_interactions()
            log.info(f"  - Loaded {len(df):,} interactions")

            # 2. ID 인코딩 (user_id, item_id -> 연속적인 인덱스)
            log.info("Step 2/4: Encoding user/item IDs...")
            df_enc = self._encode_ids(df)
            log.info(f"  - Users: {self.num_users:,}, Items: {self.num_items:,}")

            # 3. Train/Valid 분할
            log.info(f"Step 3/4: Splitting train/validation data (strategy: {self.split_strategy})...")
            temporal_strategies = ["temporal_user", "temporal_global"]
            if self.valid_ratio > 0 or self.split_strategy in ["leave_one_out"] + temporal_strategies:
                if self.split_strategy == "random":
                    train_df, self.valid_gt = self._train_valid_split_random(df_enc)
                elif self.split_strategy == "leave_one_out":
                    train_df, self.valid_gt = self._train_valid_split_leave_one_out(df_enc)
                elif self.split_strategy == "temporal_user":
                    train_df, self.valid_gt = self._train_valid_split_temporal_user(df_enc, df)
                elif self.split_strategy == "temporal_global":
                    train_df, self.valid_gt = self._train_valid_split_temporal_global(df_enc, df)
                else:
                    raise ValueError(f"Unknown split_strategy: {self.split_strategy}")

                n_valid = sum(len(items) for items in self.valid_gt.values())
                log.info(f"  - Train: {len(train_df):,} interactions")
                log.info(f"  - Valid: {n_valid:,} interactions")
            else:
                train_df = df_enc
                self.valid_gt = {u: [] for u in range(self.num_users)}
                log.info(f"  - Train: {len(train_df):,} interactions (no validation)")

            # 4. Sparse Matrix 생성 (CSR 형식)
            log.info("Step 4/4: Building sparse user-item matrix...")
            self.train_mat = self._build_user_item_matrix(train_df)
            log.info(f"  - Matrix shape: {self.train_mat.shape}")
            log.info(
                f"  - Sparsity: {100 * (1 - self.train_mat.nnz / (self.num_users * self.num_items)):.2f}%"
            )
        log.info("DataModule setup complete!")
        log.info("=" * 60)

    def _read_interactions(self):
        """train_ratings.csv 읽기 및 필터링"""
        file_path = os.path.join(self.data_dir, self.data_file)
        df = pd.read_csv(file_path)

        # 최소 상호작용 수 필터링
        if self.min_interactions > 0:
            user_counts = df["user"].value_counts()
            valid_users = user_counts[user_counts >= self.min_interactions].index
            df = df[df["user"].isin(valid_users)]

        return df

    def _encode_ids(self, df):
        """user_id, item_id를 0부터 시작하는 연속적인 인덱스로 변환"""
        unique_users = sorted(df["user"].unique())
        unique_items = sorted(df["item"].unique())

        self.user2idx = {uid: idx for idx, uid in enumerate(unique_users)}
        self.idx2user = {idx: uid for uid, idx in self.user2idx.items()}
        self.item2idx = {iid: idx for idx, iid in enumerate(unique_items)}
        self.idx2item = {idx: iid for iid, idx in self.item2idx.items()}

        self.num_users = len(self.user2idx)
        self.num_items = len(self.item2idx)

        df_enc = df.copy()
        df_enc["user"] = df["user"].map(self.user2idx)
        df_enc["item"] = df["item"].map(self.item2idx)

        return df_enc

    def _train_valid_split_random(self, df_enc):
        """각 유저별로 랜덤하게 valid_ratio 비율만큼 validation set으로 분할"""
        random.seed(self.seed)
        np.random.seed(self.seed)

        train_rows = []
        valid_gt = {u: [] for u in range(self.num_users)}

        for u_idx in range(self.num_users):
            user_items = df_enc[df_enc["user"] == u_idx]["item"].tolist()
            n_items = len(user_items)

            if n_items == 0:
                continue

            # validation 개수 계산
            n_valid = max(1, int(n_items * self.valid_ratio))

            # 랜덤 샘플링
            valid_items = random.sample(user_items, n_valid)
            train_items = [it for it in user_items if it not in valid_items]

            # validation ground truth 저장
            valid_gt[u_idx] = valid_items

            # train 데이터 저장
            for it in train_items:
                train_rows.append({"user": u_idx, "item": it})

        train_df = pd.DataFrame(train_rows)
        return train_df, valid_gt

    def _train_valid_split_leave_one_out(self, df_enc):
        """각 유저별로 마지막 1개를 validation set으로 분할 (Leave-One-Out)"""
        random.seed(self.seed)
        np.random.seed(self.seed)

        train_rows = []
        valid_gt = {u: [] for u in range(self.num_users)}

        for u_idx in range(self.num_users):
            user_items = df_enc[df_enc["user"] == u_idx]["item"].tolist()
            n_items = len(user_items)

            if n_items == 0:
                continue

            # 랜덤하게 1개를 validation으로 선택
            valid_item = random.choice(user_items)
            train_items = [it for it in user_items if it != valid_item]

            # validation ground truth 저장
            valid_gt[u_idx] = [valid_item]

            # train 데이터 저장
            for it in train_items:
                train_rows.append({"user": u_idx, "item": it})

        train_df = pd.DataFrame(train_rows)
        return train_df, valid_gt

    def _train_valid_split_temporal_user(self, df_enc, df_original):
        """유저별 시간 기반 분할: 각 유저의 interaction을 시간순으로 정렬 후 분할"""
        # df_original에 timestamp가 있다고 가정
        # timestamp 컬럼이 없으면 에러 발생
        if "time" not in df_original.columns:
            raise ValueError("Temporal split requires 'time' column in the data")

        # df_enc에 timestamp 추가
        df_enc = df_enc.copy()
        df_enc["time"] = df_original["time"].values

        train_rows = []
        valid_gt = {u: [] for u in range(self.num_users)}

        for u_idx in range(self.num_users):
            user_df = df_enc[df_enc["user"] == u_idx].sort_values("time")
            user_items = user_df["item"].tolist()
            n_items = len(user_items)

            if n_items == 0:
                continue

            # temporal_split_ratio 기준으로 분할 (예: 0.8 = 80% train, 20% valid)
            split_idx = max(1, int(n_items * self.temporal_split_ratio))

            train_items = user_items[:split_idx]
            valid_items = user_items[split_idx:]

            # validation ground truth 저장
            valid_gt[u_idx] = valid_items

            # train 데이터 저장
            for it in train_items:
                train_rows.append({"user": u_idx, "item": it})

        train_df = pd.DataFrame(train_rows)
        return train_df, valid_gt

    def _train_valid_split_temporal_global(self, df_enc, df_original):
        """전역 시간 기반 분할: 전체 데이터를 시간순으로 정렬 후 분할"""
        if "time" not in df_original.columns:
            raise ValueError("Temporal global split requires 'time' column in the data")

        # df_enc에 timestamp 추가
        df_enc = df_enc.copy()
        df_enc["time"] = df_original["time"].values

        # 전체 데이터를 시간순으로 정렬
        df_sorted = df_enc.sort_values("time")

        # temporal_split_ratio 기준으로 분할점 계산
        split_idx = int(len(df_sorted) * self.temporal_split_ratio)

        # Train/Valid 분할
        train_df = df_sorted.iloc[:split_idx][["user", "item"]]
        valid_df = df_sorted.iloc[split_idx:][["user", "item"]]

        # Validation ground truth 구성 (user별로 그룹화)
        valid_gt = {u: [] for u in range(self.num_users)}
        for _, row in valid_df.iterrows():
            user_idx = row["user"]
            item_idx = row["item"]
            valid_gt[user_idx].append(item_idx)

        return train_df, valid_gt

    def _build_user_item_matrix(self, df):
        """DataFrame -> Sparse CSR Matrix (num_users, num_items)"""
        rows = df["user"].values
        cols = df["item"].values
        data = np.ones(len(df))

        mat = sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(self.num_users, self.num_items),
            dtype=np.float32,
        )

        return mat

    def train_dataloader(self):
        """학습용 DataLoader: CSR matrix를 dense tensor로 변환하여 배치 생성"""
        # CSR -> Dense Tensor
        train_dense = torch.FloatTensor(self.train_mat.toarray())
        dataset = TensorDataset(train_dense)

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

    def val_dataloader(self):
        """검증용 DataLoader: 동일한 데이터 사용 (validation은 메트릭 계산으로 평가)"""
        train_dense = torch.FloatTensor(self.train_mat.toarray())
        dataset = TensorDataset(train_dense)

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

    def get_validation_ground_truth(self):
        """Validation ground truth 반환 (메트릭 계산용)"""
        return self.valid_gt

    def get_train_matrix(self):
        """학습 데이터 행렬 반환 (추천 생성 시 사용)"""
        return self.train_mat

    def get_full_matrix(self):
        """전체 데이터 행렬 반환 (train + validation, submission 생성 시 사용)"""
        # train + validation 데이터를 모두 포함한 sparse matrix 생성
        rows = []
        cols = []

        # train 데이터 추가
        train_rows, train_cols = self.train_mat.nonzero()
        rows.extend(train_rows.tolist())
        cols.extend(train_cols.tolist())

        # validation 데이터 추가
        for u_idx, items in self.valid_gt.items():
            for item_idx in items:
                rows.append(u_idx)
                cols.append(item_idx)

        # sparse matrix 생성
        data = np.ones(len(rows))
        full_mat = sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(self.num_users, self.num_items),
            dtype=np.float32,
        )

        return full_mat
