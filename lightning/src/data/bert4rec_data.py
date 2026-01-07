import os
import logging
import numpy as np
import pandas as pd
import lightning as L
import torch
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader

log = logging.getLogger(__name__)


class BERT4RecDataset(Dataset):
    """
    Dataset for BERT4Rec training

    Applies BERT-style masking to sequences during training
    """

    def __init__(
        self,
        user_sequences,
        num_items,
        max_len,
        random_mask_prob,
        mask_token,
        pad_token,
        last_item_mask_ratio=0.2,
        sampling_strategy="recent",
        item_genres=None,
        item_directors=None,
        item_writers=None,
        item_title_embeddings=None,
        title_embedding_dim=0,
    ):
        """
        Args:
            user_sequences: Dict[user_id, List[item_id]] - User interaction sequences
            num_items: Number of unique items
            max_len: Maximum sequence length
            random_mask_prob: Probability of masking items (for random masking)
            mask_token: Token ID for [MASK]
            pad_token: Token ID for padding
            last_item_mask_ratio: Probability of additionally masking the last item on top of random masking (default: 0.2)
                                  This boosts next-item prediction while maintaining data diversity
            sampling_strategy: Strategy for truncating sequences ("recent" or "weighted")
                - "recent": Take the most recent max_len items (default)
                - "weighted": Sample max_len items using recency-weighted probabilities
            item_genres: Dict[item_idx, List[genre_idx]] - Item genre mappings
            item_directors: Dict[item_idx, director_idx] - Item director mappings
            item_writers: Dict[item_idx, List[writer_idx]] - Item writer mappings
            item_title_embeddings: Dict[item_idx, np.array] - Pre-computed title embeddings
            title_embedding_dim: Dimension of title embeddings
        """
        self.user_sequences = user_sequences
        self.num_items = num_items
        self.max_len = max_len
        self.random_mask_prob = random_mask_prob
        self.mask_token = mask_token
        self.pad_token = pad_token
        self.last_item_mask_ratio = last_item_mask_ratio
        self.sampling_strategy = sampling_strategy

        # Metadata
        self.item_genres = item_genres or {}
        self.item_directors = item_directors or {}
        self.item_writers = item_writers or {}
        self.item_title_embeddings = item_title_embeddings or {}
        self.title_embedding_dim = title_embedding_dim

        # Check if any metadata is actually enabled
        self.has_metadata = bool(
            self.item_genres or self.item_directors or
            self.item_writers or self.item_title_embeddings
        )

        # Get list of users for indexing
        self.users = list(user_sequences.keys())

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        """
        Returns:
            tokens: [max_len] - Masked sequence
            labels: [max_len] - Labels for loss (0 for non-masked positions)
            metadata: Dict with metadata tensors (if available)
        """
        user = self.users[idx]
        seq = self.user_sequences[user]

        # 1. First truncate/sample the sequence (following BERT4Rec paper)
        if self.sampling_strategy == "weighted":
            seq = self._weighted_sample_sequence(seq)
        else:  # "recent" (default)
            seq = seq[-self.max_len :]

        # 2. Then apply random masking (for data diversity)
        tokens, labels = self._random_mask_sequence(seq)

        # 3. Additionally mask the last item with probability last_item_mask_ratio
        # This boosts next-item prediction performance while maintaining data diversity
        if len(seq) > 0 and np.random.random() < self.last_item_mask_ratio:
            # Force mask the last item (overwrite if already masked)
            last_idx = len(seq) - 1
            tokens[last_idx] = self.mask_token
            labels[last_idx] = seq[last_idx]  # Original item as label

        # 4. Pad if necessary
        pad_len = self.max_len - len(tokens)
        if pad_len > 0:
            tokens = [self.pad_token] * pad_len + tokens
            labels = [self.pad_token] * pad_len + labels

        # Prepare metadata only if enabled
        metadata = self._prepare_metadata(tokens) if self.has_metadata else {}

        return torch.LongTensor(tokens), torch.LongTensor(labels), metadata

    def _weighted_sample_sequence(self, seq):
        """
        Sample items from sequence using recency-weighted sampling without replacement

        This method samples max_len items from the sequence where:
        - More recent items have higher probability (linearly increasing: 1, 2, 3, ...)
        - Original order is preserved (no shuffling)
        - No duplicates (each item sampled at most once)

        Probability formula for linear weights:
        - Weight of position i: i (where i = 1, 2, 3, ..., seq_len)
        - Sum of weights: seq_len * (seq_len + 1) / 2
        - Probability of position i: i / (seq_len * (seq_len + 1) / 2)

        Args:
            seq: List[int] - Original sequence

        Returns:
            List[int] - Sampled sequence maintaining original order
        """
        seq_len = len(seq)

        # If sequence is shorter than max_len, return as is
        if seq_len <= self.max_len:
            return seq

        # Create recency-weighted probabilities (1, 2, 3, ..., seq_len)
        # More recent items (at the end) have higher weights
        # Using formula: weight_sum = n(n+1)/2
        weights = np.arange(1, seq_len + 1, dtype=np.float64)
        probabilities = weights / (seq_len * (seq_len + 1) / 2)

        # Sample indices without replacement
        sampled_indices = np.random.choice(
            seq_len, size=self.max_len, replace=False, p=probabilities
        )

        # Sort indices to maintain original order
        sampled_indices = np.sort(sampled_indices)

        # Extract items at sampled indices
        sampled_seq = [seq[i] for i in sampled_indices]

        return sampled_seq

    def _prepare_metadata(self, tokens):
        """
        Prepare metadata tensors for a sequence

        Args:
            tokens: List[int] - Token sequence (already padded/truncated)

        Returns:
            Dict with metadata tensors
        """
        metadata = {}
        max_genres = 5  # Maximum number of genres per item
        max_writers = 5  # Maximum number of writers per item

        # Genres (multi-hot, padded)
        if self.item_genres:
            genre_batch = []
            for item_idx in tokens:
                if item_idx in self.item_genres:
                    genres = self.item_genres[item_idx][:max_genres]  # truncate
                    genres = genres + [0] * (max_genres - len(genres))  # pad
                else:
                    genres = [0] * max_genres
                genre_batch.append(genres)
            metadata["genres"] = torch.LongTensor(genre_batch)  # [seq_len, max_genres]

        # Directors (single value per item)
        if self.item_directors:
            director_batch = [
                self.item_directors.get(item_idx, 0) for item_idx in tokens
            ]
            metadata["directors"] = torch.LongTensor(director_batch)  # [seq_len]

        # Writers (multi-hot, padded)
        if self.item_writers:
            writer_batch = []
            for item_idx in tokens:
                if item_idx in self.item_writers:
                    writers = self.item_writers[item_idx][:max_writers]  # truncate
                    writers = writers + [0] * (max_writers - len(writers))  # pad
                else:
                    writers = [0] * max_writers
                writer_batch.append(writers)
            metadata["writers"] = torch.LongTensor(
                writer_batch
            )  # [seq_len, max_writers]

        # Title embeddings (pre-computed)
        if self.item_title_embeddings and self.title_embedding_dim > 0:
            title_batch = []
            for item_idx in tokens:
                if item_idx in self.item_title_embeddings:
                    title_batch.append(self.item_title_embeddings[item_idx])
                else:
                    title_batch.append(np.zeros(self.title_embedding_dim))
            metadata["title_embs"] = torch.FloatTensor(
                np.array(title_batch)
            )  # [seq_len, title_dim]

        return metadata

    def _random_mask_sequence(self, seq):
        """
        Apply BERT-style random masking to sequence

        Args:
            seq: List[int] - Original sequence
        Returns:
            tokens: List[int] - Masked sequence
            labels: List[int] - Labels (0 for non-masked, original item for masked)
        """
        tokens = []
        labels = []

        for item in seq:
            prob = np.random.random()

            if prob < self.random_mask_prob:
                # This position will be masked
                prob /= self.random_mask_prob

                if prob < 0.8:
                    # 80%: Replace with [MASK]
                    tokens.append(self.mask_token)
                elif prob < 0.9:
                    # 10%: Replace with random item (excluding pad)
                    tokens.append(np.random.randint(1, self.num_items + 1))
                else:
                    # 10%: Keep original
                    tokens.append(item)

                labels.append(item)  # Original item as label
            else:
                # Not masked
                tokens.append(item)
                labels.append(self.pad_token)  # Ignore in loss (label = 0)

        return tokens, labels

    def _mask_last_item(self, seq):
        """
        Mask only the last item in the sequence (for next item prediction)

        Args:
            seq: List[int] - Original sequence
        Returns:
            tokens: List[int] - Sequence with last item masked
            labels: List[int] - Labels (0 except for last position)
        """
        if len(seq) == 0:
            return [], []

        tokens = list(seq)
        labels = [self.pad_token] * len(seq)

        # Mask the last item
        last_item = tokens[-1]
        tokens[-1] = self.mask_token
        labels[-1] = last_item

        return tokens, labels


class BERT4RecValidationDataset(Dataset):
    """
    Dataset for BERT4Rec validation

    Each sample includes:
    - Full sequence with [MASK] at the end
    - Ground truth item (next item to predict)
    """

    def __init__(
        self,
        user_sequences,
        user_targets,
        num_items,
        max_len,
        mask_token,
        pad_token,
        item_genres=None,
        item_directors=None,
        item_writers=None,
        item_title_embeddings=None,
        title_embedding_dim=0,
    ):
        """
        Args:
            user_sequences: Dict[user_id, List[item_id]] - User training sequences
            user_targets: Dict[user_id, item_id] - Ground truth items to predict
            num_items: Number of unique items
            max_len: Maximum sequence length
            mask_token: Token ID for [MASK]
            pad_token: Token ID for padding
            item_genres: Dict[item_idx, List[genre_idx]] - Item genre mappings
            item_directors: Dict[item_idx, director_idx] - Item director mappings
            item_writers: Dict[item_idx, List[writer_idx]] - Item writer mappings
            item_title_embeddings: Dict[item_idx, np.array] - Pre-computed title embeddings
            title_embedding_dim: Dimension of title embeddings
        """
        self.user_sequences = user_sequences
        self.user_targets = user_targets
        self.num_items = num_items
        self.max_len = max_len
        self.mask_token = mask_token
        self.pad_token = pad_token

        # Metadata
        self.item_genres = item_genres or {}
        self.item_directors = item_directors or {}
        self.item_writers = item_writers or {}
        self.item_title_embeddings = item_title_embeddings or {}
        self.title_embedding_dim = title_embedding_dim

        # Check if any metadata is actually enabled
        self.has_metadata = bool(
            self.item_genres or self.item_directors or
            self.item_writers or self.item_title_embeddings
        )

        self.users = list(user_sequences.keys())

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        """
        Returns:
            tokens: [max_len] - Sequence with [MASK] at the end
            labels: [max_len] - Dummy labels (not used in validation)
            metadata: Dict with metadata tensors
            target: int - Ground truth item
        """
        user = self.users[idx]
        seq = self.user_sequences[user]
        target = self.user_targets[user]

        # Add [MASK] token at the end
        tokens = (list(seq) + [self.mask_token])[-self.max_len :]

        # Pad if necessary
        pad_len = self.max_len - len(tokens)
        if pad_len > 0:
            tokens = [self.pad_token] * pad_len + tokens

        # Dummy labels (all zeros, not used in validation)
        labels = [self.pad_token] * self.max_len

        # Prepare metadata only if enabled
        metadata = self._prepare_metadata(tokens) if self.has_metadata else {}

        return (
            torch.LongTensor(tokens),
            torch.LongTensor(labels),
            metadata,
            torch.LongTensor([target]),
        )

    def _prepare_metadata(self, tokens):
        """
        Prepare metadata tensors for a sequence
        (Same implementation as BERT4RecDataset)

        Args:
            tokens: List[int] - Token sequence (already padded/truncated)

        Returns:
            Dict with metadata tensors
        """
        metadata = {}
        max_genres = 5
        max_writers = 5

        # Genres
        if self.item_genres:
            genre_batch = []
            for item_idx in tokens:
                if item_idx in self.item_genres:
                    genres = self.item_genres[item_idx][:max_genres]
                    genres = genres + [0] * (max_genres - len(genres))
                else:
                    genres = [0] * max_genres
                genre_batch.append(genres)
            metadata["genres"] = torch.LongTensor(genre_batch)

        # Directors
        if self.item_directors:
            director_batch = [
                self.item_directors.get(item_idx, 0) for item_idx in tokens
            ]
            metadata["directors"] = torch.LongTensor(director_batch)

        # Writers
        if self.item_writers:
            writer_batch = []
            for item_idx in tokens:
                if item_idx in self.item_writers:
                    writers = self.item_writers[item_idx][:max_writers]
                    writers = writers + [0] * (max_writers - len(writers))
                else:
                    writers = [0] * max_writers
                writer_batch.append(writers)
            metadata["writers"] = torch.LongTensor(writer_batch)

        # Title embeddings
        if self.item_title_embeddings and self.title_embedding_dim > 0:
            title_batch = []
            for item_idx in tokens:
                if item_idx in self.item_title_embeddings:
                    title_batch.append(self.item_title_embeddings[item_idx])
                else:
                    title_batch.append(np.zeros(self.title_embedding_dim))
            metadata["title_embs"] = torch.FloatTensor(np.array(title_batch))

        return metadata


class BERT4RecDataModule(L.LightningDataModule):
    """
    LightningDataModule for BERT4Rec

    Hydra 설정 사용:
        data:
            data_dir: "~/data/train/"
            data_file: "train_ratings.csv"
            batch_size: 128
            max_len: 50
            random_mask_prob: 0.15
            last_item_mask_ratio: 0.2
            sampling_strategy: "recent"  # or "weighted"
            min_interactions: 3
            seed: 42
            num_workers: 4

    Data Format:
        CSV with columns: user, item, time (optional)
    """

    def __init__(
        self,
        data_dir: str,
        data_file: str = "train_ratings.csv",
        batch_size: int = 128,
        max_len: int = 50,
        random_mask_prob: float = 0.2,
        last_item_mask_ratio: float = 0.0,
        sampling_strategy: str = "recent",
        min_interactions: int = 3,
        seed: int = 42,
        num_workers: int = 4,
        use_full_data: bool = False,
    ):
        super().__init__()
        self.data_dir = os.path.expanduser(data_dir)
        self.data_file = data_file
        self.batch_size = batch_size
        self.max_len = max_len
        self.random_mask_prob = random_mask_prob
        self.last_item_mask_ratio = last_item_mask_ratio
        self.sampling_strategy = sampling_strategy
        self.min_interactions = min_interactions
        self.seed = seed
        self.num_workers = num_workers
        self.use_full_data = use_full_data

        # Special tokens
        self.pad_token = 0
        self.mask_token = None  # Will be set to num_items + 1

        # Data will be loaded in setup()
        self.num_users = None
        self.num_items = None
        self.user_train = None
        self.user_valid = None
        self.item2idx = None
        self.user2idx = None
        self.idx2item = None
        self.idx2user = None

        # Item release years (indexed) - item_years: Dict[item_idx, year]
        self.item_years = {}

        # User's last click years - user_last_click_years: Dict[user_idx, year]
        self.user_last_click_years = {}

    def prepare_data(self):
        """
        Download or prepare data (called only on 1 GPU/TPU)
        For BERT4Rec, we just check if data file exists
        """
        data_path = os.path.join(self.data_dir, self.data_file)
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")

    def setup(self, stage=None):
        """
        Load and split data (called on every GPU/TPU)

        Args:
            stage: 'fit', 'validate', 'test', or 'predict'
        """
        # Load data
        data_path = os.path.join(self.data_dir, self.data_file)
        log.info(f"Loading data from {data_path}")

        df = pd.read_csv(data_path)
        log.info(f"Data loaded. Shape: {df.shape}")

        # Check required columns
        if "user" not in df.columns or "item" not in df.columns:
            raise ValueError("Data must have 'user' and 'item' columns")

        # Get unique items and users
        item_ids = df["item"].unique()
        user_ids = df["user"].unique()

        # Create mappings (item: 1~num_items, user: 0~num_users-1)
        self.item2idx = pd.Series(data=np.arange(len(item_ids)) + 1, index=item_ids)
        self.user2idx = pd.Series(data=np.arange(len(user_ids)), index=user_ids)

        # Reverse mappings
        self.idx2item = pd.Series(index=self.item2idx.values, data=self.item2idx.index)
        self.idx2user = pd.Series(index=self.user2idx.values, data=self.user2idx.index)

        self.num_items = len(item_ids)
        self.num_users = len(user_ids)
        self.mask_token = self.num_items + 1

        log.info(f"num_users: {self.num_users}, num_items: {self.num_items}")

        # Re-index dataframe
        df["item_idx"] = df["item"].map(self.item2idx)
        df["user_idx"] = df["user"].map(self.user2idx)

        # Sort by user and time (if available)
        if "time" in df.columns:
            df = df.sort_values(["user_idx", "time"])
        else:
            df = df.sort_values(["user_idx"])

        # Group by user
        user_sequences = defaultdict(list)
        for user_idx, item_idx in zip(df["user_idx"], df["item_idx"]):
            user_sequences[user_idx].append(item_idx)

        # Filter users with minimum interactions
        user_sequences = {
            u: seq
            for u, seq in user_sequences.items()
            if len(seq) >= self.min_interactions
        }

        log.info(
            f"Users after filtering (min_interactions={self.min_interactions}): {len(user_sequences)}"
        )

        # Split: last item for validation, rest for training
        self.user_train = {}
        self.user_valid = {}

        for user, seq in user_sequences.items():
            if self.use_full_data:
                # train data = full data, valid data = dummy
                self.user_train[user] = seq
                self.user_valid[user] = seq[-1]  # Dummy
            else:
                # train, validation split
                self.user_train[user] = seq[:-1]
                self.user_valid[user] = seq[-1]  # Last item

        log.info(f"Train sequences: {len(self.user_train)}")
        log.info(f"Valid sequences: {len(self.user_valid)}")

        # Update num_users to filtered count
        self.num_users = len(self.user_train)

        # Load item metadata (pass df to avoid re-reading)
        log.info("Setting up Item meta-data ...")
        self._load_item_metadata(df)
        log.info("Item meta-data setup complete")

    def _load_item_metadata(self, df):
        """
        Item 메타데이터 로드 (genres, directors, writers, titles, years)

        Args:
            df: Already loaded DataFrame with 'user', 'item', 'time' columns

        Loads item metadata and calculates user's last click years
        Sets:
            - self.item_years: Dict[item_idx, year]
            - self.user_last_click_years: Dict[user_idx, year]
            - self.item_genres: Dict[item_idx, List[genre_idx]]
            - self.item_directors: Dict[item_idx, director_idx]
            - self.item_writers: Dict[item_idx, List[writer_idx]]
            - self.item_title_embeddings: Dict[item_idx, np.array]
            - self.num_genres: int
            - self.num_directors: int
            - self.num_writers: int
            - self.title_embedding_dim: int
        """
        # Item release years (indexed) - item_years: Dict[item_idx, year]
        self.item_years = {}

        # User's last click years - user_last_click_years: Dict[user_idx, year]
        self.user_last_click_years = {}

        years_path = os.path.join(self.data_dir, "years.tsv")

        # Check if years.tsv exists
        if not os.path.exists(years_path):
            log.warning(
                f"years.tsv not found at {years_path}. Skipping years metadata."
            )
        else:
            try:
                # Load item release years
                years_df = pd.read_csv(years_path, sep="\t")
                log.info(f"Loaded {len(years_df)} items with release year info")

                # Create item_years mapping (original_item_id -> year)
                item_year_map = dict(zip(years_df["item"], years_df["year"]))

                # Convert to indexed mapping (item_idx -> year)
                for item_id, year in item_year_map.items():
                    if item_id in self.item2idx.index:
                        item_idx = self.item2idx[item_id]
                        self.item_years[item_idx] = year

                log.info(f"Mapped {len(self.item_years)} items to release years")

                # Calculate user's last click year from interaction data (use already loaded df)
                if "time" in df.columns:
                    # Convert timestamp to year
                    df_copy = df.copy()
                    df_copy["click_year"] = pd.to_datetime(
                        df_copy["time"], unit="s"
                    ).dt.year

                    # Map to user_idx (already done in setup, but just in case)
                    if "user_idx" not in df_copy.columns:
                        df_copy["user_idx"] = df_copy["user"].map(self.user2idx)

                    # Get last click year per user
                    user_last_years = (
                        df_copy.groupby("user_idx")["click_year"].max().to_dict()
                    )
                    self.user_last_click_years = user_last_years

                    log.info(
                        f"Calculated last click year for {len(self.user_last_click_years)} users"
                    )
                else:
                    log.warning(
                        "No 'time' column found. Cannot calculate last click years."
                    )

            except Exception as e:
                log.error(f"Error loading years metadata: {e}")
                # Initialize empty dicts on error
                self.item_years = {}
                self.user_last_click_years = {}

        # ===== Load Genres Metadata =====
        self.item_genres = {}
        self.genre2idx = {}
        self.num_genres = 1  # 0 is reserved for padding

        genres_path = os.path.join(self.data_dir, "genres.tsv")
        if os.path.exists(genres_path):
            try:
                genres_df = pd.read_csv(genres_path, sep="\t")
                log.info(f"Loaded {len(genres_df)} genre entries")

                # Build genre vocabulary
                unique_genres = genres_df["genre"].unique()
                self.genre2idx = {
                    genre: idx + 1 for idx, genre in enumerate(unique_genres)
                }
                self.num_genres = len(self.genre2idx) + 1  # +1 for padding

                # Build item->genres mapping (1:N relationship)
                for item_id, group in genres_df.groupby("item"):
                    if item_id in self.item2idx.index:
                        item_idx = self.item2idx[item_id]
                        genre_indices = [
                            self.genre2idx[g] for g in group["genre"].values
                        ]
                        self.item_genres[item_idx] = genre_indices

                log.info(
                    f"Loaded {self.num_genres-1} unique genres for {len(self.item_genres)} items"
                )
            except Exception as e:
                log.error(f"Error loading genres: {e}")
                self.item_genres = {}
                self.genre2idx = {}
                self.num_genres = 1
        else:
            log.warning(f"genres.tsv not found at {genres_path}")

        # ===== Load Directors Metadata =====
        self.item_directors = {}
        self.director2idx = {}
        self.num_directors = 1  # 0 is reserved for padding

        directors_path = os.path.join(self.data_dir, "directors.tsv")
        if os.path.exists(directors_path):
            try:
                directors_df = pd.read_csv(directors_path, sep="\t")
                log.info(f"Loaded {len(directors_df)} director entries")

                # Build director vocabulary
                unique_directors = directors_df["director"].unique()
                self.director2idx = {
                    d: idx + 1 for idx, d in enumerate(unique_directors)
                }
                self.num_directors = len(self.director2idx) + 1

                # Build item->director mapping (1:1 relationship)
                director_map = dict(zip(directors_df["item"], directors_df["director"]))
                for item_id, director_id in director_map.items():
                    if item_id in self.item2idx.index:
                        item_idx = self.item2idx[item_id]
                        self.item_directors[item_idx] = self.director2idx[director_id]

                log.info(
                    f"Loaded {self.num_directors-1} unique directors for {len(self.item_directors)} items"
                )
            except Exception as e:
                log.error(f"Error loading directors: {e}")
                self.item_directors = {}
                self.director2idx = {}
                self.num_directors = 1
        else:
            log.warning(f"directors.tsv not found at {directors_path}")

        # ===== Load Writers Metadata =====
        self.item_writers = {}
        self.writer2idx = {}
        self.num_writers = 1  # 0 is reserved for padding

        writers_path = os.path.join(self.data_dir, "writers.tsv")
        if os.path.exists(writers_path):
            try:
                writers_df = pd.read_csv(writers_path, sep="\t")
                log.info(f"Loaded {len(writers_df)} writer entries")

                # Build writer vocabulary
                unique_writers = writers_df["writer"].unique()
                self.writer2idx = {w: idx + 1 for idx, w in enumerate(unique_writers)}
                self.num_writers = len(self.writer2idx) + 1

                # Build item->writers mapping (1:N relationship)
                for item_id, group in writers_df.groupby("item"):
                    if item_id in self.item2idx.index:
                        item_idx = self.item2idx[item_id]
                        writer_indices = [
                            self.writer2idx[w] for w in group["writer"].values
                        ]
                        self.item_writers[item_idx] = writer_indices

                log.info(
                    f"Loaded {self.num_writers-1} unique writers for {len(self.item_writers)} items"
                )
            except Exception as e:
                log.error(f"Error loading writers: {e}")
                self.item_writers = {}
                self.writer2idx = {}
                self.num_writers = 1
        else:
            log.warning(f"writers.tsv not found at {writers_path}")

        # ===== Load Title Embeddings =====
        self.item_title_embeddings = {}
        self.title_embedding_dim = 0

        # Try loading from TSV file first (new format from preprocess_title_genre_embeddings.py)
        title_emb_tsv_path = os.path.join(self.data_dir, "title_embeddings/titles.tsv")

        if os.path.exists(title_emb_tsv_path):
            try:
                log.info(f"Loading title embeddings from {title_emb_tsv_path}")
                with open(title_emb_tsv_path, "r") as f:
                    # Skip header
                    next(f)

                    for line in f:
                        parts = line.strip().split("\t")
                        if len(parts) != 2:
                            continue

                        item_id = int(parts[0])
                        emb_values = np.array([float(x) for x in parts[1].split()])

                        # Set embedding dimension from first item
                        if self.title_embedding_dim == 0:
                            self.title_embedding_dim = len(emb_values)

                        # Convert to indexed mapping
                        if item_id in self.item2idx.index:
                            item_idx = self.item2idx[item_id]
                            self.item_title_embeddings[item_idx] = emb_values

                log.info(
                    f"Loaded {len(self.item_title_embeddings)} title embeddings (dim={self.title_embedding_dim})"
                )
            except Exception as e:
                log.error(f"Error loading title embeddings from TSV: {e}")
                log.warning(
                    "Title embeddings not found. Run scripts/preprocess_title_genre_embeddings.py first."
                )
                self.item_title_embeddings = {}
                self.title_embedding_dim = 0
        else:
            # Fallback: try loading from pickle file (legacy format)
            title_emb_pkl_path = os.path.join(
                self.data_dir, "title_embeddings/title_embeddings.pkl"
            )

            if os.path.exists(title_emb_pkl_path):
                try:
                    log.info(
                        f"Loading title embeddings from {title_emb_pkl_path} (legacy pickle format)"
                    )
                    import pickle

                    with open(title_emb_pkl_path, "rb") as f:
                        title_emb_dict = pickle.load(f)  # {original_item_id: embedding}

                    # Load metadata
                    metadata_path = os.path.join(
                        self.data_dir, "title_embeddings/metadata.pkl"
                    )
                    with open(metadata_path, "rb") as f:
                        metadata = pickle.load(f)
                    self.title_embedding_dim = metadata["embedding_dim"]

                    # Convert to indexed mapping
                    for item_id, emb in title_emb_dict.items():
                        if item_id in self.item2idx.index:
                            item_idx = self.item2idx[item_id]
                            self.item_title_embeddings[item_idx] = emb

                    log.info(
                        f"Loaded {len(self.item_title_embeddings)} title embeddings (dim={self.title_embedding_dim})"
                    )
                except Exception as e:
                    log.error(f"Error loading title embeddings from pickle: {e}")
                    self.item_title_embeddings = {}
                    self.title_embedding_dim = 0
            else:
                log.warning(
                    f"Title embeddings not found. Run scripts/preprocess_title_genre_embeddings.py first."
                )

    def train_dataloader(self):
        """Create training dataloader"""
        dataset = BERT4RecDataset(
            user_sequences=self.user_train,
            num_items=self.num_items,
            max_len=self.max_len,
            random_mask_prob=self.random_mask_prob,
            mask_token=self.mask_token,
            pad_token=self.pad_token,
            last_item_mask_ratio=self.last_item_mask_ratio,
            sampling_strategy=self.sampling_strategy,
            item_genres=self.item_genres,
            item_directors=self.item_directors,
            item_writers=self.item_writers,
            item_title_embeddings=self.item_title_embeddings,
            title_embedding_dim=self.title_embedding_dim,
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,  # 속도 향상
            prefetch_factor=2,  # 속도 향상
            collate_fn=self._collate_fn_with_metadata,
        )

    def _collate_fn_with_metadata(self, batch):
        """Custom collate function to handle metadata"""
        tokens = torch.stack([item[0] for item in batch])
        labels = torch.stack([item[1] for item in batch])

        # Stack metadata from all samples in batch (skip if empty)
        metadata_batch = {}
        if batch[0][2]:  # Check if metadata dict is not empty
            metadata_keys = batch[0][2].keys()
            for key in metadata_keys:
                metadata_batch[key] = torch.stack([item[2][key] for item in batch])

        return tokens, labels, metadata_batch

    def val_dataloader(self):
        """Create validation dataloader"""
        dataset = BERT4RecValidationDataset(
            user_sequences=self.user_train,
            user_targets=self.user_valid,
            num_items=self.num_items,
            max_len=self.max_len,
            mask_token=self.mask_token,
            pad_token=self.pad_token,
            item_genres=self.item_genres,
            item_directors=self.item_directors,
            item_writers=self.item_writers,
            item_title_embeddings=self.item_title_embeddings,
            title_embedding_dim=self.title_embedding_dim,
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,  # 속도 향상
            prefetch_factor=2,        # 속도 향상
            collate_fn=self._collate_fn_val_with_metadata,
        )

    def _collate_fn_val_with_metadata(self, batch):
        """Custom collate function for validation with metadata"""
        tokens = torch.stack([item[0] for item in batch])
        labels = torch.stack([item[1] for item in batch])
        targets = torch.cat([item[3] for item in batch])  # [batch_size]

        # Stack metadata (skip if empty)
        metadata_batch = {}
        if batch[0][2]:  # Check if metadata dict is not empty
            metadata_keys = batch[0][2].keys()
            for key in metadata_keys:
                metadata_batch[key] = torch.stack([item[2][key] for item in batch])

        return tokens, labels, metadata_batch, targets

    def get_user_sequence(self, user_id):
        """
        Get training sequence for a specific user (by original ID)

        Args:
            user_id: Original user ID
        Returns:
            List of item indices
        """
        user_idx = self.user2idx.get(user_id)
        if user_idx is None:
            return None
        return self.user_train.get(user_idx)

    def get_all_sequences(self):
        """
        Get all training sequences (for model input)

        Returns:
            Dict[user_idx, List[item_idx]]
        """
        return self.user_train

    def get_full_sequences(self):
        """
        Get full sequences including validation items (for exclusion in inference)

        Returns:
            Dict[user_idx, Set[item_idx]]
        """
        full_sequences = {}
        for user_idx in self.user_train.keys():
            # train + validation 아이템 모두 포함
            full_seq = self.user_train[user_idx] + [self.user_valid[user_idx]]
            full_sequences[user_idx] = full_seq
        return full_sequences

    def get_future_item_sequences(self):
        """
        Get future item sequences where item release year > user's last click year
        (for future information leakage analysis)

        Uses:
            - self.item_years: Dict[item_idx, year] - Item release years
            - self.user_last_click_years: Dict[user_idx, year] - User's last click years

        Returns:
            Dict[user_idx, Set[item_idx]] - Future items per user
        """
        future_item_sequences = {}

        for user_idx in self.user_train.keys():
            # Get full sequence (train + valid)
            full_seq = self.user_train[user_idx] + [self.user_valid[user_idx]]

            # Get user's last click year
            last_click_year = self.user_last_click_years.get(user_idx)
            if last_click_year is None:
                # No click year info for this user
                future_item_sequences[user_idx] = set()  # 빈 set으로 통일
                continue

            # Filter items where release year > last click year
            # 모든 아이템 중에서 future items 찾기 (full_seq가 아님!)
            future_items = set()
            for item_idx, item_year in self.item_years.items():
                if item_year > last_click_year:
                    future_items.add(item_idx)

            future_item_sequences[user_idx] = future_items

        return future_item_sequences

    def get_item_metadata(self):
        """
        Get item metadata mappings for inference

        Returns:
            Dict with metadata mappings or None if no metadata available:
            {
                'genres': Dict[item_idx, List[genre_idx]],
                'directors': Dict[item_idx, director_idx],
                'writers': Dict[item_idx, List[writer_idx]],
                'title_embs': Dict[item_idx, np.array]
            }
        """
        metadata = {}

        if self.item_genres:
            metadata["genres"] = self.item_genres
        if self.item_directors:
            metadata["directors"] = self.item_directors
        if self.item_writers:
            metadata["writers"] = self.item_writers
        if self.item_title_embeddings:
            metadata["title_embs"] = self.item_title_embeddings

        return metadata if metadata else None
