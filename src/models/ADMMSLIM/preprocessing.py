import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix

def load_and_preprocess_data(data_path, years_path, valid_ratio: float = 0.0, seed: int = 42):
    """
    데이터를 로드하고 전처리하는 함수.

    Args:
        data_path (str): 데이터셋이 저장된 경로.

    Returns:
        tuple: 전처리된 데이터프레임, 사용자-아이템 상호작용 행렬, 사용자 인코더, 아이템 인코더.

        - train_df (DataFrame): 사용자-아이템 상호작용 데이터프레임.
        - user_item_matrix (scipy.sparse.csr_matrix): 희소 형태의 사용자-아이템 상호작용 행렬.
        - user_encoder (LabelEncoder): 사용자 ID를 숫자로 변환하는 인코더.
        - item_encoder (LabelEncoder): 아이템 ID를 숫자로 변환하는 인코더.
    """
    train_df = pd.read_csv(data_path)
    train_df['watched'] = 1
    train_df = train_df.drop(columns=['time'])
    years_df = pd.read_csv(years_path, sep="\t")

    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()

    train_df['user_id'] = user_encoder.fit_transform(train_df['user'])
    train_df['item_id'] = item_encoder.fit_transform(train_df['item'])
    years_df['item_id'] = item_encoder.transform(years_df['item'])

    num_users = train_df['user_id'].nunique()
    num_items = train_df['item_id'].nunique()

    if valid_ratio and valid_ratio > 0:
        rng = np.random.default_rng(seed)
        valid_mask = np.zeros(len(train_df), dtype=bool)

        for _, idx in train_df.groupby('user_id').indices.items():
            idx = np.asarray(idx)
            if idx.size <= 1:
                continue
            n_valid = max(1, int(np.floor(idx.size * valid_ratio)))
            chosen = rng.choice(idx, size=n_valid, replace=False)
            valid_mask[chosen] = True

        train_part = train_df.loc[~valid_mask]
        valid_part = train_df.loc[valid_mask]

        train_interactions = train_part['watched'].values
        train_matrix = csr_matrix(
            (train_interactions, (train_part['user_id'].values, train_part['item_id'].values)),
            shape=(num_users, num_items)
        )

        valid_interactions = valid_part['watched'].values
        valid_matrix = csr_matrix(
            (valid_interactions, (valid_part['user_id'].values, valid_part['item_id'].values)),
            shape=(num_users, num_items)
        )

        return train_df, years_df, train_matrix, valid_matrix, user_encoder, item_encoder

    interactions = train_df['watched'].values
    user_item_matrix = csr_matrix(
        (interactions, (train_df['user_id'].values, train_df['item_id'].values)),
        shape=(num_users, num_items)
    )

    return train_df, years_df, user_item_matrix, None, user_encoder, item_encoder