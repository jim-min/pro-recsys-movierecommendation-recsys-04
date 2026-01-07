import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def read_interactions(data_dir, filename="train_ratings.csv"):
    path = os.path.join(data_dir, filename)
    df = pd.read_csv(path)
    if "user" not in df.columns or "item" not in df.columns:
        raise ValueError("train_ratings.csv must have columns: user, item")
    return df


def encode_ids(df):
    users = df["user"].astype(int).values
    items = df["item"].astype(int).values

    uniq_users = np.unique(users)
    uniq_items = np.unique(items)

    user2idx = {u: i for i, u in enumerate(uniq_users)}
    item2idx = {i: j for j, i in enumerate(uniq_items)}

    idx2user = {i: u for u, i in user2idx.items()}
    idx2item = {j: i for i, j in item2idx.items()}

    user_idx = np.array([user2idx[u] for u in users], dtype=np.int64)
    item_idx = np.array([item2idx[i] for i in items], dtype=np.int64)

    df_enc = pd.DataFrame({"user_idx": user_idx, "item_idx": item_idx})
    return df_enc, user2idx, idx2user, item2idx, idx2item


def build_user_item_matrix(df_enc, num_users, num_items):
    rows = df_enc["user_idx"].values
    cols = df_enc["item_idx"].values
    data = np.ones(len(df_enc), dtype=np.float32)
    mat = csr_matrix((data, (rows, cols)), shape=(num_users, num_items))
    mat.sum_duplicates()
    return mat


def build_user_positives(mat):
    positives = []
    for u in range(mat.shape[0]):
        start, end = mat.indptr[u], mat.indptr[u + 1]
        positives.append(mat.indices[start:end])
    return positives


def set_seed(seed):
    import random, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_valid_split_random(
    df_enc,
    num_users,
    min_interactions=5,
    valid_ratio=0.2,
    seed=42,
):
    """
    Random interaction masking for validation
    - 유저별 interaction 중 일부를 valid로 분리
    """
    rng = np.random.default_rng(seed)

    user_groups = df_enc.groupby("user_idx")["item_idx"].apply(list)

    train_rows = []
    valid_dict = {u: [] for u in range(num_users)}

    for u, items in user_groups.items():
        if len(items) < min_interactions:
            # interaction 적은 유저는 전부 train
            for it in items:
                train_rows.append((u, it))
            continue

        n_valid = max(1, int(len(items) * valid_ratio))
        valid_items = rng.choice(items, size=n_valid, replace=False)
        valid_set = set(valid_items)

        for it in items:
            if it in valid_set:
                valid_dict[u].append(it)
            else:
                train_rows.append((u, it))

    train_df = pd.DataFrame(train_rows, columns=["user_idx", "item_idx"])
    return train_df, valid_dict
