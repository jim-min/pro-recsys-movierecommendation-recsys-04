import os
import numpy as np
import pandas as pd
import torch


def read_interactions(data_dir, filename="train_ratings.csv"):
    path = os.path.join(data_dir, filename)
    df = pd.read_csv(path)

    if "user" not in df.columns or "item" not in df.columns:
        raise ValueError("train_ratings.csv must have columns: user, item")

    if "time" not in df.columns:
        df["time"] = 0

    df["user"] = df["user"].astype(int)
    df["item"] = df["item"].astype(int)
    df["time"] = df["time"].astype(int)
    return df


def encode_ids(df: pd.DataFrame):
    users = df["user"].values
    items = df["item"].values

    uniq_users = np.unique(users)
    uniq_items = np.unique(items)

    user2idx = {u: i for i, u in enumerate(uniq_users)}
    item2idx = {it: j for j, it in enumerate(uniq_items)}

    idx2user = {i: u for u, i in user2idx.items()}
    idx2item = {j: it for it, j in item2idx.items()}

    df_enc = pd.DataFrame({
        "user_idx": np.array([user2idx[u] for u in users], dtype=np.int64),
        "item_idx": np.array([item2idx[it] for it in items], dtype=np.int64),
        "time": df["time"].values.astype(np.int64),
    })
    return df_enc, user2idx, idx2user, item2idx, idx2item


def build_user_sequences(df_enc: pd.DataFrame, num_users: int):
    df_sorted = df_enc.sort_values(["user_idx", "time"], ascending=[True, True])
    groups = df_sorted.groupby("user_idx")["item_idx"].apply(list)

    seqs = [[] for _ in range(num_users)]
    for u, items in groups.items():
        seqs[int(u)] = list(map(int, items))
    return seqs


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
