import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack


def read_interactions(data_dir, filename="train_ratings.csv"):
    path = os.path.join(data_dir, filename)
    df = pd.read_csv(path)
    required = ["user", "item"]
    if not all(c in df.columns for c in required):
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
    """
    기본 Interaction Matrix (Binary or Frequency)
    Time Decay 없이 순수 클릭/시청 여부만 반영
    """
    rows = df_enc["user_idx"].values
    cols = df_enc["item_idx"].values
    data = np.ones(len(df_enc), dtype=np.float32)

    mat = csr_matrix((data, (rows, cols)), shape=(num_users, num_items))
    
    # 중복 Interaction이 있다면 1로 유지하거나 더함 (여기선 sum_duplicates로 빈도 반영)
    mat.sum_duplicates()
    return mat


def load_metadata_matrix(data_dir, item2idx, num_items):
    """
    Side Information (tsv files) 로드
    Returns:
        dict[str, csr_matrix]
        {
            "genres":    (num_items x num_genres),
            "directors": (num_items x num_directors),
            "writers":   (num_items x num_writers),
            "years":     (num_items x num_years),
            "titles":    (num_items x vocab_size)  # TF-IDF
        }
    """
    import os
    import numpy as np
    import pandas as pd
    from scipy.sparse import csr_matrix
    from sklearn.feature_extraction.text import TfidfVectorizer

    print("📂 Loading Metadata (Separated + TF-IDF titles)...")

    files = {
        "genres": "genres.tsv",
        "directors": "directors.tsv",
        "writers": "writers.tsv",
        "years": "years.tsv",
        "titles": "titles.tsv",   # 🔥 titles 추가
    }

    meta_mats = {}

    for name, filename in files.items():
        path = os.path.join(data_dir, filename)
        if not os.path.exists(path):
            print(f"⚠️ {filename} not found. Skipping.")
            continue

        # ===============================
        # 🔥 Titles: TF-IDF 처리
        # ===============================
        if name == "titles":
            df = pd.read_csv(path, sep="\t")
            df.columns = ["item", "title"]

            df = df[df["item"].isin(item2idx)]
            if len(df) == 0:
                print("⚠️ titles: no valid rows.")
                continue

            item_indices = df["item"].map(item2idx).values
            titles = df["title"].astype(str).values

            vectorizer = TfidfVectorizer(
                max_features=3000,
                ngram_range=(1, 2),
                stop_words="english",
                norm="l2",
            )

            tfidf = vectorizer.fit_transform(titles)

            # item index 기준으로 full matrix 생성
            full_mat = csr_matrix((num_items, tfidf.shape[1]))
            full_mat[item_indices] = tfidf

            meta_mats["titles"] = full_mat
            print(f"   ✅ titles (tfidf): {full_mat.shape}")
            continue

        # ===============================
        # 기존 categorical metadata
        # ===============================
        df = pd.read_csv(path, sep="\t")
        df.columns = ["item", "value"]

        df = df[df["item"].isin(item2idx)]
        if len(df) == 0:
            print(f"⚠️ {name}: no valid rows.")
            continue

        row_idx = df["item"].map(item2idx).values
        values = df["value"].astype(str).values

        uniq_vals = np.unique(values)
        val2idx = {v: i for i, v in enumerate(uniq_vals)}
        col_idx = np.array([val2idx[v] for v in values])

        data = np.ones(len(row_idx), dtype=np.float32)
        mat = csr_matrix(
            (data, (row_idx, col_idx)),
            shape=(num_items, len(uniq_vals)),
        )

        meta_mats[name] = mat
        print(f"   ✅ {name}: {mat.shape}")

    if not meta_mats:
        print("⚠️ No metadata loaded.")
        return None

    return meta_mats




def set_seed(seed):
    import random, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_valid_split_random(df_enc, num_users, min_interactions=5, valid_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    user_groups = df_enc.groupby("user_idx")["item_idx"].apply(list)
    train_rows = []
    valid_dict = {u: [] for u in range(num_users)}

    for u, items in user_groups.items():
        if len(items) < min_interactions:
            for it in items:
                train_rows.append((u, it))
            continue
        n_items = len(items)
        n_valid = int(n_items * valid_ratio)

        # 🔒 안전장치
        if n_items <= 1 or n_valid == 0:
            continue

        n_valid = min(n_valid, n_items - 1)

        valid_items = rng.choice(items, size=n_valid, replace=False)

        valid_set = set(valid_items)
        for it in items:
            if it in valid_set:
                valid_dict[u].append(it)
            else:
                train_rows.append((u, it))
    
    train_df = pd.DataFrame(train_rows, columns=["user_idx", "item_idx"])
    return train_df, valid_dict


def build_user_positives(mat):
    positives = []
    for u in range(mat.shape[0]):
        start, end = mat.indptr[u], mat.indptr[u + 1]
        positives.append(mat.indices[start:end])
    return positives