import os
import pandas as pd
import numpy as np
from scipy.stats import entropy

# =========================
# 1. CONFIG
# =========================
BASE_DIR = "/data/ephemeral/home/data/raw/train"

OUT_PATH = "ensemble_top10_user_adaptive.csv"

# =========================
# 2. LOAD DATA
# =========================

# main interaction
train = pd.read_csv(f"{BASE_DIR}/train_ratings.csv")

# item side information (TSV)
genres = pd.read_csv(f"{BASE_DIR}/genres.tsv", sep="\t")       # item, genre
directors = pd.read_csv(f"{BASE_DIR}/directors.tsv", sep="\t")# item, director
writers = pd.read_csv(f"{BASE_DIR}/writers.tsv", sep="\t")    # item, writer
titles = pd.read_csv(f"{BASE_DIR}/titles.tsv", sep="\t")      # item, title
years = pd.read_csv(f"{BASE_DIR}/years.tsv", sep="\t")        # item, year

# model outputs (Top-10 only, user,item 순서 중요)
ease = pd.read_csv("/data/ephemeral/home/Seung/output/EASE/vanila_EASE.csv")
bert = pd.read_csv("/data/ephemeral/home/Seung/output/EASE/output (3).csv")

# =========================
# 3. TOP-K ORDER → SCORE
# =========================
def add_score_from_order(df, topk=10):
    df = df.copy()
    df["_order"] = df.groupby("user").cumcount()
    df["score"] = (topk - df["_order"]) / topk
    return df.drop(columns="_order")

ease = add_score_from_order(ease, topk=10)
bert = add_score_from_order(bert, topk=10)

# =========================
# 4. USER BASIC FEATURES
# =========================
train["time"] = pd.to_datetime(train["time"], unit="s")

user_basic = train.groupby("user").agg(
    interaction_count=("item", "count"),
    active_days=("time", lambda x: (x.max() - x.min()).days + 1),
    avg_time_gap=("time", lambda x: x.sort_values().diff().dt.total_seconds().mean())
).reset_index()



user_basic = user_basic.merge(revisit, on="user", how="left").fillna(0)

# =========================
# 5. ITEM SIDE INFO → USER
# =========================
item_meta = (
    train[["user", "item"]]
    .merge(genres, on="item", how="left")
    .merge(directors, on="item", how="left")
    .merge(years, on="item", how="left")
)

def calc_entropy(series):
    p = series.value_counts(normalize=True)
    return entropy(p)

user_item_feat = item_meta.groupby("user").agg(
    genre_entropy=("genre", calc_entropy),
    director_entropy=("director", calc_entropy),
    year_std=("year", "std")
).reset_index().fillna(0)

# =========================
# 6. USER FEATURE TABLE
# =========================
user_feat = user_basic.merge(user_item_feat, on="user", how="left").fillna(0)

# standardize
for col in user_feat.columns:
    if col != "user":
        user_feat[col] = (
            (user_feat[col] - user_feat[col].mean()) /
            (user_feat[col].std() + 1e-9)
        )

# =========================
# 7. COMPUTE α_u (BERT weight, EASE biased)
# =========================
user_feat["alpha_raw"] = (
    0.25 * user_feat["interaction_count"] +
    0.25 * (1 - user_feat["genre_entropy"]) +
    0.10 * (1 - user_feat["director_entropy"])
    - 0.6    # 🔥 EASE 우선 bias
)

user_feat["alpha"] = 1 / (1 + np.exp(-user_feat["alpha_raw"]))
user_feat["alpha"] = user_feat["alpha"].clip(0.05, 0.40)

alpha_map = user_feat.set_index("user")["alpha"].to_dict()

# =========================
# 8. USER-ADAPTIVE ENSEMBLE
# =========================
final_rows = []

users = set(ease.user) & set(bert.user)

for u in users:
    alpha = alpha_map.get(u, 0.15)  # 기본도 EASE 쪽

    e_u = ease[ease.user == u][["item", "score"]]
    b_u = bert[bert.user == u][["item", "score"]]

    merged = (
        pd.merge(e_u, b_u, on="item", how="outer",
                 suffixes=("_ease", "_bert"))
        .fillna(0)
    )

    merged["final_score"] = (
        alpha * merged["score_bert"] +
        (1 - alpha) * merged["score_ease"]
    )

    top10 = merged.sort_values("final_score", ascending=False).head(10)

    for _, r in top10.iterrows():
        final_rows.append({
            "user": u,
            "item": r["item"],
            "final_score": r["final_score"],
            "alpha_bert": alpha
        })

final_df = pd.DataFrame(final_rows)
final_df.to_csv(OUT_PATH, index=False)

print(f"✅ Saved: {OUT_PATH}")
