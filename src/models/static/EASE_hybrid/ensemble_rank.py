import os
import pandas as pd
from collections import defaultdict

ROOT = "/data/ephemeral/home/Seung/output/EASE_Hybrid"
SEEDS = [42, 43, 44, 45]
TOPK = 10

user_item_score = defaultdict(lambda: defaultdict(float))

for seed in SEEDS:
    path = os.path.join(ROOT, f"seed_{seed}", "submission.csv")
    print(f"📥 Loading {path}")
    df = pd.read_csv(path)

    for user, group in df.groupby("user"):
        items = group["item"].tolist()
        for rank, item in enumerate(items):
            user_item_score[user][item] += 1.0 / (rank + 1)

rows = []
for user, item_scores in user_item_score.items():
    ranked_items = sorted(
        item_scores.items(), key=lambda x: -x[1]
    )[:TOPK]

    for item, _ in ranked_items:
        rows.append((user, item))

out_path = os.path.join(ROOT, "ensemble_rank.csv")
pd.DataFrame(rows, columns=["user", "item"]).to_csv(out_path, index=False)

print("=" * 60)
print("✅ Rank Ensemble Complete")
print(f"Saved to: {out_path}")
print("=" * 60)
