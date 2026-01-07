import os
import argparse
import numpy as np
import pandas as pd

from data_utils import (
    read_interactions,
    encode_ids,
    build_user_item_matrix,
    train_valid_split_random,
    set_seed,
)
from metrics import recall_at_k
from model import EASE
from trainer import EASETrainer
from recommend import recommend_topk
from utils import build_valid_lists


def main():
    parser = argparse.ArgumentParser()

    # ===============================
    # Paths
    # ===============================
    parser.add_argument("--data_dir", default="/data/ephemeral/home/Seung/data/train/")
    parser.add_argument("--output_dir", default="/data/ephemeral/home/Seung/output/EASE/")

    # ===============================
    # EASE params
    # ===============================
    parser.add_argument("--lambda_reg", type=float, default=300.0)

    # ===============================
    # Validation / Submission
    # ===============================
    parser.add_argument("--valid_ratio", type=float, default=0.0)
    parser.add_argument("--eval_topk", type=int, default=10)
    parser.add_argument("--submit_topk", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # ===============================
    # Load & Encode
    # ===============================
    df = read_interactions(args.data_dir)
    df_enc, u2i, i2u, it2i, i2it = encode_ids(df)

    num_users = len(u2i)
    num_items = len(it2i)

    # ===============================
    # Train / Valid Split
    # ===============================
    if args.valid_ratio > 0:
        train_df, valid_gt = train_valid_split_random(
            df_enc,
            num_users=num_users,
            valid_ratio=args.valid_ratio,
            seed=args.seed,
        )
        print(f"🧪 Validation ON (ratio={args.valid_ratio})")
    else:
        train_df = df_enc
        valid_gt = {u: [] for u in range(num_users)}
        print("🧪 Validation OFF (final training)")

    train_mat = build_user_item_matrix(train_df, num_users, num_items)
    
    # 🔍 sanity check (중요!)
    print("Original interactions:", len(df_enc))
    print("Train matrix nnz:", train_mat.nnz)

    # ===============================
    # Train EASE
    # ===============================
    model = EASE(lambda_reg=args.lambda_reg)
    trainer = EASETrainer(model, train_mat)
    trainer.train()

    # ===============================
    # Predict
    # ===============================
    score_mat = trainer.predict()

    rec = recommend_topk(
        score_mat,
        train_mat,
        topk=args.submit_topk,
        mask_train=True,
    )

    # ===============================
    # Validation
    # ===============================
    if args.valid_ratio > 0:
        actual, pred = build_valid_lists(valid_gt, rec)
        val_recall = recall_at_k(actual, pred, k=args.eval_topk)
        print(f"📊 Validation Recall@{args.eval_topk}: {val_recall:.6f}")

    # ===============================
    # Submission
    # ===============================
    rows = []
    for u_idx in range(num_users):
        for it in rec[u_idx]:
            rows.append((i2u[u_idx], i2it[int(it)]))

    out_path = os.path.join(args.output_dir, "submission.csv")
    pd.DataFrame(rows, columns=["user", "item"]).to_csv(out_path, index=False)

    print("✅ submission.csv 생성 완료")
    print(f"submit_topk = {args.submit_topk}")
    print("Saved to:", out_path)


if __name__ == "__main__":
    main()
