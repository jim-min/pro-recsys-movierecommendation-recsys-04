import os
import argparse
import numpy as np
import pandas as pd

from data_utils import (
    read_interactions,
    encode_ids,
    build_user_item_matrix,
    load_metadata_matrix,
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
    parser.add_argument(
        "--data_dir",
        default="/data/ephemeral/home/Seung/data/train/",
    )
    parser.add_argument(
        "--output_root",
        default="/data/ephemeral/home/Seung/output/EASE_Hybrid/",
    )

    # ===============================
    # Model Params
    # ===============================
    parser.add_argument("--lambda_reg", type=float, default=794.63)
    parser.add_argument("--valid_ratio", type=float, default=0.0)
    parser.add_argument("--eval_topk", type=int, default=10)
    parser.add_argument("--submit_topk", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    set_seed(args.seed)

    # ===============================
    # Seed-based output dir
    # ===============================
    output_dir = os.path.join(args.output_root, f"seed_{args.seed}")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("🚀 Running EASE Hybrid")
    print(f"Seed        : {args.seed}")
    print(f"Lambda_reg  : {args.lambda_reg}")
    print(f"Valid_ratio : {args.valid_ratio}")
    print(f"Output_dir  : {output_dir}")
    print("=" * 60)

    # ===============================
    # 1. Load & Shuffle Data (🔥 핵심)
    # ===============================
    df = read_interactions(args.data_dir)
    df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    df_enc, u2i, i2u, it2i, i2it = encode_ids(df)

    num_users = len(u2i)
    num_items = len(it2i)
    print(f"Users: {num_users}, Items: {num_items}")

    # ===============================
    # 2. Train / Valid Split
    # ===============================
    if args.valid_ratio > 0:
        train_df, valid_gt = train_valid_split_random(
            df_enc,
            num_users=num_users,
            valid_ratio=args.valid_ratio,
            seed=args.seed,
        )
        print("🧪 Validation ON")
    else:
        train_df = df_enc
        valid_gt = {u: [] for u in range(num_users)}
        print("🧪 Validation OFF (Full data)")

    # ===============================
    # 3. Build Matrices
    # ===============================
    train_mat = build_user_item_matrix(train_df, num_users, num_items)
    print(f"Train matrix nnz: {train_mat.nnz}")

    feat_mats = load_metadata_matrix(args.data_dir, it2i, num_items)

    # ===============================
    # 4. Train EASE
    # ===============================
    meta_weights = {
        "genres": 35.0,
        "directors": 35.0,
        "writers": 35.0,
        "years": 35.0,
        "titles": 35.0,
    }

    model = EASE(
        lambda_reg=args.lambda_reg,
        meta_weights=meta_weights,
    )

    trainer = EASETrainer(
        model,
        train_mat,
        feat_mats=feat_mats,
    )
    trainer.train()

    # ===============================
    # 5. Predict & Recommend
    # ===============================
    score_mat = trainer.predict()

    rec = recommend_topk(
        score_mat,
        train_mat,
        topk=args.submit_topk,
        mask_train=True,
    )

    # ===============================
    # 6. Validation (optional)
    # ===============================
    if args.valid_ratio > 0:
        actual, pred = build_valid_lists(valid_gt, rec)
        val_recall = recall_at_k(actual, pred, k=args.eval_topk)
        print(f"📊 Validation Recall@{args.eval_topk}: {val_recall:.6f}")

    # ===============================
    # 7. Save submission
    # ===============================
    rows = []
    for u_idx in range(num_users):
        for it in rec[u_idx]:
            rows.append((i2u[u_idx], i2it[int(it)]))

    out_path = os.path.join(output_dir, "submission.csv")
    pd.DataFrame(rows, columns=["user", "item"]).to_csv(out_path, index=False)

    print("✅ Submission saved")
    print(f"Path: {out_path}")


if __name__ == "__main__":
    main()
