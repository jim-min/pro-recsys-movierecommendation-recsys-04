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
        "--output_dir",
        default="/data/ephemeral/home/Seung/output/EASE_Hybrid/",
    )

    # ===============================
    # Model Params
    # ===============================
    parser.add_argument(
        "--lambda_reg",
        type=float,
        default=794.63,
    )
    parser.add_argument(
        "--valid_ratio",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--eval_topk",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--submit_topk",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    # ===============================
    # Sweep control
    # ===============================
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run alpha sweep for metadata weights",
    )

    args = parser.parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # ===============================
    # Logging
    # ===============================
    log_path = os.path.join(args.output_dir, "alpha_sweep_results.csv")
    if not os.path.exists(log_path):
        pd.DataFrame(
            columns=[
                "alpha",
                "lambda_reg",
                "recall@10",
            ]
        ).to_csv(log_path, index=False)

    print("=" * 60)
    print("🚀 Running Hybrid EASE (Normalized Metadata)")
    print(f"   lambda_reg : {args.lambda_reg}")
    print(f"   valid_ratio: {args.valid_ratio}")
    print("=" * 60)

    # ===============================
    # 1. Load Data
    # ===============================
    df = read_interactions(args.data_dir)
    df_enc, u2i, i2u, it2i, i2it = encode_ids(df)

    num_users = len(u2i)
    num_items = len(it2i)
    print(f"Users: {num_users}, Items: {num_items}")

    # ===============================
    # 2. Split
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
        print("🧪 Validation OFF")

    # ===============================
    # 3. Matrices
    # ===============================
    train_mat = build_user_item_matrix(train_df, num_users, num_items)
    print(f"Train matrix nnz: {train_mat.nnz}")

    feat_mats = load_metadata_matrix(args.data_dir, it2i, num_items)

    # ===============================
    # 4. Alpha Sweep (log-scale)
    # ===============================
    if args.sweep:
        # 🔥 로그 스케일 기반 sweep
        alpha_list = [25.0, 30.0, 35.0, 40.0, 45.0]

        for alpha in alpha_list:
            meta_weights = {
                "genres": alpha,
                "directors": alpha,
                "writers": alpha,
                "years": alpha,
                "titles": alpha
            }

            print("=" * 70)
            print(f"🧪 Running alpha = {alpha}")
            print("=" * 70)

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

            score_mat = trainer.predict()
            rec = recommend_topk(
                score_mat,
                train_mat,
                topk=args.submit_topk,
                mask_train=True,
            )

            actual, pred = build_valid_lists(valid_gt, rec)
            val_recall = recall_at_k(actual, pred, k=args.eval_topk)

            print(f"📊 Recall@{args.eval_topk}: {val_recall:.6f}")

            pd.DataFrame(
                [[alpha, args.lambda_reg, val_recall]],
                columns=["alpha", "lambda_reg", "recall@10"],
            ).to_csv(log_path, mode="a", header=False, index=False)

        print("✅ Alpha sweep finished.")
        return

    # ===============================
    # 5. Single Run (no sweep)
    # ===============================
    alpha = 35.0
    meta_weights = {
        "genres": 16.04,
        "directors": 43.61,
        "writers": 34.80,
        "years": 7.94,
        "titles": 12.37
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

    score_mat = trainer.predict()
    rec = recommend_topk(
        score_mat,
        train_mat,
        topk=args.submit_topk,
        mask_train=True,
        )

    if args.valid_ratio > 0:
        actual, pred = build_valid_lists(valid_gt, rec)
        val_recall = recall_at_k(actual, pred, k=args.eval_topk)
        print(f"📊 Validation Recall@{args.eval_topk}: {val_recall:.6f}")

    rows = []
    for u_idx in range(num_users):
        for it in rec[u_idx]:
            rows.append((i2u[u_idx], i2it[int(it)]))

    out_path = os.path.join(args.output_dir, "submission.csv")
    pd.DataFrame(rows, columns=["user", "item"]).to_csv(out_path, index=False)

    print("✅ Submission file saved!")
    print(f"   Path: {out_path}")


if __name__ == "__main__":
    main()
