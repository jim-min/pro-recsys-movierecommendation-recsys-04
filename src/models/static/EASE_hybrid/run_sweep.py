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
    # Params
    # ===============================
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--eval_topk", type=int, default=10)
    parser.add_argument("--submit_topk", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    # ===============================
    # Sweep flag
    # ===============================
    parser.add_argument(
        "--sweep_lambda",
        action="store_true",
        help="Run lambda_reg sweep with interaction + metadata",
    )

    args = parser.parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # ===============================
    # Logging
    # ===============================
    log_path = os.path.join(
        args.output_dir,
        "lambda_sweep_interaction_metadata.csv",
    )
    if not os.path.exists(log_path):
        pd.DataFrame(
            columns=["lambda_reg", "recall@10"]
        ).to_csv(log_path, index=False)

    print("=" * 70)
    print("🚀 Lambda Sweep (Interaction + Metadata EASE)")
    print(f"   valid_ratio: {args.valid_ratio}")
    print("=" * 70)

    # ===============================
    # Load Data
    # ===============================
    df = read_interactions(args.data_dir)
    df_enc, u2i, i2u, it2i, i2it = encode_ids(df)

    num_users = len(u2i)
    num_items = len(it2i)

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

    train_mat = build_user_item_matrix(train_df, num_users, num_items)
    print(f"Train matrix nnz: {train_mat.nnz}")

    feat_mats = load_metadata_matrix(args.data_dir, it2i, num_items)

    # ===============================
    # Fixed metadata weights (최종 세팅)
    # ===============================
    alpha = 35.0
    meta_weights = {
        "genres": alpha,
        "directors": alpha,
        "writers": alpha,
        "years": alpha,
        "titles": alpha
    }

    print("📌 Fixed metadata weights:")
    for k, v in meta_weights.items():
        print(f"   {k}: {v}")

    # ===============================
    # Lambda sweep (interaction 포함)
    # ===============================
    if args.sweep_lambda:
        lambda_list = [300, 400, 500, 600, 800, 1000]

        for lam in lambda_list:
            print("=" * 80)
            print(f"🧪 Running lambda_reg = {lam}")
            print("=" * 80)

            model = EASE(
                lambda_reg=lam,
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
            recall = recall_at_k(actual, pred, k=args.eval_topk)

            print(f"📊 Recall@{args.eval_topk}: {recall:.6f}")

            pd.DataFrame(
                [[lam, recall]],
                columns=["lambda_reg", "recall@10"],
            ).to_csv(log_path, mode="a", header=False, index=False)

        print("✅ Lambda sweep finished.")
        return


if __name__ == "__main__":
    main()
