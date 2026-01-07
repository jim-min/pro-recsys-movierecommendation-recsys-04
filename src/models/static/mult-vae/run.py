import os
import argparse
import torch
import pandas as pd

from data_utils import (
    read_interactions,
    encode_ids,
    build_user_item_matrix,
    train_valid_split_random,
    set_seed,
)
from model import MultiVAE
from trainer import MultiVAETrainer
from recommend import recommend_topk


def main():
    parser = argparse.ArgumentParser()

    # ===============================
    # Paths
    # ===============================
    parser.add_argument("--data_dir", default="/data/ephemeral/home/Seung/data/train/")
    parser.add_argument("--output_dir", default="/data/ephemeral/home/Seung/output/MultVAE/")

    # ===============================
    # Training
    # ===============================
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)

    # ===============================
    # Validation / Submission
    # ===============================
    parser.add_argument("--valid_ratio", type=float, default=0.0)
    parser.add_argument("--eval_topk", type=int, default=10)
    parser.add_argument("--submit_topk", type=int, default=10)

    # ===============================
    # Early Stop
    # ===============================
    parser.add_argument("--early_stop_patience", type=int, default=500)

    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    ckpt_path = os.path.join(args.output_dir, "best.pt")

    # ===============================
    # Load & Encode
    # ===============================
    df = read_interactions(args.data_dir)
    df_enc, u2i, i2u, it2i, i2it = encode_ids(df)

    num_users = len(u2i)
    num_items = len(it2i)

    # ===============================
    # Train / Valid split
    # ===============================
    if args.valid_ratio > 0:
        train_df, valid_gt = train_valid_split_random(
            df_enc, num_users, valid_ratio=args.valid_ratio
        )
    else:
        train_df = df_enc
        valid_gt = {u: [] for u in range(num_users)}

    train_mat = build_user_item_matrix(train_df, num_users, num_items)

    # ===============================
    # Model & Trainer
    # ===============================
    model = MultiVAE(num_items)

    trainer = MultiVAETrainer(
        model=model,
        train_mat=train_mat,
        valid_gt=valid_gt,
        num_items=num_items,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
        ckpt_path=ckpt_path,
        early_stop_patience=args.early_stop_patience,
    )

    trainer.train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        topk=args.eval_topk,
    )

    # ===============================
    # Load Best Model
    # ===============================
    best = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(best["model_state"])

    # ===============================
    # Submission (Top-K selectable)
    # ===============================
    rec = recommend_topk(
        model=model,
        train_mat=train_mat,
        topk=args.submit_topk,
        device=device,
        batch_size=args.batch_size,
        mask_train=True,
    )

    rows = []
    for u_idx in range(num_users):
        for it in rec[u_idx]:
            rows.append((i2u[u_idx], i2it[int(it)]))

    out_path = os.path.join(args.output_dir, "submission.csv")
    pd.DataFrame(rows, columns=["user", "item"]).to_csv(out_path, index=False)

    print("✅ Finished")
    print(f"submit_topk = {args.submit_topk}")
    print("Saved to:", out_path)


if __name__ == "__main__":
    main()
