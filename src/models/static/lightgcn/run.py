import os
import argparse
import numpy as np
import pandas as pd
import torch

from data_utils import (
    read_interactions,
    encode_ids,
    build_user_item_matrix,
    build_user_positives,
    set_seed,
    train_valid_split_random,
)
from metrics import recall_at_k
from model import LightGCN
from trainer import LightGCNTrainer
from recommend import recommend_topk


def build_norm_adj(train_mat):
    from scipy.sparse import csr_matrix, diags

    num_users, num_items = train_mat.shape
    R = train_mat.tocsr()

    rows = np.concatenate([R.nonzero()[0], R.nonzero()[1] + num_users])
    cols = np.concatenate([R.nonzero()[1] + num_users, R.nonzero()[0]])
    data = np.ones(len(rows), dtype=np.float32)

    A = csr_matrix(
        (data, (rows, cols)),
        shape=(num_users + num_items, num_users + num_items),
    )

    deg = np.array(A.sum(axis=1)).squeeze()
    deg[deg == 0.0] = 1.0
    D_inv = diags(1.0 / np.sqrt(deg))
    norm_A = D_inv @ A @ D_inv
    norm_A = norm_A.tocoo()

    indices = torch.from_numpy(
        np.vstack((norm_A.row, norm_A.col))
    ).long()
    values = torch.from_numpy(norm_A.data).float()

    return torch.sparse_coo_tensor(
        indices, values, norm_A.shape
    ).coalesce()


def main():
    parser = argparse.ArgumentParser()

    # ===============================
    # Paths
    # ===============================
    parser.add_argument("--data_dir", default="/data/ephemeral/home/Seung/data/train/")
    parser.add_argument("--output_dir", default="/data/ephemeral/home/Seung/output/LightGCN/")

    # ===============================
    # Model / Training
    # ===============================
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--steps_per_epoch", type=int, default=200)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early_stop_patience", type=int, default=20)
    parser.add_argument("--no_cuda", action="store_true")

    # ===============================
    # Validation / Submission
    # ===============================
    parser.add_argument("--valid_ratio", type=float, default=0.0)
    parser.add_argument("--eval_topk", type=int, default=10)
    parser.add_argument("--submit_topk", type=int, default=10)  # 🔥 10 or 100

    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
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
    user_pos = build_user_positives(train_mat)
    norm_adj = build_norm_adj(train_mat).to(device)

    # ===============================
    # Model & Trainer
    # ===============================
    model = LightGCN(
        num_users=num_users,
        num_items=num_items,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
    )

    trainer = LightGCNTrainer(
        model=model,
        norm_adj=norm_adj,
        user_pos_list=user_pos,
        num_items=num_items,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
        ckpt_path=ckpt_path,
        early_stop_patience=args.early_stop_patience,
    )

    # ===============================
    # Train
    # ===============================
    trainer.train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        steps_per_epoch=args.steps_per_epoch,
    )

    # ===============================
    # Load Best Model
    # ===============================
    best = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(best["model_state"])
    print(f"✅ Best model loaded (epoch={best['epoch']})")

    # ===============================
    # Recommend (Top-K selectable)
    # ===============================
    rec = recommend_topk(
        model=model,
        norm_adj=norm_adj,
        train_mat=train_mat,
        topk=args.submit_topk,   # 🔥 핵심
        device=device,
        mask_train=True,
    )

    # ===============================
    # Validation (optional)
    # ===============================
    if args.valid_ratio > 0:
        actual, pred = [], []
        for u in range(num_users):
            if len(valid_gt[u]) == 0:
                continue
            actual.append(valid_gt[u])
            pred.append(rec[u].tolist())

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
