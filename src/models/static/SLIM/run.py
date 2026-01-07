import os
import argparse
import numpy as np
import pandas as pd
from scipy.sparse import save_npz, load_npz

from data_utils import (
    read_interactions,
    encode_ids,
    build_user_item_matrix,
    train_valid_split_random,
    set_seed,
)
from metrics import recall_at_k
from model import SLIM
from trainer import SLIMTrainer
from recommend import recommend_topk
from utils import build_valid_lists


def main():
    parser = argparse.ArgumentParser()

    # ===============================
    # Paths
    # ===============================
    parser.add_argument("--data_dir", default="/data/ephemeral/home/Seung/data/train/")
    parser.add_argument("--output_dir", default="/data/ephemeral/home/Seung/output/SLIM/")

    # ===============================
    # SLIM params
    # ===============================
    parser.add_argument("--alpha", type=float, default=1e-4)
    parser.add_argument("--l1_ratio", type=float, default=0.1)
    parser.add_argument("--positive", action="store_true")  # default False -> 아래에서 True로 처리
    parser.add_argument("--max_iter", type=int, default=80)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--n_jobs", type=int, default=8)

    # ===============================
    # Validation / Submission
    # ===============================
    parser.add_argument("--valid_ratio", type=float, default=0.0)
    parser.add_argument("--eval_topk", type=int, default=10)
    parser.add_argument("--submit_topk", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    # ===============================
    # Cache (W 저장/로드)
    # ===============================
    parser.add_argument("--load_W", action="store_true")  # 이미 학습한 W 재사용
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    W_path = os.path.join(args.output_dir, "W_slim.npz")

    # -------------------------------
    # Load & Encode
    # -------------------------------
    df = read_interactions(args.data_dir)
    df_enc, u2i, i2u, it2i, i2it = encode_ids(df)

    num_users = len(u2i)
    num_items = len(it2i)

    # -------------------------------
    # Train / Valid Split
    # -------------------------------
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

    # sanity print (원하면 지워도 됨)
    print("Original interactions:", len(df_enc))
    print("Train matrix nnz:", train_mat.nnz)

    # -------------------------------
    # Train / Load SLIM W
    # -------------------------------
    model = SLIM()

    if args.load_W and os.path.exists(W_path):
        print(f"✅ Loading W from {W_path}")
        W = load_npz(W_path)
        model.set_W(W)
    else:
        # 기본은 positive=True 추천이라, 플래그 없으면 True로 쓰자
        positive = True if not args.positive else True

        trainer = SLIMTrainer(
            model=model,
            train_mat=train_mat,
            alpha=args.alpha,
            l1_ratio=args.l1_ratio,
            positive=positive,
            max_iter=args.max_iter,
            tol=args.tol,
            n_jobs=args.n_jobs,
            verbose=1,
        )
        W = trainer.train()
        save_npz(W_path, W)
        print(f"💾 Saved W to {W_path}")

    # -------------------------------
    # Predict (scores) + Recommend
    # -------------------------------
    # score_mat = X @ W  (dense로 뽑아 추천)
    score_sparse = model.predict_scores(train_mat)  # (U x I) sparse
    score_mat = score_sparse.toarray()              # 메모리 ok면 이 방식이 제일 간단

    rec = recommend_topk(
        score_mat=score_mat,
        train_mat=train_mat,
        topk=args.submit_topk,
        mask_train=True,
    )

    # -------------------------------
    # Validation (optional)
    # -------------------------------
    if args.valid_ratio > 0:
        actual, pred = build_valid_lists(valid_gt, rec)
        val_recall = recall_at_k(actual, pred, k=args.eval_topk)
        print(f"📊 Validation Recall@{args.eval_topk}: {val_recall:.6f}")

        # 🔥 SLIM sweep 기록
        log_path = os.path.join(args.output_dir, "slim_log.txt")
        with open(log_path, "a") as f:
            f.write(
                f"alpha={args.alpha}, "
                f"l1_ratio={args.l1_ratio}, "
                f"recall@{args.eval_topk}={val_recall:.6f}\n"
            )


    # -------------------------------
    # Submission
    # -------------------------------
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
