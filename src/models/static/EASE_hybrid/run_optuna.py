import os
import argparse
import optuna
import numpy as np
import pandas as pd
import logging
from datetime import datetime

from data_utils import (
    read_interactions,
    encode_ids,
    build_user_item_matrix,
    load_metadata_matrix,
    train_valid_split_random,
    set_seed,
)
from model import EASE
from trainer import EASETrainer
from recommend import recommend_topk
from metrics import recall_at_k
from utils import build_valid_lists


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/data/ephemeral/home/Seung/data/train/")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument("--study_name", type=str, default="ease_hybrid_optuna")
    parser.add_argument("--storage", type=str, default=None)  # sqlite:///ease_optuna.db
    args = parser.parse_args()

    set_seed(args.seed)

    # ===============================
    # Logging setup
    # ===============================
    LOG_DIR = "./optuna_log"
    os.makedirs(LOG_DIR, exist_ok=True)

    log_file = os.path.join(
        LOG_DIR,
        f"optuna_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)

    logger.info("===== Optuna EASE Hybrid Started =====")
    logger.info(f"Args: {vars(args)}")

    # ===============================
    # 1) Data prepare (딱 1번)
    # ===============================
    df = read_interactions(args.data_dir)
    df_enc, u2i, i2u, it2i, i2it = encode_ids(df)

    num_users = len(u2i)
    num_items = len(it2i)
    logger.info(f"Users: {num_users}, Items: {num_items}")

    if args.valid_ratio > 0:
        train_df, valid_gt = train_valid_split_random(
            df_enc,
            num_users=num_users,
            valid_ratio=args.valid_ratio,
            seed=args.seed,
        )
        logger.info("Validation ON")
    else:
        train_df = df_enc
        valid_gt = {u: [] for u in range(num_users)}
        logger.info("Validation OFF")

    train_mat = build_user_item_matrix(train_df, num_users, num_items)
    logger.info(f"Train matrix nnz: {train_mat.nnz}")

    feat_mats = load_metadata_matrix(args.data_dir, it2i, num_items)

    # ===============================
    # 2) Objective
    # ===============================
    def objective(trial):
        lambda_reg = trial.suggest_float("lambda_reg", 100, 2000, log=True)

        meta_weights = {
            "genres":    trial.suggest_float("w_genres", 0.0, 50.0),
            "directors": trial.suggest_float("w_directors", 0.0, 50.0),
            "writers":   trial.suggest_float("w_writers", 0.0, 50.0),
            "years":     trial.suggest_float("w_years", 0.0, 20.0),
            "titles":    trial.suggest_float("w_titles", 0.0, 20.0),
        }

        logger.info(f"[Trial {trial.number}] lambda_reg = {lambda_reg:.3f}")

        for k, v in meta_weights.items():
            logger.info(f"    - {k:<10}: {v:.4f}")

        model = EASE(lambda_reg=lambda_reg, meta_weights=meta_weights)
        trainer = EASETrainer(model, train_mat, feat_mats=feat_mats)
        trainer.train()

        score_mat = trainer.predict()
        rec = recommend_topk(
            score_mat,
            train_mat,
            topk=args.topk,
            mask_train=True,
        )

        actual, pred = build_valid_lists(valid_gt, rec)
        recall = recall_at_k(actual, pred, k=args.topk)

        logger.info(
            f"[Trial {trial.number}] "
            f"Recall@{args.topk} = {recall:.6f}"
        )
        logger.info("-" * 50)

        return recall

    # ===============================
    # 3) Optuna run
    # ===============================
    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
    )

    study.optimize(objective, n_trials=args.n_trials)

    # ===============================
    # 4) Save results
    # ===============================
    logger.info("===== Optuna Finished =====")
    logger.info(f"BEST PARAMS: {study.best_params}")
    logger.info(f"BEST Recall@{args.topk}: {study.best_value:.6f}")

    df_trials = study.trials_dataframe()
    csv_path = os.path.join(LOG_DIR, "trials.csv")
    df_trials.to_csv(csv_path, index=False)

    logger.info(f"Saved trials dataframe to {csv_path}")


if __name__ == "__main__":
    main()
