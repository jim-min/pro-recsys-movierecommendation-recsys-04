"""
RecVAE Inference Script with PyTorch Lightning

Usage:
    python predict_rec_vae.py
    python predict_rec_vae.py inference.checkpoint_path=path/to/checkpoint.ckpt
    python predict_rec_vae.py inference.topk=20
"""

import os
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import pandas as pd
import torch

from src.models.rec_vae import RecVAE
from src.data.recsys_data import RecSysDataModule
from src.utils import get_directories, get_latest_checkpoint
from src.utils.recommend import recommend_topk
from src.utils.metrics import recall_at_k

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="rec_vae")
def main(cfg: DictConfig):
    # Get checkpoint and TensorBoard directories (최근 실행 디렉토리)
    checkpoint_dir, tensorboard_dir = get_directories(cfg, stage="predict")
    log.info(f"Checkpoint directory: {checkpoint_dir}")
    log.info(f"TensorBoard directory: {tensorboard_dir}")

    # Print config
    log.info("Configuration:")
    log.info(OmegaConf.to_yaml(cfg))

    # ------------------------
    # Inference params ✅ (먼저 정의!)
    # ------------------------
    topk = cfg.inference.topk
    batch_size = cfg.inference.batch_size

    # ------------------------
    # Output paths ✅ (먼저 정의!)
    # ------------------------
    run_dir = os.path.dirname(checkpoint_dir)
    submissions_dir = os.path.join(run_dir, "submissions")
    os.makedirs(submissions_dir, exist_ok=True)

    output_path = os.path.join(submissions_dir, f"rec_vae_predictions_{topk}.csv")

    # ------------------------
    # DataModule
    # ------------------------
    log.info("Initializing DataModule...")
    datamodule = RecSysDataModule(
        data_dir=cfg.data.data_dir,
        batch_size=batch_size,  # ✅ 통일
        valid_ratio=cfg.data.valid_ratio,
        min_interactions=cfg.data.min_interactions,
        seed=cfg.data.seed,
        data_file=cfg.data.data_file,
        split_strategy=cfg.data.split_strategy,
        temporal_split_ratio=cfg.data.temporal_split_ratio,
    )

    datamodule.setup()
    log.info(f"num_users: {datamodule.num_users}, num_items: {datamodule.num_items}")

    # ------------------------
    # Load checkpoint
    # ------------------------
    checkpoint_path = cfg.inference.checkpoint_path
    if checkpoint_path is None:
        checkpoint_path = get_latest_checkpoint(checkpoint_dir)
        log.info(f"No checkpoint specified, using: {checkpoint_path}")

    log.info(f"Loading RecVAE model from: {checkpoint_path}")
    model = RecVAE.load_from_checkpoint(
        checkpoint_path,
        num_items=datamodule.num_items,
        weights_only=False,
    )
    model.eval()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    log.info(f"Using device: {device}")

    # ------------------------
    # Save user latent z (mu) from BEST RecVAE  ✅ FIXED
    # ------------------------
    log.info("Extracting and saving user latent z (mu)...")

    train_mat = datamodule.get_train_matrix()  # usually csr_matrix
    num_users = train_mat.shape[0]

    all_mu = []

    with torch.no_grad():
        for start in range(0, num_users, batch_size):
            end = min(start + batch_size, num_users)

            # csr → dense (batch)
            batch_dense = torch.from_numpy(
                train_mat[start:end].toarray()
            ).float().to(device)

            _, mu, _ = model(batch_dense)
            all_mu.append(mu.cpu())

    mu = torch.cat(all_mu, dim=0)  # [num_users, latent_dim]
    user_ids = [datamodule.idx2user[u] for u in range(num_users)]

    z_save_path = os.path.join(submissions_dir, "rec_vae_user_latent_z.pt")
    torch.save({"user_id": user_ids, "z": mu}, z_save_path)
    log.info(f"User latent z saved to: {z_save_path}")

    # ------------------------
    # Validation Evaluation
    # ------------------------
    log.info(f"Generating Top-{topk} recommendations for validation...")

    recommendations_valid = recommend_topk(
        model,
        train_mat,  # ✅ 위에서 가져온 train_mat 재사용
        k=topk,
        device=device,
        batch_size=batch_size,
    )

    valid_gt = datamodule.get_validation_ground_truth()
    valid_gt_list = [valid_gt[u] for u in range(datamodule.num_users)]
    pred_list = [rec.tolist() for rec in recommendations_valid]

    recall = recall_at_k(valid_gt_list, pred_list, k=topk)
    log.info(f"Validation Recall@{topk}: {recall:.4f}")

    # ------------------------
    # Submission Generation
    # ------------------------
    log.info(f"Generating Top-{topk} recommendations for submission...")
    full_mat = datamodule.get_full_matrix()

    recommendations_submission = recommend_topk(
        model,
        full_mat,
        k=topk,
        device=device,
        batch_size=batch_size,
    )

    log.info("Creating submission file...")
    results = []
    for u_idx in range(datamodule.num_users):
        user_id = datamodule.idx2user[u_idx]
        for item_idx in recommendations_submission[u_idx]:
            item_id = datamodule.idx2item[int(item_idx)]
            results.append({"user": user_id, "item": item_id})

    pred_df = pd.DataFrame(results)
    pred_df.to_csv(output_path, index=False)

    log.info(f"Predictions saved to: {output_path}")
    log.info(f"Total recommendations: {len(results)}")
    log.info(f"Average recommendations per user: {len(results) / datamodule.num_users:.2f}")

    # ------------------------
    # EDA용 추천 + GT 저장
    # ------------------------
    log.info("Saving EDA CSV with recommendations and validation GT...")

    eda_rows = []
    for u_idx in range(datamodule.num_users):
        user_id = datamodule.idx2user[u_idx]

        rec_items_idx = recommendations_valid[u_idx]
        rec_items = [datamodule.idx2item[int(i)] for i in rec_items_idx]

        gt_items_idx = valid_gt.get(u_idx, [])
        gt_items = [datamodule.idx2item[int(i)] for i in gt_items_idx]

        row = {
            "user": user_id,
            **{f"rec_{i+1}": rec_items[i] for i in range(len(rec_items))},
            # ✅ CSV 분석 편하게 문자열로 저장 (원하면 다시 split 가능)
            "gt_items": " ".join(map(str, gt_items)),
        }
        eda_rows.append(row)

    eda_df = pd.DataFrame(eda_rows)
    rec_cols = [f"rec_{i+1}" for i in range(topk)]
    eda_df = eda_df[["user"] + rec_cols + ["gt_items"]]

    eda_output_path = os.path.join(submissions_dir, f"rec_vae_eda_top{topk}.csv")
    eda_df.to_csv(eda_output_path, index=False)

    log.info(f"EDA CSV saved to: {eda_output_path}")
    log.info("✅ All processes finished!")


if __name__ == "__main__":
    main()
