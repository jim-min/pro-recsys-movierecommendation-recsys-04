"""
MultiVAE Inference Script with PyTorch Lightning

Usage:
    python predict_multi_vae.py
    python predict_multi_vae.py inference.checkpoint_path=path/to/checkpoint.ckpt
    python predict_multi_vae.py inference.topk=20
"""

import os
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import pandas as pd
import torch

from src.models.multi_vae import MultiVAE
from src.data.recsys_data import RecSysDataModule
from src.utils import get_directories, get_latest_checkpoint
from src.utils.recommend import recommend_topk
from src.utils.metrics import recall_at_k

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="multi_vae")
def main(cfg: DictConfig):
    # Get checkpoint and TensorBoard directories (최근 실행 디렉토리)
    checkpoint_dir, tensorboard_dir = get_directories(cfg, stage="predict")
    log.info(f"Checkpoint directory: {checkpoint_dir}")
    log.info(f"TensorBoard directory: {tensorboard_dir}")

    # Print config
    log.info("Configuration:")
    log.info(OmegaConf.to_yaml(cfg))

    # Initialize DataModule
    log.info("Initializing DataModule...")
    datamodule = RecSysDataModule(
        data_dir=cfg.data.data_dir,
        batch_size=cfg.inference.batch_size,
        valid_ratio=cfg.data.valid_ratio,
        min_interactions=cfg.data.min_interactions,
        seed=cfg.data.seed,
        data_file=cfg.data.data_file,
        split_strategy=cfg.data.split_strategy,
        temporal_split_ratio=cfg.data.temporal_split_ratio,
        use_cache=cfg.data_cache.use_cache,
        cache_dir=cfg.data_cache.cache_dir,
    )

    # Setup data
    datamodule.setup()
    log.info(f"num_users: {datamodule.num_users}, num_items: {datamodule.num_items}")

    # Load checkpoint
    checkpoint_path = cfg.inference.checkpoint_path

    if checkpoint_path is None:
        # Find the latest checkpoint in the checkpoint directory
        checkpoint_path = get_latest_checkpoint(checkpoint_dir)
        log.info(f"No checkpoint specified, using: {checkpoint_path}")

    log.info(f"Loading model from: {checkpoint_path}")
    model = MultiVAE.load_from_checkpoint(
        checkpoint_path,
        num_items=datamodule.num_items,
        weights_only=False,
    )
    model.eval()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    log.info(f"Using device: {device}")

    # Get inference parameters
    topk = cfg.inference.topk
    batch_size = cfg.inference.batch_size

    # Output path: run_dir/submissions/multi_vae_predictions.csv
    # checkpoint_dir: run_dir/checkpoints
    # run_dir: checkpoint_dir의 상위 디렉토리
    run_dir = os.path.dirname(checkpoint_dir)
    output_path = os.path.join(
        run_dir, "submissions", f"multi_vae_predictions_{topk}.csv"
    )

    # Create submissions directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # === Validation Evaluation ===
    log.info(f"Generating Top-{topk} recommendations for validation...")
    train_mat = datamodule.get_train_matrix()

    recommendations_valid = recommend_topk(
        model,
        train_mat,
        k=topk,
        device=device,
        batch_size=batch_size,
    )

    # Validation Recall@K 계산
    valid_gt = datamodule.get_validation_ground_truth()
    valid_gt_list = [valid_gt[u] for u in range(datamodule.num_users)]
    pred_list = [rec.tolist() for rec in recommendations_valid]

    recall = recall_at_k(valid_gt_list, pred_list, k=topk)
    log.info(f"Validation Recall@{topk}: {recall:.4f}")

    # === Submission Generation ===
    log.info(f"Generating Top-{topk} recommendations for submission...")
    full_mat = datamodule.get_full_matrix()

    recommendations_submission = recommend_topk(
        model,
        full_mat,
        k=topk,
        device=device,
        batch_size=batch_size,
    )

    # Save predictions
    log.info(f"Creating submission file...")
    results = []
    for u_idx in range(datamodule.num_users):
        for item_idx in recommendations_submission[u_idx]:
            user_id = datamodule.idx2user[u_idx]
            item_id = datamodule.idx2item[int(item_idx)]
            results.append({"user": user_id, "item": item_id})

    pred_df = pd.DataFrame(results)
    pred_df.to_csv(output_path, index=False)

    log.info(f"Predictions saved to: {output_path}")
    log.info(f"Total recommendations: {len(results)}")
    log.info(
        f"Average recommendations per user: {len(results) / datamodule.num_users:.2f}"
    )
    log.info("✅ All processes finished!")


if __name__ == "__main__":
    main()
