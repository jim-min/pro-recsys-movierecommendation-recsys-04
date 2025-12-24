"""
BERT4Rec Inference Script with PyTorch Lightning

Usage:
    python predict_bert4rec.py
    python predict_bert4rec.py inference.checkpoint_path=path/to/checkpoint.ckpt
    python predict_bert4rec.py inference.topk=20
"""

import os
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.models.bert4rec import BERT4Rec
from src.data.bert4rec_data import BERT4RecDataModule
from src.utils import get_directories, get_latest_checkpoint
from src.utils.recommend import recommend_topk
from src.utils.metrics import recall_at_k

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="bert4rec")
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
    datamodule = BERT4RecDataModule(
        data_dir=cfg.data.data_dir,
        data_file=cfg.data.data_file,
        batch_size=cfg.inference.batch_size,
        max_len=cfg.model.max_len,
        mask_prob=cfg.model.mask_prob,  # Not used in inference
        min_interactions=cfg.data.min_interactions,
        seed=cfg.data.seed,
        num_workers=cfg.data.num_workers,
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
    model = BERT4Rec.load_from_checkpoint(checkpoint_path)
    model.eval()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    log.info(f"Using device: {device}")

    # Get inference parameters
    topk = cfg.inference.topk
    batch_size = cfg.inference.batch_size

    # Output path: run_dir/submissions/bert4rec_predictions.csv
    # checkpoint_dir: run_dir/checkpoints
    # run_dir: checkpoint_dir의 상위 디렉토리
    run_dir = os.path.dirname(checkpoint_dir)
    output_path = os.path.join(
        run_dir, "submissions", f"bert4rec_predictions_{topk}.csv"
    )

    # Create submissions directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # === Validation Evaluation ===
    # 이미 트레인과정에서 NDCG@10을 계산함여 출력함

    # Generate predictions ================================================
    log.info(
        f"Generating top-{topk} recommendations for {datamodule.num_users} users..."
    )

    results = []
    user_sequences = datamodule.get_all_sequences()
    user_indices = list(user_sequences.keys())

    # Process in batches
    for start_idx in tqdm(range(0, len(user_indices), batch_size), desc="Inference"):
        end_idx = min(start_idx + batch_size, len(user_indices))
        batch_users = user_indices[start_idx:end_idx]

        # Prepare sequences
        batch_seqs = []
        batch_exclude = []

        for user_idx in batch_users:
            seq = user_sequences[user_idx]
            batch_seqs.append(seq)
            batch_exclude.append(set(seq))  # Exclude already interacted items

        # Get predictions
        with torch.no_grad():
            top_items = model.predict(
                user_sequences=batch_seqs, topk=topk, exclude_items=batch_exclude
            )

        # Convert to original IDs and save
        for i, user_idx in enumerate(batch_users):
            # Get original user ID
            original_user_id = datamodule.idx2user[user_idx]

            # Get top-k item indices
            item_indices = top_items[i]

            # Convert to original item IDs
            for item_idx in item_indices:
                if item_idx > 0 and item_idx <= datamodule.num_items:
                    original_item_id = datamodule.idx2item[item_idx]
                    results.append({"user": original_user_id, "item": original_item_id})

    # Save predictions
    log.info(f"Saving predictions to: {output_path}")
    pred_df = pd.DataFrame(results)
    pred_df.to_csv(output_path, index=False)

    log.info(f"Predictions saved! Total recommendations: {len(results)}")
    log.info(
        f"Average recommendations per user: {len(results) / datamodule.num_users:.2f}"
    )


if __name__ == "__main__":
    main()
