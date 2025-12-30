"""
MultiVAE Training Script with PyTorch Lightning

Usage:
    python train_multi_vae.py
    python train_multi_vae.py model.hidden_dims=[400,200] training.lr=0.001
"""

import os
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import TensorBoardLogger
import torch

from src.data.recsys_data import RecSysDataModule
from src.models.multi_vae import MultiVAE
from src.utils import get_directories

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="multi_vae")
def main(cfg: DictConfig):
    # Hydra의 자동 출력 디렉토리 가져오기
    hydra_cfg = HydraConfig.get()
    log.info(f"Hydra output directory: {hydra_cfg.runtime.output_dir}")

    # Print config
    log.info("Configuration:")
    log.info(OmegaConf.to_yaml(cfg))

    # Set seed for reproducibility
    L.seed_everything(cfg.seed, workers=True)

    # Float32 matmul precision 설정
    torch.set_float32_matmul_precision(cfg.float32_matmul_precision)

    # Initialize DataModule
    log.info("Initializing DataModule...")
    datamodule = RecSysDataModule(
        data_dir=cfg.data.data_dir,
        batch_size=cfg.data.batch_size,
        valid_ratio=cfg.data.valid_ratio,
        min_interactions=cfg.data.min_interactions,
        seed=cfg.data.seed,
        data_file=cfg.data.data_file,
        split_strategy=cfg.data.split_strategy,
        temporal_split_ratio=cfg.data.temporal_split_ratio,
    )

    # Setup data to get num_items
    datamodule.setup()
    log.info(f"Number of users: {datamodule.num_users}")
    log.info(f"Number of items: {datamodule.num_items}")

    # Initialize Model
    log.info("Initializing Model...")
    model = MultiVAE(
        num_items=datamodule.num_items,
        hidden_dims=cfg.model.hidden_dims,
        dropout=cfg.model.dropout,
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        kl_max_weight=cfg.training.kl_max_weight,
        kl_anneal_steps=cfg.training.kl_anneal_steps,
    )

    # Get checkpoint and TensorBoard directories
    checkpoint_dir, tensorboard_dir = get_directories(cfg, stage="fit")
    log.info(f"Checkpoint directory: {checkpoint_dir}")
    log.info(f"TensorBoard directory: {tensorboard_dir}")

    # Logger
    logger = TensorBoardLogger(
        save_dir=tensorboard_dir,
        name="",  # 이미 경로에 포함되어 있으므로 빈 문자열
        version="",  # version_0 디렉토리 생성 방지
    )

    # Callbacks
    callbacks = []

    # ModelCheckpoint: Save best model based on validation metric
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,  # 실행마다 고유한 경로
        filename="multi-vae-{epoch:02d}-{val_loss:.4f}",
        monitor=cfg.checkpoint.monitor,
        mode=cfg.checkpoint.mode,
        save_top_k=cfg.checkpoint.save_top_k,
        save_last=False,
        verbose=True,
    )
    callbacks.append(checkpoint_callback)

    # EarlyStopping: Stop training if validation metric doesn't improve
    if cfg.training.early_stopping:
        early_stopping = EarlyStopping(
            monitor=cfg.checkpoint.monitor,
            patience=cfg.training.early_stopping_patience,
            mode=cfg.checkpoint.mode,
            verbose=True,
        )
        callbacks.append(early_stopping)

    # LearningRateMonitor: Log learning rate
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks.append(lr_monitor)

    # Trainer
    log.info("Initializing Trainer...")
    trainer = L.Trainer(
        max_epochs=cfg.training.max_epochs,
        devices="auto",
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=10,
        val_check_interval=1.0,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    # Train
    log.info("Starting training...")
    trainer.fit(model, datamodule=datamodule)

    # Load best model and log final metrics
    log.info("Training completed!")
    log.info(f"Best model path: {checkpoint_callback.best_model_path}")
    log.info(
        f"Best {cfg.checkpoint.monitor}: {checkpoint_callback.best_model_score:.4f}"
    )


if __name__ == "__main__":
    main()
