"""
BERT4Rec Training Script with PyTorch Lightning

Usage:
    python train_bert4rec.py
    python train_bert4rec.py model.hidden_units=128 training.lr=0.001
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
from src.models.bert4rec import BERT4Rec
from src.data.bert4rec_data import BERT4RecDataModule
from src.utils import get_directories

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="bert4rec")
def main(cfg: DictConfig):
    # Hydra의 자동 출력 디렉토리 가져오기
    hydra_cfg = HydraConfig.get()
    log.info(f"Hydra output directory: {hydra_cfg.runtime.output_dir}")

    # Print config
    log.info("Configuration:")
    log.info(OmegaConf.to_yaml(cfg))

    # Set seed for reproducibility
    L.seed_everything(cfg.data.seed, workers=True)

    # Float32 matmul precision 설정
    torch.set_float32_matmul_precision(cfg.float32_matmul_precision)

    # Initialize DataModule
    log.info("Initializing DataModule...")
    datamodule = BERT4RecDataModule(
        data_dir=cfg.data.data_dir,
        data_file=cfg.data.data_file,
        batch_size=cfg.data.batch_size,
        max_len=cfg.model.max_len,
        mask_prob=cfg.model.mask_prob,
        min_interactions=cfg.data.get("min_interactions", 3),
        seed=cfg.data.seed,
        num_workers=cfg.data.get("num_workers", 4),
    )

    # Setup data to get num_items
    datamodule.setup()
    log.info(f"Number of users: {datamodule.num_users}")
    log.info(f"Number of items: {datamodule.num_items}")

    # Initialize Model
    log.info("Initializing Model...")
    model = BERT4Rec(
        num_items=datamodule.num_items,
        hidden_units=cfg.model.hidden_units,
        num_heads=cfg.model.num_heads,
        num_layers=cfg.model.num_layers,
        max_len=cfg.model.max_len,
        dropout_rate=cfg.model.dropout_rate,
        mask_prob=cfg.model.mask_prob,
        lr=cfg.training.lr,
        weight_decay=cfg.training.get("weight_decay", 0.0),
        share_embeddings=cfg.model.get("share_embeddings", True),
    )

    # Get checkpoint and TensorBoard directories
    checkpoint_dir, tensorboard_dir = get_directories(cfg, stage='fit')
    log.info(f"Checkpoint directory: {checkpoint_dir}")
    log.info(f"TensorBoard directory: {tensorboard_dir}")

    # Logger
    logger = TensorBoardLogger(
        save_dir=tensorboard_dir,
        name="",  # 이미 경로에 포함되어 있으므로 빈 문자열
    )

    # Callbacks
    callbacks = []

    # ModelCheckpoint: Save best model based on validation metric
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,  # 실행마다 고유한 경로
        filename="bert4rec-{epoch:02d}-{val_ndcg@10:.4f}",
        monitor=cfg.checkpoint.monitor,
        mode=cfg.checkpoint.mode,
        save_top_k=cfg.checkpoint.save_top_k,
        verbose=True,
    )
    callbacks.append(checkpoint_callback)

    # EarlyStopping: Stop training if validation metric doesn't improve
    if cfg.training.get("early_stopping", False):
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
        max_epochs=cfg.training.num_epochs,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=cfg.training.log_every_n_steps,
        val_check_interval=cfg.training.val_check_interval,
        gradient_clip_val=cfg.training.gradient_clip_val,
        deterministic=cfg.training.deterministic,
        precision=cfg.training.precision,
    )

    # Train
    log.info("Starting training...")
    trainer.fit(model, datamodule=datamodule)

    # Load best model and log final metrics
    log.info("Training completed!")
    log.info(f"Best model path: {checkpoint_callback.best_model_path}")
    log.info(
        f"Best {cfg.training.monitor_metric}: {checkpoint_callback.best_model_score:.4f}"
    )


if __name__ == "__main__":
    main()
