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


@hydra.main(version_base=None, config_path="configs", config_name="bert4rec_v2")
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
        random_mask_prob=cfg.model.random_mask_prob,
        last_item_mask_ratio=cfg.model.last_item_mask_ratio,
        min_interactions=cfg.data.min_interactions,
        seed=cfg.data.seed,
        num_workers=cfg.data.num_workers,
        use_full_data=cfg.data.use_full_data,
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
        random_mask_prob=cfg.model.random_mask_prob,
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        share_embeddings=cfg.model.share_embeddings,
        # Metadata parameters
        num_genres=datamodule.num_genres,
        num_directors=datamodule.num_directors,
        num_writers=datamodule.num_writers,
        title_embedding_dim=datamodule.title_embedding_dim,
        use_genre_emb=cfg.model.use_genre_emb,
        use_director_emb=cfg.model.use_director_emb,
        use_writer_emb=cfg.model.use_writer_emb,
        use_title_emb=cfg.model.use_title_emb,
        metadata_fusion=cfg.model.metadata_fusion,
        metadata_dropout=cfg.model.metadata_dropout,
    )

    # Log metadata info
    log.info(f"Metadata dimensions:")
    log.info(f"  Genres: {datamodule.num_genres}")
    log.info(f"  Directors: {datamodule.num_directors}")
    log.info(f"  Writers: {datamodule.num_writers}")
    log.info(f"  Title embedding dim: {datamodule.title_embedding_dim}")

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

    # Log hyperparameters to TensorBoard
    hparams = {
        # Data
        "data/batch_size": cfg.data.batch_size,
        "data/min_interactions": cfg.data.min_interactions,
        "data/seed": cfg.data.seed,
        "data/use_full_data": cfg.data.use_full_data,
        # Model
        "model/hidden_units": cfg.model.hidden_units,
        "model/num_heads": cfg.model.num_heads,
        "model/num_layers": cfg.model.num_layers,
        "model/max_len": cfg.model.max_len,
        "model/dropout_rate": cfg.model.dropout_rate,
        "model/random_mask_prob": cfg.model.random_mask_prob,
        "model/last_item_mask_ratio": cfg.model.last_item_mask_ratio,
        "model/share_embeddings": cfg.model.share_embeddings,
        "model/use_genre_emb": cfg.model.use_genre_emb,
        "model/use_director_emb": cfg.model.use_director_emb,
        "model/use_writer_emb": cfg.model.use_writer_emb,
        "model/use_title_emb": cfg.model.use_title_emb,
        "model/metadata_fusion": cfg.model.metadata_fusion,
        "model/metadata_dropout": cfg.model.metadata_dropout,
        # Training
        "training/num_epochs": cfg.training.num_epochs,
        "training/lr": cfg.training.lr,
        "training/weight_decay": cfg.training.weight_decay,
        "training/gradient_clip_val": cfg.training.gradient_clip_val,
        "training/precision": cfg.training.precision,
        "training/early_stopping": cfg.training.early_stopping,
        "training/early_stopping_patience": cfg.training.early_stopping_patience,
        # Metadata dimensions (data-dependent)
        "metadata/num_genres": datamodule.num_genres,
        "metadata/num_directors": datamodule.num_directors,
        "metadata/num_writers": datamodule.num_writers,
        "metadata/title_embedding_dim": datamodule.title_embedding_dim,
        "metadata/num_items": datamodule.num_items,
        "metadata/num_users": datamodule.num_users,
    }

    # Define metrics to track in HPARAMS (required for TensorBoard HPARAMS tab)
    # 실제 학습 중 로깅되는 metric 이름과 동일하게 설정
    metrics = {
        "val/recall@5": 0.0,
        "val/recall@10": 0.0,
        "val/ndcg@5": 0.0,
        "val/ndcg@10": 0.0,
    }

    # Log both hparams and metrics for TensorBoard HPARAMS plugin
    logger.log_hyperparams(hparams, metrics)

    # Callbacks
    callbacks = []

    # ModelCheckpoint: Save best model based on validation or training metric
    use_full_data = cfg.data.use_full_data

    if use_full_data:
        # Full data training: monitor train_loss
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="bert4rec-{epoch:02d}-{train_loss:.4f}",
            monitor="train_loss",
            mode="min",
            save_top_k=cfg.checkpoint.save_top_k,
            verbose=True,
        )
        log.info("Checkpoint monitoring: train_loss (use_full_data=True)")
    else:
        # Standard training: monitor validation metric
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="bert4rec-{epoch:02d}-{val_ndcg@10:.4f}",
            monitor=cfg.checkpoint.monitor,
            mode=cfg.checkpoint.mode,
            save_top_k=cfg.checkpoint.save_top_k,
            verbose=True,
        )
        log.info(f"Checkpoint monitoring: {cfg.checkpoint.monitor}")

    callbacks.append(checkpoint_callback)

    # EarlyStopping: Stop training if metric doesn't improve
    if cfg.training.early_stopping:
        if use_full_data:
            # Full data mode: monitor train_loss
            early_stopping = EarlyStopping(
                monitor="train_loss",
                patience=cfg.training.early_stopping_patience,
                mode="min",
                verbose=True,
            )
            callbacks.append(early_stopping)
            log.info("Early stopping enabled (monitoring train_loss for full data mode)")
        else:
            # Standard mode: monitor validation metric
            early_stopping = EarlyStopping(
                monitor=cfg.checkpoint.monitor,
                patience=cfg.training.early_stopping_patience,
                mode=cfg.checkpoint.mode,
                verbose=True,
            )
            callbacks.append(early_stopping)
            log.info(f"Early stopping enabled (monitoring {cfg.checkpoint.monitor})")
    else:
        log.info("Early stopping disabled")

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
    if use_full_data:
        # Full data training: no validation
        trainer.fit(model, train_dataloaders=datamodule.train_dataloader())
    else:
        # Standard training: with validation
        trainer.fit(model, datamodule=datamodule)

    # Load best model and log final metrics
    log.info("Training completed!")
    log.info(f"Best model path: {checkpoint_callback.best_model_path}")
    log.info(
        f"Best {cfg.training.monitor_metric}: {checkpoint_callback.best_model_score:.4f}"
    )

    # Print final gate values if using gate fusion
    if hasattr(model, '_final_gate_values'):
        feature_names, avg_gates = model._final_gate_values
        gate_str = " | ".join(
            [f"{name}: {avg_gates[i].item():.4f}" for i, name in enumerate(feature_names)]
        )
        log.info(f"Final Gate values: {gate_str}")


if __name__ == "__main__":
    main()
