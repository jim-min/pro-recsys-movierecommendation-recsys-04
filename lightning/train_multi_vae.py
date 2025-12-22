import os
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
import torch
import pandas as pd
import numpy as np

from src.data.recsys_data import RecSysDataModule
from src.models.multi_vae import MultiVAE
from src.utils.recommend import recommend_topk
from src.utils.metrics import recall_at_k

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="multi_vae")
def main(cfg: DictConfig):
    # Hydra의 자동 출력 디렉토리 가져오기
    hydra_cfg = HydraConfig.get()
    run_dir = hydra_cfg.runtime.output_dir
    log.info(f"Hydra output directory: {run_dir}")

    # 설정 출력
    log.info("=" * 80)
    log.info("Configuration:")
    log.info(OmegaConf.to_yaml(cfg))
    log.info("=" * 80)

    # 시드 고정
    L.seed_everything(cfg.seed)

    # Float32 matmul precision 설정
    torch.set_float32_matmul_precision(cfg.float32_matmul_precision)

    # DataModule 초기화
    logging.info("Initializing RecSys DataModule...")
    data_module = RecSysDataModule(
        data_dir=cfg.data.data_dir,
        batch_size=cfg.data.batch_size,
        valid_ratio=cfg.data.valid_ratio,
        min_interactions=cfg.data.min_interactions,
        seed=cfg.seed,
        data_file=cfg.data.data_file,
        split_strategy=cfg.data.split_strategy,
        temporal_split_ratio=cfg.data.get("temporal_split_ratio", 0.8),
    )

    # 모델 초기화를 위해 setup 호출 (num_users, num_items 필요)
    data_module.setup()

    log.info(f"Number of users: {data_module.num_users}")
    log.info(f"Number of items: {data_module.num_items}")

    # 모델 초기화
    log.info("Initializing MultiVAE model...")
    model = MultiVAE(
        num_items=data_module.num_items,
        hidden_dims=cfg.model.hidden_dims,  # [ 600, 200]
        dropout=cfg.model.dropout,
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        kl_max_weight=cfg.training.kl_max_weight,
        kl_anneal_steps=cfg.training.kl_anneal_steps,
    )

    # Hydra 실행별 디렉토리 설정
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    log.info(f"Checkpoint directory: {checkpoint_dir}")

    # TensorBoard 디렉토리: saved/tensorboard_logs/multi-vae/날짜/시간
    # run_dir에서 날짜/시간 추출
    # run_dir 예: ./saved/hydra_logs/2025-01-15/10-30-45
    # -> 마지막 2개 경로 요소 추출: 2025-01-15/10-30-45
    run_parts = run_dir.split(os.sep)
    run_timestamp = os.path.join(run_parts[-2], run_parts[-1])  # 날짜/시간
    tensorboard_dir = os.path.join(
        cfg.tensorboard.save_dir, cfg.model_name, run_timestamp
    )
    os.makedirs(tensorboard_dir, exist_ok=True)
    log.info(f"TensorBoard directory: {tensorboard_dir}")

    # TensorBoard Logger
    logger = TensorBoardLogger(
        save_dir=tensorboard_dir,
        name="",  # 이미 경로에 포함되어 있으므로 빈 문자열
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,  # 실행마다 고유한 경로
        filename="multi-vae-{epoch:02d}-{val_loss:.4f}",
        monitor=cfg.checkpoint.monitor,
        mode=cfg.checkpoint.mode,
        save_top_k=cfg.checkpoint.save_top_k,
        save_last=True,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=20,
        mode="min",
        verbose=True,
    )

    # Trainer 설정
    trainer = L.Trainer(
        max_epochs=cfg.training.max_epochs,
        devices=cfg.trainer.devices,
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        val_check_interval=cfg.trainer.val_check_interval,
        enable_progress_bar=cfg.trainer.enable_progress_bar,
        enable_model_summary=cfg.trainer.enable_model_summary,
        default_root_dir=run_dir,  # Hydra 실행 디렉토리 사용
    )

    # 학습
    log.info("Starting training...")
    trainer.fit(model, data_module)

    # Best 체크포인트 로드 (현재 실행의 best만 선택됨)
    log.info("Loading best checkpoint...")
    log.info(f"Best checkpoint path: {checkpoint_callback.best_model_path}")
    best_model = MultiVAE.load_from_checkpoint(
        checkpoint_callback.best_model_path,
        num_items=data_module.num_items,
        weights_only=False,
    )

    # Validation을 위한 Top-K 추천 생성 (train_mat만 제외)
    log.info(f"Generating Top-{cfg.recommend.topk} recommendations for validation...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_mat = data_module.get_train_matrix()

    recommendations_valid = recommend_topk(
        best_model,
        train_mat,
        k=cfg.recommend.topk,
        device=device,
        batch_size=cfg.data.batch_size,
    )

    # Validation Recall@K 계산
    valid_gt = data_module.get_validation_ground_truth()
    valid_gt_list = [valid_gt[u] for u in range(data_module.num_users)]
    pred_list = [rec.tolist() for rec in recommendations_valid]

    recall = recall_at_k(valid_gt_list, pred_list, k=cfg.recommend.topk)
    log.info(f"Validation Recall@{cfg.recommend.topk}: {recall:.4f}")

    # Submission을 위한 Top-K 추천 생성 (train + valid 전체 제외)
    log.info(f"Generating Top-{cfg.recommend.topk} recommendations for submission...")
    full_mat = data_module.get_full_matrix()

    recommendations_submission = recommend_topk(
        best_model,
        full_mat,
        k=cfg.recommend.topk,
        device=device,
        batch_size=cfg.data.batch_size,
    )

    # Submission 파일 생성
    log.info("Creating submission file...")
    submission_dir = os.path.join(run_dir, "submissions")
    os.makedirs(submission_dir, exist_ok=True)

    rows = []
    for u_idx in range(data_module.num_users):
        for item_idx in recommendations_submission[u_idx]:
            user_id = data_module.idx2user[u_idx]
            item_id = data_module.idx2item[int(item_idx)]
            rows.append((user_id, item_id))

    submission_df = pd.DataFrame(rows, columns=["user", "item"])
    submission_path = os.path.join(submission_dir, "submission.csv")
    submission_df.to_csv(submission_path, index=False)

    log.info(f"Submission saved to: {submission_path}")
    log.info("✅ All processes finished!")


if __name__ == "__main__":
    main()
