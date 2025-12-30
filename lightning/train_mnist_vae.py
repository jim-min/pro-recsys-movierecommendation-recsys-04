import lightning as L
import torch
import logging
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import seed_everything
from torchinfo import summary

from src.data.MNIST_data import MNISTDataModule
from src.models.mnist_vae import MnistVAE

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="mnist_vae")
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

    # 시드 설정
    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)
        log.info(f"Random seed set to: {cfg.seed}")

    # CUDA Tensor Core 최적화 설정
    torch.set_float32_matmul_precision(cfg.float32_matmul_precision)

    # 데이터 준비
    mnist = MNISTDataModule(
        data_dir=cfg.data.data_dir,
        batch_size=cfg.data.batch_size,
        target_labels=cfg.data.target_labels,
    )
    mnist.setup("fit")

    # 샘플 데이터 확인
    (X, y) = next(iter(mnist.train_dataloader()))
    log.info(f"X.shape={X.shape}, y.shape={y.shape}")

    # 모델 생성
    vae = MnistVAE(
        input_dim=cfg.model.input_dim,
        hidden_dims=cfg.model.hidden_dims,
        latent_dim=cfg.model.latent_dim,
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        kl_weight=cfg.model.kl_weight,
    )

    # 모델 요약
    summary(vae)

    # Hydra 실행별 디렉토리 설정
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    log.info(f"Checkpoint directory: {checkpoint_dir}")

    # TensorBoard 디렉토리: saved/tensorboard_logs/mnist-vae/날짜/시간
    run_parts = run_dir.split(os.sep)
    run_timestamp = os.path.join(run_parts[-2], run_parts[-1])
    tensorboard_dir = os.path.join(
        cfg.tensorboard.save_dir, cfg.model_name, run_timestamp
    )
    os.makedirs(tensorboard_dir, exist_ok=True)
    log.info(f"TensorBoard directory: {tensorboard_dir}")

    # 텐서보드 로거 설정
    logger = TensorBoardLogger(
        save_dir=tensorboard_dir,
        name="",  # 이미 경로에 포함되어 있으므로 빈 문자열
        version="",  # version_0 디렉토리 생성 방지
    )

    # 체크포인트 콜백 설정
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="mnist-vae-{epoch:02d}-{val_loss:.2f}",
        save_top_k=cfg.checkpoint.save_top_k,
        monitor=cfg.checkpoint.monitor,
        mode=cfg.checkpoint.mode,
    )

    # Trainer 설정
    trainer = L.Trainer(
        default_root_dir=run_dir,
        max_epochs=cfg.training.max_epochs,
        devices=cfg.trainer.devices,
        logger=logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        val_check_interval=cfg.trainer.val_check_interval,
        enable_progress_bar=cfg.trainer.enable_progress_bar,
        enable_model_summary=cfg.trainer.enable_model_summary,
    )

    # 학습 수행
    log.info("Starting training...")
    trainer.fit(vae, mnist)

    # 테스트 수행
    log.info("Running test...")
    test_results = trainer.test(vae, mnist)
    log.info(f"Test results: {test_results}")

    log.info("✅ Training finished!")

    return vae, mnist


if __name__ == "__main__":
    main()
