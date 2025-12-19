import lightning as L
import torch
import logging
import datetime
import hydra
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import seed_everything
from torchinfo import summary

from src.data.MNIST_data import MNISTDataModule
from src.models.auto_encoder import Encoder, Decoder, AutoEncoder

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="autoencoder")
def main(cfg: DictConfig):
    # 로그 포맷 설정
    logging.basicConfig(
        level=getattr(logging, cfg.logging.level),
        format=cfg.logging.format,
        datefmt=cfg.logging.datefmt,
        force=True,
    )

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
    autoencoder = AutoEncoder()

    # 모델 요약
    summary(autoencoder)

    # 텐서보드 로거 설정
    version_name = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    logger = TensorBoardLogger(
        save_dir=cfg.logging.save_dir,
        name=cfg.logging.name,
        version=version_name,
    )

    # 체크포인트 콜백 설정
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.logging.checkpoint_dir,
        filename="autoencoder-{epoch:02d}-{val_loss:.2f}",
        save_top_k=cfg.logging.save_top_k,
        monitor="val_loss",
        mode="min",
    )

    # Trainer 설정
    trainer = L.Trainer(
        default_root_dir=cfg.logging.default_root_dir,
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
    trainer.fit(autoencoder, mnist)

    # 테스트 수행
    test_results = trainer.test(autoencoder, mnist)
    log.info(f"Test results: {test_results}")

    return autoencoder, mnist


if __name__ == "__main__":
    main()
