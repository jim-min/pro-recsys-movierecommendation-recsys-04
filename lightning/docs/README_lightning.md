# PyTorch Lightning 프로젝트 가이드

## 프로젝트 개요

PyTorch Lightning을 사용한 MNIST Autoencoder 학습 프로젝트입니다.
Hydra를 사용한 설정 관리와 TensorBoard 로깅을 포함합니다.

## 설치

```bash
uv add lightning hydra-core tensorboard torchinfo
```

## 프로젝트 구조

```
lightning/
├── configs/
│   ├── autoencoder.yaml          # Autoencoder 실험 설정
│   ├── mnist_vae.yaml            # MNIST VAE 실험 설정
│   ├── multi_vae.yaml            # Multi-VAE 실험 설정
│   └── common/
│       └── default_setup.yaml    # 공통 설정
├── src/
│   ├── data/
│   │   ├── MNIST_data.py         # MNIST DataModule
│   │   └── recsys_data.py        # RecSys DataModule
│   ├── models/
│   │   ├── auto_encoder.py       # AutoEncoder 모델
│   │   ├── mnist_vae.py          # MNIST VAE 모델
│   │   └── multi_vae.py          # Multi-VAE 모델
│   └── utils/
│       ├── visualize_autoencoder.py  # 시각화 유틸리티
│       ├── metrics.py            # 평가 메트릭
│       └── recommend.py          # 추천 생성
├── notebooks/
│   ├── visualize_autoencoder.ipynb   # Autoencoder 시각화
│   ├── visualize_mnist_vae.ipynb     # MNIST VAE 시각화
│   └── visualize_multi_vae.ipynb     # Multi-VAE 시각화
├── train_autoencoder.py          # Autoencoder 학습
├── train_mnist_vae.py            # MNIST VAE 학습
├── train_multi_vae.py            # Multi-VAE 학습
├── visualize_multi_vae.py        # Multi-VAE 시각화 스크립트
├── docs/
│   ├── README_autoencoder.md     # Autoencoder 가이드
│   ├── README_mnist_vae.md       # MNIST VAE 가이드
│   ├── README_multi_vae.md       # Multi-VAE 가이드
│   └── README_lightning.md       # 이 문서
└── saved/                        # 학습 결과 저장
    ├── hydra_logs/               # Hydra 실행별 로그
    │   ├── autoencoder/          # Autoencoder 모델
    │   ├── mnist-vae/            # MNIST VAE 모델
    │   └── multi-vae/            # Multi-VAE 모델
    │       └── YYYY-MM-DD/       # 날짜별 디렉토리
    │           └── HH-MM-SS/     # 시간별 디렉토리
    │               ├── checkpoints/  # 체크포인트
    │               ├── submissions/  # Submission 파일 (multi-vae)
    │               └── *.log     # 로그 파일
    └── tensorboard_logs/         # TensorBoard 로그
        ├── mnist-autoencoder/    # 모델별 디렉토리
        ├── mnist-vae/
        └── multi-vae/
            └── YYYY-MM-DD/       # 날짜별
                └── HH-MM-SS/     # 시간별
```

## 빠른 시작

### 1. 학습 실행

```bash
python train_autoencoder.py
```

Hydra 설정 오버라이드:
```bash
# epoch 수 변경
python train_autoencoder.py training.max_epochs=50

# batch size 변경
python train_autoencoder.py data.batch_size=64

# GPU 지정
python train_autoencoder.py trainer.devices=1
```

### 2. 결과 시각화

```bash
jupyter notebook notebooks/visualize_autoencoder.ipynb
```

### 3. TensorBoard 실행

```bash
tensorboard --logdir saved/tensorboard_logs
```

## 설정 관리 (Hydra)

### 메인 설정 파일: `configs/autoencoder.yaml`

```yaml
defaults:
  - common: default_setup  # 공통 설정 로드
  - _self_                 # 현재 파일의 설정이 우선함

tensorboard:
  name: "mnist-autoencoder"

# 데이터 설정
data:
  data_dir: "~/data/"
  batch_size: 32
  target_labels: [0, 1, 3, 8]

# 모델 설정
model:
  latent_dim: 2
  hidden_dim: 256

# 학습 설정
training:
  max_epochs: 2
  lr: 1e-3
  weight_decay: 0.1
  T_max: 100  # CosineAnnealingLR

# Trainer 설정
trainer:
  devices: "auto"
  log_every_n_steps: 5
  val_check_interval: 1.0
  enable_progress_bar: true
  enable_model_summary: true
```

### 공통 설정: `configs/common/default_setup.yaml`

```yaml
# 공통 설정 (모든 실험에서 공유)
# @package _global_
hydra:
  run:
    dir: ./saved/hydra_logs/${model_name}/${now:%Y-%m-%d}/${now:%H-%M-%S}
  job:
    chdir: False  # 로그 파일이 생성되는 위치로 작업 디렉터리 변경 안 함
  job_logging:
    version: 1
    formatters:
      simple:
        datefmt: "%H:%M:%S"
        format: "%(asctime)s.%(msecs)03d|%(levelname)s|%(filename)s:%(lineno)d| %(message)s"
    handlers:
      console:
        class: logging.StreamHandler
        formatter: simple
        stream: ext://sys.stdout
      file:
        class: logging.FileHandler
        formatter: simple
        filename: ${hydra:runtime.output_dir}/${hydra:job.name}.log
    root:
      level: INFO
      handlers: [console, file]  # 콘솔과 파일 모두 출력

tensorboard:
  save_dir: "./saved/tensorboard_logs"
  name: "experiment"

checkpoint:
  save_top_k: 1
  monitor: "val_loss"
  mode: "min"

# 기타 설정
seed: 42
float32_matmul_precision: "medium"  # 'high', 'medium', or 'highest'
```

**주요 변경사항**:
- Hydra 로그 디렉토리: `saved/hydra_logs/${model_name}/날짜/시간/`
- 각 모델 config에서 `model_name` 정의 (예: `model_name: "multi-vae"`)
- 모델별로 독립적인 디렉토리 구조 유지

## MNIST Autoencoder 예제

### 1. LightningModule 구현

```python
class AutoEncoder(L.LightningModule):
    def __init__(self, latent_dim=2, hidden_dim=256):
        super().__init__()
        self.encoder = Encoder(latent_dim, hidden_dim)
        self.decoder = Decoder(latent_dim, hidden_dim)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)
        x_hat = self(x)
        loss = F.mse_loss(x_hat, x)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)
        x_hat = self(x)
        loss = F.mse_loss(x_hat, x)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=100
        )
        return [optimizer], [scheduler]
```

### 2. DataModule 구현

```python
class MNISTDataModule(L.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, target_labels=None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.target_labels = target_labels

    def setup(self, stage=None):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        full_train = MNIST(self.data_dir, train=True,
                          transform=transform, download=True)

        # 특정 레이블만 필터링
        if self.target_labels:
            indices = [i for i, (_, label) in enumerate(full_train)
                      if label in self.target_labels]
            full_train = Subset(full_train, indices)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                         batch_size=self.batch_size, shuffle=True)
```

### 3. Hydra를 사용한 학습 스크립트

```python
@hydra.main(version_base=None, config_path="configs",
            config_name="autoencoder")
def main(cfg: DictConfig):
    # 데이터 준비
    mnist = MNISTDataModule(
        data_dir=cfg.data.data_dir,
        batch_size=cfg.data.batch_size,
        target_labels=cfg.data.target_labels,
    )

    # 모델 생성
    autoencoder = AutoEncoder(
        latent_dim=cfg.model.latent_dim,
        hidden_dim=cfg.model.hidden_dim
    )

    # 로거 및 콜백 설정
    logger = TensorBoardLogger(
        save_dir=tensorboard_dir,
        name="",
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.checkpoint.dirpath,
        save_top_k=cfg.checkpoint.save_top_k,
        monitor=cfg.checkpoint.monitor,
        mode=cfg.checkpoint.mode,
    )

    # Trainer 생성 및 학습
    trainer = L.Trainer(
        max_epochs=cfg.training.max_epochs,
        devices=cfg.trainer.devices,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        val_check_interval=cfg.trainer.val_check_interval,
        enable_progress_bar=cfg.trainer.enable_progress_bar,
        enable_model_summary=cfg.trainer.enable_model_summary,
        logger=logger,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(autoencoder, mnist)
```

## 시각화

이 프로젝트는 4가지 시각화 기능을 제공합니다:

1. **샘플 데이터 시각화**: 원본 vs 재구성 이미지 비교
2. **Latent Space 시각화**: 2D latent space에 데이터 투영
3. **Latent 좌표에서 생성**: 특정 좌표의 이미지 생성
4. **Latent Grid**: 전체 latent space 탐색

자세한 사용법은 `notebooks/visualize_autoencoder.ipynb` 참고

## 주요 기능

### 1. Hydra 설정 관리

```bash
# 설정 오버라이드
python train_autoencoder.py training.lr=1e-4

# 여러 설정 동시 변경
python train_autoencoder.py training.max_epochs=50 data.batch_size=64

# 설정 확인
python train_autoencoder.py --cfg job
```

### 2. 콜백 (Callbacks)

```python
from lightning.pytorch.callbacks import ModelCheckpoint

checkpoint_callback = ModelCheckpoint(
    dirpath=cfg.checkpoint.checkpoint_dir,
    filename="autoencoder-{epoch:02d}-{val_loss:.2f}",
    save_top_k=cfg.checkpoint.save_top_k,
    monitor=cfg.checkpoint.monitor,
    mode=cfg.checkpoint.mode,
)

trainer = L.Trainer(callbacks=[checkpoint_callback])
```

### 3. 로깅

프로젝트는 두 가지 로깅을 사용합니다:

**Python 로깅 (콘솔)**
```python
log.info(f"Train dataset size: {len(train_dataset)}")
# 출력: 14:57:23|INFO|train_autoencoder.py:44| Train dataset size: 22182
```

**TensorBoard 로깅**
```python
self.log('train_loss', loss)
self.log('val_loss', val_loss)
```

### 4. GPU 사용

```python
# Hydra 설정에서
trainer:
  devices: "auto"  # 자동 감지
  # devices: 1     # 1개 GPU 사용
  # devices: [0,1] # 특정 GPU 사용
```

## 모델 저장 및 로드

```python
# 체크포인트에서 자동 로드
# Hydra 실행 디렉토리 기반 경로
autoencoder = AutoEncoder.load_from_checkpoint(
    "saved/hydra_logs/2025-12-20/16-44-09/checkpoints/autoencoder-epoch=19-val_loss=0.05.ckpt",
    weights_only=False,
)

# 추론 모드
autoencoder.eval()
```

## 추론 (Inference)

```python
import torch

autoencoder.eval()
with torch.no_grad():
    # 인코딩
    z = autoencoder.encoder(x)

    # 디코딩
    x_reconstructed = autoencoder.decoder(z)

    # 전체 과정
    output = autoencoder(x)
```

## 팁

### 1. Hydra outputs 디렉토리 비활성화

Hydra는 기본적으로 `outputs/` 디렉토리를 생성합니다. 비활성화하려면:

```yaml
# configs/autoencoder.yaml에 추가
hydra:
  output_subdir: null
  run:
    dir: .
```

### 2. 특정 레이블만 학습

```yaml
data:
  target_labels: [0, 1, 3, 8]  # 4개 숫자만 사용
```

### 3. Learning Rate Scheduler

프로젝트는 CosineAnnealingLR을 사용합니다:

```python
def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=self.T_max
    )
    return [optimizer], [scheduler]
```

## 유용한 링크

- PyTorch Lightning: https://lightning.ai/docs/pytorch/
- Hydra: https://hydra.cc/docs/intro/
- TensorBoard: https://www.tensorflow.org/tensorboard
