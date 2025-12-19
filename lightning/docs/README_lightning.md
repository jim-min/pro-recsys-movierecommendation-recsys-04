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
│   ├── autoencoder.yaml          # 메인 실험 설정
│   └── logging/
│       └── default.yaml           # 공통 로깅 설정
├── src/
│   ├── data/
│   │   └── MNIST_data.py          # MNIST DataModule
│   ├── models/
│   │   └── auto_encoder.py        # AutoEncoder 모델
│   └── utils/
│       └── visualize_autoencoder.py  # 시각화 유틸리티
├── notebooks/
│   └── visualize_autoencoder.ipynb   # 결과 시각화 노트북
├── train_autoencoder.py           # 학습 스크립트
└── saved/                         # 학습 결과 저장
    ├── checkpoints/               # 모델 체크포인트
    ├── tensorboard_logs/          # TensorBoard 로그
    └── logs/                      # Trainer 로그
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
  - logging: default  # 공통 로깅 설정 로드
  - _self_

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
  max_epochs: 20
  lr: 1e-3
```

### 공통 로깅 설정: `configs/logging/default.yaml`

```yaml
save_dir: "./saved/tensorboard_logs"
checkpoint_dir: "./saved/checkpoints"
save_top_k: 1
format: "%(asctime)s|%(levelname)s|%(filename)s:%(lineno)d| %(message)s"
level: "DEBUG"
```

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
    # 로그 포맷 설정
    logging.basicConfig(
        level=getattr(logging, cfg.logging.level),
        format=cfg.logging.format,
        datefmt=cfg.logging.datefmt,
        force=True,
    )

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
        save_dir=cfg.logging.save_dir,
        name=cfg.logging.name,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.logging.checkpoint_dir,
        save_top_k=cfg.logging.save_top_k,
        monitor="val_loss",
    )

    # Trainer 생성 및 학습
    trainer = L.Trainer(
        max_epochs=cfg.training.max_epochs,
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
    dirpath=cfg.logging.checkpoint_dir,
    filename="autoencoder-{epoch:02d}-{val_loss:.2f}",
    save_top_k=cfg.logging.save_top_k,
    monitor="val_loss",
    mode="min",
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
autoencoder = AutoEncoder.load_from_checkpoint(
    "saved/checkpoints/autoencoder-epoch=19-val_loss=0.05.ckpt"
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
