# MNIST AutoEncoder

MNIST 손글씨 숫자 데이터셋을 위한 간단한 AutoEncoder (AE) 구현입니다.

## 목차
- [개요](#개요)
- [프로젝트 구조](#프로젝트-구조)
- [사용 방법](#사용-방법)
- [AutoEncoder 핵심 개념](#autoencoder-핵심-개념)
- [주요 질문과 답변](#주요-질문과-답변)

---

## 개요

AutoEncoder는 **비지도 학습(Unsupervised Learning)** 방식의 신경망으로, 데이터를 압축하고 복원하는 것을 학습합니다.

### 기본 구조

```
     Encoder          Decoder
x → [784→256→2] → z → [2→256→784] → x_hat

입력 이미지 (784) → 잠재 벡터 (2) → 재구성 이미지 (784)
```

### 목적

1. **차원 축소 (Dimensionality Reduction)**: 고차원 데이터를 저차원으로 압축
2. **특징 추출 (Feature Learning)**: 데이터의 중요한 특징을 자동으로 학습
3. **노이즈 제거 (Denoising)**: 노이즈가 있는 데이터를 복원
4. **이상 탐지 (Anomaly Detection)**: 재구성 오차가 큰 샘플 감지

### VAE와의 비교

AutoEncoder는 **결정론적** 모델로 압축/복원에 집중하고, VAE는 **확률적** 모델로 생성에 특화되어 있습니다.

상세한 비교는 [README_mnist_vae.md](./README_mnist_vae.md#autoencoder-vs-vae)를 참고하세요.

---

## 프로젝트 구조

```
lightning/
├── src/
│   ├── models/
│   │   └── auto_encoder.py           # AutoEncoder 모델
│   └── data/
│       └── MNIST_data.py              # 데이터 로더
├── configs/
│   └── autoencoder.yaml               # Hydra 설정
├── train_autoencoder.py               # 학습 스크립트
├── notebooks/
│   └── visualize_autoencoder.ipynb    # 시각화 노트북
└── docs/
    └── README_autoencoder.md          # 이 문서
```

---

## 사용 방법

### 1. 학습

```bash
python train_autoencoder.py
```

설정 변경:
```bash
# 에포크 수 변경
python train_autoencoder.py training.max_epochs=10

# 배치 크기 변경
python train_autoencoder.py data.batch_size=64

# 특정 숫자만 학습
python train_autoencoder.py data.target_labels=[0,1,3,8]
```

### 2. 시각화

```bash
jupyter notebook notebooks/visualize_autoencoder.ipynb
```

**노트북 사용법**:
```python
# Cell 1: 실행할 run_timestamp 지정
run_timestamp = None  # 가장 최근 실행 자동 선택
# run_timestamp = "2025-12-20/16-44-09"  # 특정 실행 지정
```

노트북은 자동으로 Hydra 실행 디렉토리(`saved/hydra_logs/{run_timestamp}/checkpoints/`)에서 최적 체크포인트를 로드합니다.

포함된 시각화:
1. **잠재 공간 시각화** - 2D 산점도로 각 숫자의 분포 확인
2. **재구성 결과** - 원본 vs 재구성 이미지 비교
3. **잠재 공간 경계** - 결정 경계 시각화
4. **이미지 보간** - 두 이미지 사이를 선형 보간

### 3. 모델 구조

```python
from src.models.auto_encoder import AutoEncoder

# 모델 생성
autoencoder = AutoEncoder()

# 구조:
# Encoder: [784 → 256 → 2]  (ReLU 활성화)
# Decoder: [2 → 256 → 784]  (ReLU 활성화)

# 추론
x = torch.randn(1, 784)
z = autoencoder.encoder(x)      # (1, 2) 잠재 벡터
x_hat = autoencoder.decoder(z)  # (1, 784) 재구성
```

---

## AutoEncoder 핵심 개념

### 1. 구조

```python
class AutoEncoder(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()  # 784 → 256 → 2
        self.decoder = Decoder()  # 2 → 256 → 784
```

**Encoder** (압축):
```python
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 2)  # 2D 잠재 공간
        )
```

**Decoder** (복원):
```python
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(2, 256),
            nn.ReLU(),
            nn.Linear(256, 28 * 28)
        )
```

### 2. Loss Function

```python
loss = F.mse_loss(x_hat, x)
```

**MSE (Mean Squared Error)**:
```python
MSE = (1/N) * Σᵢ (x_hatᵢ - xᵢ)²
```

- **의미**: 재구성된 이미지와 원본 이미지의 픽셀 단위 차이
- **최소화**: 재구성을 원본에 가깝게 만듦

**다른 Loss 옵션**:
- **MAE**: `F.l1_loss(x_hat, x)` - 아웃라이어에 덜 민감
- **BCE**: `F.binary_cross_entropy(x_hat, x)` - 이진 이미지에 적합

### 3. 학습 과정

```python
def training_step(self, batch, batch_idx):
    x, _ = batch                  # 원본 이미지
    x = x.view(x.size(0), -1)     # Flatten (28×28 → 784)
    z = self.encoder(x)           # 압축 (784 → 2)
    x_hat = self.decoder(z)       # 복원 (2 → 784)
    loss = F.mse_loss(x_hat, x)   # 재구성 오차
    return loss
```

**흐름**:
```
1. 입력 이미지 x (784차원)
   ↓
2. Encoder → z (2차원)
   ↓
3. Decoder → x_hat (784차원)
   ↓
4. Loss = ||x - x_hat||²
   ↓
5. Backpropagation으로 가중치 업데이트
```

### 4. 최적화

```python
def configure_optimizers(self):
    optimizer = torch.optim.AdamW(
        self.parameters(),
        lr=1e-3,
        weight_decay=0.1
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=100
    )
    return [optimizer], [scheduler]
```

- **AdamW**: Adam + Weight Decay (L2 정규화)
- **CosineAnnealingLR**: Learning rate를 코사인 함수로 조정

---

## 주요 질문과 답변

### Q1: `forward()` 메서드가 없는 이유는?

**A**: Lightning에서 `forward()`는 **선택사항**입니다.

**두 가지 스타일**:

#### 스타일 1: `forward()` 없음 (현재 AutoEncoder)
```python
class AutoEncoder(L.LightningModule):
    def training_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)      # 직접 호출
        x_hat = self.decoder(z)  # 직접 호출
        loss = F.mse_loss(x_hat, x)
        return loss
```

**특징**:
- ✅ 간단한 모델에 적합
- ❌ 코드 중복 (training/validation/test에서 반복)

#### 스타일 2: `forward()` 있음 (추천)
```python
class AutoEncoder(L.LightningModule):
    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)
        x_hat, z = self(x)  # forward() 호출
        loss = F.mse_loss(x_hat, x)
        return loss
```

**특징**:
- ✅ 코드 중복 제거
- ✅ 모델을 함수처럼 사용: `output = model(input)`
- ✅ 복잡한 모델에 적합

**결론**: 현재 AutoEncoder는 간단하기 때문에 `forward()` 없이 구현되었고, 이는 완전히 정상입니다.

### Q2: AutoEncoder의 한계는?

**A**: AutoEncoder는 **생성 모델로서 한계**가 있습니다.

**문제점**:

1. **불규칙한 잠재 공간**
```python
# 학습 데이터의 잠재 벡터
z_train = [
    [1.5, 2.3],   # "7"
    [-3.1, 0.5],  # "1"
    [0.2, -2.8],  # "3"
]

# 새로운 점에서 샘플링
z_new = [0.0, 0.0]  # 학습 데이터와 멀리 떨어진 점
x_generated = decoder(z_new)  # ❌ 의미없는 출력
```

2. **보간의 어려움**
```python
z_7 = encoder(image_7)  # [1.5, 2.3]
z_1 = encoder(image_1)  # [-3.1, 0.5]

# 선형 보간
z_mid = 0.5 * z_7 + 0.5 * z_1  # [-0.8, 1.4]
x_mid = decoder(z_mid)  # ❌ 자연스럽지 않은 결과
```

**해결**: VAE를 사용하면 잠재 공간이 규칙적이어서 이러한 문제 해결

### Q3: AutoEncoder를 언제 사용하나?

**A**: 다음 경우에 AutoEncoder가 적합합니다:

✅ **추천하는 경우**:
- **차원 축소**: PCA 대신 비선형 차원 축소
- **특징 추출**: 다운스트림 태스크를 위한 표현 학습
- **노이즈 제거**: Denoising AutoEncoder
- **이상 탐지**: 재구성 오차가 큰 샘플 찾기
- **압축**: 데이터 압축

❌ **추천하지 않는 경우**:
- **새로운 데이터 생성**: VAE나 GAN 사용
- **잠재 공간 탐색**: VAE 사용
- **보간**: VAE 사용

### Q4: 잠재 차원을 어떻게 선택하나?

**A**: Trade-off가 있습니다.

| 잠재 차원 | 장점 | 단점 |
|---------|-----|-----|
| **작음 (2-10)** | 시각화 가능<br>강력한 압축 | 정보 손실<br>재구성 품질 낮음 |
| **중간 (50-100)** | 균형적 | - |
| **큼 (200+)** | 정보 보존<br>재구성 품질 높음 | 시각화 불가<br>압축 효과 적음 |

**현재 설정**:
```yaml
model:
  latent_dim: 2  # 2D 시각화용
```

**실전 팁**:
- 시각화 목적: 2-3차원
- 실용 목적: Validation loss를 보고 결정
- 시작점: 원본 차원의 10% 정도

### Q5: MSE vs BCE Loss?

**A**: 데이터 특성에 따라 선택합니다.

**MSE (Mean Squared Error)**:
```python
loss = F.mse_loss(x_hat, x)
```
- ✅ 연속 값 이미지 (그레이스케일, RGB)
- ✅ 큰 오차에 민감
- ❌ 이진 이미지에는 부적합

**BCE (Binary Cross Entropy)**:
```python
loss = F.binary_cross_entropy(x_hat, x)
```
- ✅ 이진 이미지 ([0, 1] 범위)
- ✅ 확률적 해석 가능
- ❌ Decoder에 Sigmoid 필요

**현재 AutoEncoder는 MSE 사용**:
```python
# auto_encoder.py Line 18
loss = F.mse_loss(x_hat, x)
```

**VAE는 BCE 사용**:
```python
# mnist_vae.py Line 142
recon_loss = F.binary_cross_entropy(x_hat, x)
```

### Q6: Encoder와 Decoder의 구조는 대칭적이어야 하나?

**A**: 꼭 그럴 필요는 없지만, **대칭적인 것이 일반적**입니다.

**현재 구조 (대칭)**:
```python
Encoder: 784 → 256 → 2
Decoder: 2 → 256 → 784
```

**비대칭 예시**:
```python
Encoder: 784 → 512 → 256 → 128 → 2  (더 깊음)
Decoder: 2 → 128 → 784              (더 얕음)
```

**일반적 관례**:
- ✅ 대칭 구조: 이해하기 쉽고, 안정적인 학습
- ⚡ 비대칭: 특정 목적에 따라 (예: Encoder를 더 깊게)

---

## 하이퍼파라미터 설정

```yaml
# configs/autoencoder.yaml

defaults:
  - common: default_setup  # 공통 설정 로드
  - _self_                 # 현재 파일의 설정이 우선함

tensorboard:
  name: "mnist-autoencoder"

data:
  data_dir: "~/data/"
  batch_size: 32
  target_labels: [0, 1, 3, 8]  # 학습할 숫자

model:
  latent_dim: 2      # 잠재 공간 차원
  hidden_dim: 256    # 히든 레이어 차원

training:
  max_epochs: 2
  lr: 1e-3
  weight_decay: 0.1
  T_max: 100  # CosineAnnealingLR

trainer:
  devices: "auto"
  log_every_n_steps: 5
  val_check_interval: 1.0
  enable_progress_bar: true
  enable_model_summary: true
```

**주요 파라미터**:

- **latent_dim**: 잠재 공간 차원
  - 작을수록: 강력한 압축, 정보 손실
  - 클수록: 정보 보존, 약한 압축

- **hidden_dim**: 히든 레이어 크기
  - 작을수록: 빠른 학습, 표현력 제한
  - 클수록: 높은 표현력, 과적합 위험

- **lr (learning rate)**: 학습률
  - `1e-3`: 일반적인 시작점
  - 너무 크면: 불안정한 학습
  - 너무 작으면: 느린 수렴

- **weight_decay**: L2 정규화
  - `0.1`: 강한 정규화
  - `0.01`: 중간 정규화
  - `0`: 정규화 없음

---

## 실전 팁

### 1. 특정 숫자만 학습

```bash
# 0, 1, 3, 8만 학습
python train_autoencoder.py data.target_labels=[0,1,3,8]
```

이렇게 하면:
- 더 빠른 학습
- 더 선명한 잠재 공간 구조
- 특정 클래스에 대한 더 나은 재구성

### 2. 과적합 방지

```python
# Dropout 추가
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Dropout(0.2),  # ← 추가
            nn.Linear(256, 2)
        )
```

### 3. 더 깊은 네트워크

```python
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
```

### 4. 재구성 품질 향상

```yaml
# 더 많은 에포크
training:
  max_epochs: 50  # 2 → 50

# 더 큰 배치
data:
  batch_size: 128  # 32 → 128

# 더 큰 히든 레이어
model:
  hidden_dim: 512  # 256 → 512
```

---

## 트러블슈팅

### 재구성 품질이 나쁨

**증상**: x_hat이 흐릿하고 원본과 많이 다름

**원인 및 해결**:
1. **잠재 차원이 너무 작음**
   ```yaml
   model:
     latent_dim: 10  # 2 → 10
   ```

2. **학습이 부족**
   ```yaml
   training:
     max_epochs: 50  # 2 → 50
   ```

3. **Learning rate 조정**
   ```yaml
   training:
     lr: 5e-4  # 1e-3 → 5e-4 (더 안정적)
   ```

### Loss가 수렴하지 않음

**증상**: train_loss가 계속 진동

**해결**:
```yaml
training:
  lr: 1e-4  # Learning rate 낮추기
  weight_decay: 0.01  # 정규화 강화
```

### 잠재 공간이 의미 없음

**증상**: 시각화했을 때 클래스 구분 안 됨

**원인**: AutoEncoder는 클래스 정보를 사용하지 않음 (비지도 학습)

**해결**:
1. Supervised Autoencoder 사용
2. VAE 사용 (더 규칙적인 잠재 공간)
3. 특정 클래스만 학습: `target_labels=[0,1]`

---

## 참고 자료

- 원논문: [Reducing the Dimensionality of Data with Neural Networks (Hinton & Salakhutdinov, 2006)](https://www.science.org/doi/10.1126/science.1127647)
- Denoising Autoencoder: [Vincent et al., 2008](http://www.jmlr.org/papers/v11/vincent10a.html)
- PyTorch Lightning 문서: https://lightning.ai/docs/pytorch/stable/

---

## 관련 문서

- [README_mnist_vae.md](./README_mnist_vae.md) - VAE 구현 및 AutoEncoder와의 비교
- [README_lightning.md](./README_lightning.md) - Lightning 프로젝트 전체 구조

---

## 라이센스

MIT License
