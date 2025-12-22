# MNIST VAE (Variational AutoEncoder)

MNIST 손글씨 숫자 데이터셋을 위한 Variational AutoEncoder (VAE) 구현입니다.

## 목차
- [개요](#개요)
- [AutoEncoder vs VAE](#autoencoder-vs-vae)
- [VAE의 확률적 이해](#vae의-확률적-이해)
- [프로젝트 구조](#프로젝트-구조)
- [사용 방법](#사용-방법)
- [VAE 핵심 개념](#vae-핵심-개념)
- [주요 질문과 답변](#주요-질문과-답변)

---

## 개요

VAE는 **생성 모델(Generative Model)**로, 다음 두 가지를 동시에 수행합니다:
1. **압축**: 고차원 이미지 (28×28=784차원) → 저차원 잠재 벡터 (2차원)
2. **생성**: 잠재 공간에서 샘플링 → 새로운 이미지 생성

---

## AutoEncoder vs VAE

VAE는 일반 AutoEncoder의 확률적 버전으로, 더 강력한 생성 능력을 가집니다.

### 핵심 차이점

| 특징 | AutoEncoder | VAE |
|-----|------------|-----|
| **모델 타입** | 결정론적 (Deterministic) | 확률적 (Probabilistic) |
| **Encoder 출력** | z (벡터) | μ, σ² (분포 파라미터) |
| **잠재 공간** | 불규칙적 | 규칙적 (N(0,I)) |
| **Loss** | Reconstruction only | Reconstruction + KL |
| **생성 능력** | ❌ 제한적 | ✅ 우수 |
| **보간** | ❌ 어려움 | ✅ 부드러움 |
| **용도** | 압축, 특징 추출 | 생성, 샘플링 |

### 시각적 비교

```
AutoEncoder 잠재 공간:
   "7"
              "1"

              "3"
     "8"
         "0"
❌ 불규칙적으로 분포
❌ 빈 공간에서 샘플링 → 의미없는 출력
❌ 보간 시 부자연스러운 결과

VAE 잠재 공간:
        "1"
    "1"    "1"
         0    "7"
    "3"    "7"
        "3"
✅ 원점 중심으로 밀집
✅ 어디서 샘플링해도 의미있는 출력
✅ 부드러운 보간 가능
```

### 코드 비교

**AutoEncoder**:
```python
class AutoEncoder(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()  # z = encoder(x)
        self.decoder = Decoder()  # x_hat = decoder(z)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)        # 결정론적 인코딩
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)  # Reconstruction만
        return loss
```

**VAE**:
```python
class MnistVAE(L.LightningModule):
    def __init__(self):
        super().__init__()
        # encoder → mu, logvar
        # decoder → x_hat

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)
        x_hat, mu, logvar = self(x)  # 확률적 인코딩

        # Reconstruction Loss
        recon_loss = F.binary_cross_entropy(x_hat, x)

        # KL Divergence (잠재 공간 정규화)
        kl_loss = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))

        loss = recon_loss + kl_loss  # Reconstruction + KL
        return loss
```

### 상세 비교

#### 1. Encoder의 차이

**AutoEncoder**:
```python
z = encoder(x)  # 단일 벡터 출력
# z = [1.2, 0.5]  (결정론적)
```

**VAE**:
```python
mu, logvar = encoder(x)  # 분포 파라미터 출력
# mu = [1.2, 0.5], logvar = [-0.5, -0.3]

# Reparameterization trick
z = mu + eps * exp(0.5 * logvar)  # 확률적 샘플링
# z = [1.15, 0.48]  (매번 다름)
```

#### 2. Loss Function의 차이

**AutoEncoder**:
```python
Loss = MSE(x_hat, x)
     = ||x - x_hat||²

목적: 재구성만 잘하면 됨
```

**VAE**:
```python
Loss = Reconstruction Loss + KL Divergence
     = -log p(x|z) + KL(q(z|x) || p(z))

목적:
1. 재구성을 잘하고 (Reconstruction)
2. 잠재 공간을 규칙적으로 만듦 (KL)
```

#### 3. 생성 능력의 차이

**AutoEncoder**:
```python
# ❌ 생성이 어려움
z_random = torch.randn(1, 2)  # [0.5, -0.3]
x_generated = decoder(z_random)
# → 학습하지 않은 영역이면 노이즈 출력
```

**VAE**:
```python
# ✅ 생성 가능
z_random = torch.randn(1, 2)  # p(z) = N(0, I)에서 샘플링
x_generated = decoder(z_random)
# → 항상 의미있는 이미지 생성
```

#### 4. 보간의 차이

**AutoEncoder**:
```python
z_7 = encoder(image_7)  # [10.5, 8.3]
z_1 = encoder(image_1)  # [-5.2, 3.1]

z_mid = 0.5 * z_7 + 0.5 * z_1  # [2.65, 5.7]
x_mid = decoder(z_mid)
# ❌ 학습하지 않은 영역 → 부자연스러운 결과
```

**VAE**:
```python
z_7 = encoder(image_7)[0]  # mu만 사용: [1.2, 0.5]
z_1 = encoder(image_1)[0]  # [-0.8, 1.3]

z_mid = 0.5 * z_7 + 0.5 * z_1  # [0.2, 0.9]
x_mid = decoder(z_mid)
# ✅ 학습된 영역 → 자연스러운 "7"과 "1" 사이 이미지
```

### 왜 VAE가 더 나은가?

**AutoEncoder의 문제**:
```python
학습 데이터:
- image_7 → z = [10.5, 8.3]
- image_1 → z = [-5.2, 3.1]
- image_3 → z = [3.8, -7.2]

문제:
1. z가 사방팔방으로 흩어짐
2. [0, 0] 같은 중간 지점은 학습 안 됨
3. 샘플링/보간 불가능
```

**VAE의 해결**:
```python
학습 데이터 (KL loss 덕분):
- image_7 → z ~ N([1.2, 0.5], [0.1, 0.1])
- image_1 → z ~ N([-0.8, 1.3], [0.1, 0.1])
- image_3 → z ~ N([0.5, -0.9], [0.1, 0.1])

장점:
1. z가 원점 주변에 밀집
2. 모든 영역이 학습됨
3. 샘플링/보간 가능
```

### 언제 무엇을 사용할까?

**AutoEncoder 사용**:
- ✅ 차원 축소 (PCA 대체)
- ✅ 특징 추출
- ✅ 노이즈 제거
- ✅ 이상 탐지
- ✅ 빠른 학습 필요

**VAE 사용**:
- ✅ 새로운 데이터 생성
- ✅ 잠재 공간 탐색
- ✅ 보간
- ✅ 확률적 모델링
- ✅ 데이터 증강

---

## VAE의 확률적 이해

### 전체 확률 모델

```
┌─────────────────────────────────────────────┐
│  VAE의 확률적 관점                            │
├─────────────────────────────────────────────┤
│                                             │
│  Prior (사전 분포):                          │
│    p(z) = N(0, I)                           │
│    "잠재 공간이 표준 정규분포를 따르길 원함"    │
│                                             │
│  Encoder (추론 모델):                        │
│    q(z|x) = N(μ(x), σ²(x))                 │
│    "입력 x가 주어졌을 때, z의 분포 추정"       │
│                                             │
│  Decoder (생성 모델):                        │
│    p(x|z) = ∏ᵢ Bernoulli(xᵢ | x_hatᵢ(z))  │
│    "잠재 벡터 z로부터 이미지 x 생성"          │
│                                             │
└─────────────────────────────────────────────┘
```

### 각 확률 분포의 의미

#### 1. **p(z) = N(0, I)** - Prior (사전 분포)

```python
# 표준 정규분포
p(z) = N(0, I)
```

- **의미**: 잠재 공간이 어떻게 생겼으면 좋겠는지에 대한 사전 가정
- **목적**:
  - 잠재 공간을 연속적이고 규칙적으로 만듦
  - 비슷한 데이터는 비슷한 z 값을 가지도록
  - 잠재 공간에서 샘플링해도 의미있는 데이터 생성 가능

#### 2. **q(z|x) = N(μ(x), σ²(x))** - Encoder

```python
# Encoder의 출력
mu, logvar = encoder(x)
# q(z|x) = N(mu, exp(logvar))
```

- **의미**: 입력 x가 주어졌을 때, 어떤 잠재 벡터 z에서 나왔을지 추정
- **예시**:
  - 숫자 "7" 이미지 → `q(z|x) = N(μ=[1.2, 0.5], σ²=[0.1, 0.1])`
  - 숫자 "1" 이미지 → `q(z|x) = N(μ=[-0.8, 1.3], σ²=[0.1, 0.1])`

#### 3. **p(x|z) = ∏ᵢ Bernoulli(xᵢ | x_hatᵢ)** - Decoder

```python
# Decoder의 출력
x_hat = decoder(z)  # [0.9, 0.1, 0.8, ..., 0.3]  (784개)

# 각 픽셀의 베르누이 분포
p(x[i]=1|z) = x_hat[i]
p(x[i]=0|z) = 1 - x_hat[i]

# 전체 이미지 (784개 독립 베르누이의 곱)
p(x|z) = ∏ᵢ₌₁⁷⁸⁴ Bernoulli(xᵢ | x_hatᵢ(z))
```

- **의미**: 잠재 벡터 z로부터 이미지 x를 생성하는 분포
- **독립 가정**: 각 픽셀은 z가 주어졌을 때 서로 독립
- **파라미터 수**: 784개 (각 픽셀마다 1개의 확률값)

---

## 프로젝트 구조

```
lightning/
├── src/
│   ├── models/
│   │   └── mnist_vae.py          # VAE 모델 구현
│   └── data/
│       └── MNIST_data.py          # 데이터 로더
├── configs/
│   └── mnist_vae.yaml             # Hydra 설정
├── train_mnist_vae.py             # 학습 스크립트
├── notebooks/
│   └── visualize_mnist_vae.ipynb  # 시각화 노트북
└── docs/
    └── README_mnist_vae.md        # 이 문서
```

---

## 사용 방법

### 1. 학습

```bash
python train_mnist_vae.py
```

설정 변경:
```bash
# 에포크 수 변경
python train_mnist_vae.py training.max_epochs=100

# 잠재 차원 변경
python train_mnist_vae.py model.latent_dim=10

# KL weight 변경
python train_mnist_vae.py model.kl_weight=0.5
```

### 2. 시각화

```bash
jupyter notebook notebooks/visualize_mnist_vae.ipynb
```

**노트북 사용법**:
```python
# Cell 2: 실행할 run_timestamp 지정
run_timestamp = None  # 가장 최근 실행 자동 선택
# run_timestamp = "2025-12-20/16-44-09"  # 특정 실행 지정
```

노트북은 자동으로 Hydra 실행 디렉토리(`saved/hydra_logs/{run_timestamp}/checkpoints/`)에서 최적 체크포인트를 로드합니다.

포함된 시각화:
1. **잠재 공간 시각화** - 2D 산점도로 각 숫자의 분포 확인
2. **재구성 결과** - 원본 vs 재구성 이미지 비교
3. **새로운 이미지 생성** - 잠재 공간에서 샘플링하여 생성
4. **잠재 공간 보간** - 두 이미지 사이를 부드럽게 변환
5. **잠재 공간 그리드** - 잠재 공간 전체를 균등 샘플링하여 시각화

### 3. 체크포인트 로드

```python
from src.models.mnist_vae import MnistVAE

# PyTorch 2.6+ 호환성: weights_only=False 필요
# Hydra 실행 디렉토리 기반 경로
vae = MnistVAE.load_from_checkpoint(
    "saved/hydra_logs/2025-12-20/16-44-09/checkpoints/mnist-vae-epoch=48-val_loss=140.23.ckpt",
    weights_only=False  # OmegaConf ListConfig 로드를 위해 필요
)
```

**왜 `weights_only=False`가 필요한가?**
- PyTorch 2.6부터 `torch.load`의 기본값이 `weights_only=True`로 변경
- 체크포인트에 저장된 `hidden_dims: [256, 128]` 같은 OmegaConf ListConfig를 로드하려면 `weights_only=False` 필요
- 신뢰할 수 있는 체크포인트에만 사용

---

## VAE 핵심 개념

### 1. Loss Function

```python
Total Loss = Reconstruction Loss + KL Divergence
```

#### Reconstruction Loss

```python
# Binary Cross Entropy
recon_loss = F.binary_cross_entropy(x_hat, x)

# 의미: -log p(x|z)
# "디코더가 z에서 원본 x를 얼마나 잘 복원하는가?"
```

#### KL Divergence

```python
kl_loss = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))

# 의미: KL(q(z|x) || p(z))
# "인코더의 출력 분포 q(z|x)와 사전 분포 p(z) 사이의 차이"
```

**왜 KL Divergence가 필요한가?**

KL loss가 없다면:
```python
❌ 각 이미지가 잠재 공간에서 멀리 떨어진 고립된 점으로 매핑
❌ 분산이 0에 가까워져 확률적 샘플링 불가능
❌ 생성 모델로서 쓸모없음 (학습 데이터만 외움)
```

KL loss가 있으면:
```python
✅ 모든 이미지가 원점 주변에 밀집
✅ 분산 ≈ 1 유지하여 확률적 샘플링 가능
✅ p(z) = N(0,I)에서 샘플링하면 의미있는 이미지 생성
✅ 잠재 공간에서 보간 가능
```

### 2. Reparameterization Trick

**문제**: 확률 분포에서 샘플링은 미분 불가능
```python
❌ z ~ N(mu, sigma^2)  # 미분 불가능
   loss.backward()     # Gradient가 mu, sigma로 전파 안 됨
```

**해결**: Reparameterization
```python
✅ eps ~ N(0, 1)       # 고정된 분포에서 샘플링
   z = mu + eps * sigma  # 미분 가능한 연산
   loss.backward()    # Gradient가 mu, sigma로 전파됨!
```

```python
def reparameterize(self, mu, logvar):
    if self.training:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std  # z = μ + ε·σ
    else:
        z = mu  # 추론 시에는 평균값만 사용
    return z
```

### 3. Decoder 출력: 샘플링 vs 기댓값

**질문**: Decoder가 p(x|z)를 생성하는데, 왜 그 분포에서 샘플링을 안 하는가?

#### 학습 시 (Training)

```python
# ❌ 샘플링하면 안 됨:
x_hat = decoder(z)           # [0.9, 0.1, 0.8, ...]
x_sampled = torch.bernoulli(x_hat)  # [1, 0, 1, ...]
loss = BCE(x_sampled, x)     # ❌ 미분 불가능!

# ✅ 확률값 직접 사용:
x_hat = decoder(z)           # [0.9, 0.1, 0.8, ...]
loss = BCE(x_hat, x)         # ✅ 미분 가능
```

**이유**: `bernoulli()` 샘플링은 미분 불가능 → 학습 불가

#### 추론 시 (Inference)

선택 가능:

**방법 1: 기댓값 사용 (기본)**
```python
x_hat = decoder(z)  # [0.9, 0.1, 0.8, ...]
# 그대로 사용 (회색조 이미지, 더 부드러움)
```

**방법 2: 샘플링**
```python
x_hat = decoder(z)
x_sampled = torch.bernoulli(x_hat)  # [1, 0, 1, ...]
# 완전한 흑백 이미지 (더 선명, 약간 노이즈)
```

보통은 **기댓값 사용**이 더 나은 결과를 제공합니다.

### 4. Encoder vs Decoder 샘플링 비교

| | Encoder (z) | Decoder (x) |
|---|---|---|
| **학습 시** | ✅ 샘플링 필수<br>(reparameterization) | ❌ 샘플링 안 함<br>(확률값 직접 사용) |
| **추론 시** | ✅ 샘플링 또는 평균 | ⚡ 선택 가능<br>(보통 확률값 사용) |
| **이유** | 미분 가능하게 샘플링 | 미분 필요 없음<br>기댓값이 더 안정적 |

---

## 주요 질문과 답변

### Q1: `x_hat`은 무엇인가?

**A**: `x_hat`은 **재구성된 이미지(Reconstructed image)**입니다.

```python
x      # 원본 이미지
x_hat  # 재구성/예측된 이미지 (x̂)

# 흐름:
x → Encoder → z → Decoder → x_hat
```

수학/통계에서 hat(^) 기호는 추정값(estimate) 또는 예측값(prediction)을 나타냅니다.

### Q2: q(z|x)와 p(z)의 의미는?

**A**:
- **p(z) = N(0, I)**: 사전 분포 (Prior)
  - 잠재 공간이 표준 정규분포를 따르길 원한다는 가정
  - 데이터 x와 무관

- **q(z|x) = N(μ(x), σ²(x))**: 근사 사후 분포 (Approximate Posterior)
  - 특정 입력 x가 주어졌을 때, 어떤 z에서 나왔을지 추정
  - 데이터 x에 의존적

**KL(q(z|x) || p(z))의 의미**:
"인코더가 예측한 분포와 우리가 원하는 사전 분포 사이의 차이"를 최소화하여 잠재 공간을 규칙적으로 만듦.

### Q3: 각 데이터 x마다 너무 다른 z 분포를 만들지 않도록 제약하는 이유는?

**A**: **생성 가능성**을 위해서입니다.

KL loss가 없으면:
```
잠재 공간:
   "1"(-777, -666)

                                 "7"(999, 888)
         원점(0, 0)

"3"(555, -444)

❌ 각 데이터가 고립된 섬처럼 존재
❌ p(z) = N(0,I)에서 샘플링 → 학습하지 않은 영역 → 노이즈만 생성
❌ 보간 불가능
```

KL loss가 있으면:
```
잠재 공간:
        "1"
    "1"    "1"
         0    "7"
    "3"    "7"
        "3"

✅ 원점 주변에 밀집
✅ p(z) = N(0,I)에서 샘플링 → 학습된 영역 → 의미있는 이미지 생성
✅ 보간 가능
```

### Q4: p(x|z)는 784개 변수를 갖는 베르누이 분포인가?

**A**: 정확히는 **784개의 독립적인 베르누이 분포의 곱**입니다.

```python
p(x|z) = ∏ᵢ₌₁⁷⁸⁴ Bernoulli(xᵢ | x_hatᵢ(z))

# 각 픽셀 i:
p(x[i]=1|z) = x_hat[i]
p(x[i]=0|z) = 1 - x_hat[i]

# 독립 가정:
# 각 픽셀은 z가 주어졌을 때 서로 독립
```

**다변량 베르누이 vs 독립 베르누이**:
- ❌ 다변량 베르누이: 픽셀 간 상관관계 모델링 (파라미터: 2^784 - 1개)
- ✅ 독립 베르누이: 픽셀 간 독립 가정 (파라미터: 784개)

**Log-likelihood와 BCE의 관계**:
```python
log p(x|z) = Σᵢ [xᵢ log(x_hatᵢ) + (1-xᵢ) log(1-x_hatᵢ)]
BCE(x_hat, x) = -log p(x|z)
```

### Q5: Bernoulli vs Gaussian 분포 선택 기준은?

**A**: 데이터 타입에 따라 선택합니다.

| 분포 가정 | 출력 활성화 | Loss 함수 | 데이터 타입 | 예시 |
|---------|-----------|----------|-----------|------|
| **Bernoulli** | Sigmoid | BCE | 이진/그레이스케일 | MNIST |
| **Gaussian** | Identity/Tanh | MSE | 연속 값 이미지 | CelebA |

**MNIST VAE는 Bernoulli**:
```python
# Decoder
decoder_layers.append(nn.Sigmoid())  # [0, 1] 범위

# Loss
recon_loss = F.binary_cross_entropy(x_hat, x)
```

**Gaussian이라면**:
```python
# Decoder
decoder_layers.append(nn.Tanh())  # 또는 Identity

# Loss
recon_loss = F.mse_loss(x_hat, x)
# = -log p(x|z) under Gaussian assumption: p(x|z) = N(x | x_hat(z), I)
```

---

## 하이퍼파라미터 설정

```yaml
# configs/mnist_vae.yaml

defaults:
  - common: default_setup  # 공통 설정 로드
  - _self_                 # 현재 파일의 설정이 우선함

# 모델 이름 (Hydra 로그 디렉토리 구조에 사용)
model_name: "mnist-vae"

data:
  data_dir: "~/data/"
  batch_size: 128
  target_labels: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # 모든 숫자

model:
  input_dim: 784          # 28*28
  hidden_dims: [256, 128] # Encoder/Decoder 히든 레이어 차원
  latent_dim: 2           # 잠재 공간 차원 (2D 시각화용)
  kl_weight: 1.0          # KL loss 가중치

training:
  max_epochs: 50
  lr: 1e-3
  weight_decay: 0.1
  T_max: 100  # CosineAnnealingLR

trainer:
  devices: "auto"
  log_every_n_steps: 10
  val_check_interval: 1.0
  enable_progress_bar: true
  enable_model_summary: true
```

**주요 파라미터 설명**:

- **latent_dim**: 잠재 공간 차원
  - `2`: 2D 시각화 가능, 표현력 제한적
  - `10-50`: 더 복잡한 데이터 표현 가능

- **kl_weight**: KL loss 가중치 (β-VAE)
  - `< 1`: 재구성에 집중, 잠재 공간 규칙성 감소
  - `= 1`: 표준 VAE
  - `> 1`: 잠재 공간 규칙성 강조, disentanglement 증가

- **hidden_dims**: 네트워크 용량
  - 더 큰 값: 더 좋은 표현력, 과적합 위험
  - 더 작은 값: 빠른 학습, 표현력 제한

---

## 참고 자료

- 원논문: [Auto-Encoding Variational Bayes (Kingma & Welling, 2014)](https://arxiv.org/abs/1312.6114)
- Tutorial: [Understanding VAE](https://arxiv.org/abs/1606.05908)
- PyTorch Lightning 문서: https://lightning.ai/docs/pytorch/stable/

---

## 트러블슈팅

### 체크포인트 로드 오류

**오류**:
```
WeightsUnpickler error: Unsupported global: GLOBAL omegaconf.listconfig.ListConfig
```

**해결**:
```python
vae = MnistVAE.load_from_checkpoint(path, weights_only=False)
```

### KL Loss가 0으로 수렴

**증상**: KL loss가 0, 재구성은 좋지만 생성 불가

**원인**: KL weight가 너무 작거나 0

**해결**: `kl_weight`를 1.0으로 설정

### 생성된 이미지가 흐릿함

**원인**: 정상입니다. VAE는 확률적 평균을 출력

**선택사항**:
1. 그대로 사용 (보통 더 나음)
2. 샘플링: `torch.bernoulli(x_hat)`으로 흑백 이미지 생성

---

## 라이센스

MIT License
