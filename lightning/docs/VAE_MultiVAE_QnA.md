# VAE와 MultiVAE 이해도 점검 - Q&A 정리

이 문서는 VAE(Variational AutoEncoder)와 MultiVAE(Collaborative Filtering용 VAE)에 대한 핵심 개념들을 Q&A 형식으로 정리한 것입니다.

---

## 목차

### [1. Reparameterization Trick](#1-reparameterization-trick-1)
- Q1: Reparameterization trick이 왜 필요한가?
- Q2: `torch.exp(0.5 * logvar)`를 계산하는 이유는?

### [2. KL Divergence](#2-kl-divergence-1)
```python
kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
```
- Q1: KL Loss 수식이 계산하는 것은?
- Q2: KL Loss의 역할은?
- Q3: `dim=1`에 대해 sum을 하고 `.mean()`을 하는 이유는?

### [3. Reconstruction Loss (MultiVAE)](#3-reconstruction-loss-multivae-1)
- Q1: 왜 Softmax를 사용하나?
- Q2: 이 수식이 실제로 계산하는 것은?
- Q3: MultiVAE의 입력 x는?

### [4. KL Annealing](#4-kl-annealing-1)
- Q1: KL Annealing이 왜 필요한가?
- Q2: 학습 단계별 효과는?
- Q3: Posterior Collapse란?
- Q4: `global_step`과 `kl_anneal_steps`는 어떻게 관리되나?

### [5. Training vs Inference](#5-training-vs-inference-1)
- Q1: Inference 시 왜 z = mu만 사용하나?
- Q2: Input Normalization의 목적은?
- Q3: Encoder Input에 Dropout을 적용하는 이유는?

### [6. Log-Softmax 특성](#6-log-softmax-특성-1)
- Q: log_softmax()는 항상 음수인가?
- Q: Preference score가 0이면 -inf가 되나?
- Reconstruction Loss와의 관계

### [7. MultiVAE vs MNIST VAE](#7-multivae-vs-mnist-vae-1)
- 주요 차이점
- 왜 이런 차이가?

### [8. 전체 Flow](#8-전체-flow-1)
- Training 시 Forward Pass
- Inference 시 추천 생성

---

## 1. Reparameterization Trick

### Q1: Reparameterization trick이 왜 필요한가?

**A:** Backpropagation을 가능하게 하기 위해서입니다.

- 단순히 `N(mu, var)`에서 직접 샘플링하면 **미분 불가능**
- `z = mu + eps * std` (eps ~ N(0,1))로 변환하면:
  - 확률적 요소는 `eps`에만 존재 (파라미터와 무관)
  - `mu`와 `std`는 **결정론적 연산**으로 전달
  - Gradient가 `mu`, `logvar`까지 전파 가능

```python
def reparameterize(self, mu, logvar):
    if self.training:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)  # 실제 샘플링은 여기서만!
        z = mu + eps * std
    else:
        z = mu  # 추론 시에는 평균값 사용
    return z
```

### Q2: `torch.exp(0.5 * logvar)`를 계산하는 이유는?

**A:** Standard deviation(std)을 구하기 위해서입니다.

**수식 유도:**
```
std = sqrt(var) = var^(1/2)

log(std) = log(var^(1/2)) = (1/2) * log(var)

std = exp(log(std)) = exp(0.5 * log(var)) = exp(0.5 * logvar)
```

**왜 log space를 사용?**
- 수치적 안정성 향상
- Variance가 매우 크거나 작을 때도 안전하게 처리

---
[목차](#목차)

## 2. KL Divergence

### Q1: KL Loss 수식이 계산하는 것은?

```python
kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
```

**A:** `q(z|x) = N(mu, var)`와 `p(z) = N(0, I)` 간의 KL divergence입니다.

### Q2: KL Loss의 역할은?

**A:** 일반화된 특성을 찾아 인코딩하도록 regularization하는 역할입니다.

- **KL loss 없으면**: Encoder가 각 입력을 latent space의 임의의 먼 지점으로 매핑
  - 입력 데이터에 과적합 (overfitting)
  - Latent space가 너무 퍼져있어서 일반화 실패

- **KL loss 있으면**: Latent 분포를 `N(0, I)`에 가깝게 제약
  - 서로 다른 입력들이 비슷한 영역에 매핑
  - 연속적이고 의미 있는 latent space 형성
  - 생성 모델로 작동 가능

### Q3: `dim=1`에 대해 sum을 하고 `.mean()`을 하는 이유는?

**A:** Shape 때문입니다.

- `mu`, `logvar`: `(batch_size, latent_dim)` shape
- `dim=1`: latent dimension 방향으로 sum → 각 샘플의 scalar KL loss
- `.mean()`: batch 전체에 대한 평균 loss

---
[목차](#목차)

## 3. Reconstruction Loss (MultiVAE)

### Q1: 왜 Softmax를 사용하나?

```python
log_softmax = F.log_softmax(logits, dim=1)
recon_loss = -(log_softmax * x).sum(dim=1).mean()
```

**A:** MultiVAE는 user-item interaction을 **multinomial distribution**으로 모델링하기 때문입니다.

- `logits`: 각 item별 선택 점수
- `softmax`: 각 item이 선택될 확률로 변환 (item 간 상대적 확률)
- `log_softmax`: 수치적 안정성 + negative log-likelihood와 직접 연결

### Q2: 이 수식이 실제로 계산하는 것은?

**A:** **Multinomial Negative Log-Likelihood**입니다.

- `log_softmax(logits)`: 각 item이 선택될 log 확률
- `x`: 실제 상호작용 벡터 (binary)
- `(log_softmax * x).sum(dim=1)`: 실제 상호작용한 item들의 log 확률의 합
- 음수를 붙여서: maximize log-likelihood = minimize negative log-likelihood

**의미:** 모델이 예측한 확률 분포에서 실제 상호작용한 item들의 확률을 최대화

### Q3: MultiVAE의 입력 x는?

**A:** User-item interaction vector

- Shape: `(batch_size, num_items)`
- 값: **Binary (0 or 1)** - implicit feedback
- 예시: `x = [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, ..., 0]`
  - 1: 상호작용 있음 (클릭, 구매 등)
  - 0: 상호작용 없음

**데이터 생성 코드:**
```python
def _build_user_item_matrix(self, df):
    rows = df["user"].values
    cols = df["item"].values
    data = np.ones(len(df))  # ← 모두 1로 설정 (binary)

    mat = sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(self.num_users, self.num_items),
    )
    return mat
```

---
[목차](#목차)

## 4. KL Annealing

### Q1: KL Annealing이 왜 필요한가?

```python
def _kl_weight(self):
    return min(
        self.kl_max_weight,
        self.kl_max_weight * self.global_step / self.kl_anneal_steps,
    )

loss = recon_loss + self._kl_weight() * kl_loss
```

**A:**
1. **학습 속도**: 처음에는 reconstruction loss만으로 빠르게 학습
2. **Posterior Collapse 방지**: KL weight가 처음부터 크면 encoder가 항상 N(0,I)만 출력

### Q2: 학습 단계별 효과는?

**초기 (KL weight ≈ 0):**
- Reconstruction 위주 학습
- Encoder/Decoder가 데이터를 재구성하는 법을 먼저 배움
- Latent space가 의미 있는 정보를 담도록 학습

**후기 (KL weight → max):**
- 이미 reconstruction 능력이 있는 상태에서
- Latent space를 표준 정규분포로 정규화
- 일반화 능력 향상

### Q3: Posterior Collapse란?

**A:** 만약 처음부터 KL weight가 크면:
- Encoder가 KL loss 최소화를 위해 항상 `N(0, I)` 출력 (mu=0, logvar=0)
- Decoder가 latent vector를 무시하고 평균적인 출력만 생성
- VAE가 작동하지 않는 상태

**KL Annealing 효과:**
- Encoder가 먼저 의미 있는 latent representation을 학습
- 점진적으로 정규화를 강화

### Q4: `global_step`과 `kl_anneal_steps`는 어떻게 관리되나?

**A:**
- `self.global_step`: PyTorch Lightning이 **자동으로 관리**
  - 매 training step마다 자동으로 +1 증가
  - 직접 업데이트 불필요

- `self.kl_anneal_steps`: **하이퍼파라미터** (고정값)
  - `__init__`에서 설정되고 변하지 않음
  - 목표 step 수 (예: 20000)

**동작 예시:**
```python
# Step 0: kl_weight = min(0.2, 0.2 * 0 / 20000) = 0
# Step 10000: kl_weight = min(0.2, 0.2 * 10000 / 20000) = 0.1
# Step 20000: kl_weight = min(0.2, 0.2 * 20000 / 20000) = 0.2
# Step 30000: kl_weight = min(0.2, 0.2 * 30000 / 20000) = 0.2 (capped)
```

---
[목차](#목차)

## 5. Training vs Inference

### Q1: Inference 시 왜 z = mu만 사용하나?

**A:**
- **Training**: 샘플링으로 stochasticity 추가 → 일반화, robust한 latent space
- **Inference**: `z = mu` 사용 → 가장 대표적인(expected) latent vector
  - 확률론적으로 `E[z] = mu`
  - 평균값이 가장 안정적인 예측 제공

### Q2: Input Normalization의 목적은?

```python
x = F.normalize(x, p=2, dim=1)  # L2 normalization
```

**A:** User마다 상호작용 개수가 다르기 때문에 공정하게 처리하기 위해서입니다.

**L2 Normalization 계산:**
```python
# 원본 x (3개 item과 상호작용, 총 100개 item)
x = [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, ..., 0]

# L2 norm 계산
||x||_2 = sqrt(1^2 + 1^2 + 1^2) = sqrt(3) ≈ 1.732

# Normalization 후
x_normalized = x / sqrt(3)
             = [0, 0.577, 0, 0, 0, 0.577, 0, 0, 0, 0, 0.577, 0, ..., 0]
```

**효과:**
- 벡터의 크기가 항상 1
- 상호작용 개수에 따라 각 값이 조정됨:
  - 3개 상호작용: 각 값 ≈ 0.577
  - 100개 상호작용: 각 값 = 0.1
- **Power user vs Casual user를 공정하게 학습**
- 상호작용의 절대 개수가 아닌 **"패턴"에 집중**

### Q3: Encoder Input에 Dropout을 적용하는 이유는?

**A:** 일반화를 위해서입니다 (Denoising AutoEncoder 효과).

- 입력의 일부 item을 랜덤하게 masking
- 모델이 일부 정보만으로도 재구성하도록 학습
- **Collaborative Filtering 특성**: 비슷한 user들의 패턴을 학습
- **Overfitting 방지**: 특정 user의 모든 interaction을 암기하지 않음

---
[목차](#목차)

## 6. Log-Softmax 특성

### Q: log_softmax()는 항상 음수인가?

**A:** 예, 항상 음수입니다.

```python
# Softmax 출력 범위: (0, 1]
# Log 특성: log(x) where 0 < x ≤ 1 → 항상 음수

logits = [2.0, 1.0, 0.5]
softmax = [0.659, 0.242, 0.099]
log_softmax = [-0.416, -1.416, -2.316]  # 모두 음수!
```

### Q: Preference score가 0이면 -inf가 되나?

**A:** 거의 그렇습니다.

```python
# logits = 0이고 다른 logits가 크면
softmax(0) ≈ 0 (매우 작은 값)
log_softmax(0) → -∞ (음의 무한대에 가까움)
```

### Reconstruction Loss와의 관계

```python
log_softmax = F.log_softmax(logits, dim=1)  # 모두 음수
recon_loss = -(log_softmax * x).sum(dim=1).mean()
#            ↑ 음수를 붙여서 양수로 만듦!

# 예시:
# x = [1, 0, 0, 0, 1, ...]
# log_softmax = [-0.5, -3.2, -4.1, -2.8, -1.2, ...]
# log_softmax * x = [-0.5, 0, 0, 0, -1.2, ...]
# sum = -1.7
# recon_loss = -(-1.7) = 1.7 (양수!)
```

**핵심:**
- `log_softmax()`: 항상 음수
- Log_softmax가 **-0.1** (높은 확률) → 좋은 예측
- Log_softmax가 **-10** (낮은 확률) → 나쁜 예측
- Loss는 앞에 `-` 붙여서 최소화 문제로 변환

---
[목차](#목차)

## 7. MultiVAE vs MNIST VAE

### 주요 차이점

| 항목 | MNIST VAE | MultiVAE |
|------|-----------|----------|
| **출력 분포** | 784개 독립 Bernoulli | 1개 Multinomial |
| **Loss** | Binary Cross-Entropy | Multinomial NLL |
| **의미** | 각 픽셀 복원 | Item 선택 확률 분포 매칭 |
| **Input Normalization** | 선택적 (픽셀 값 이미 [0,1]) | 필수 (상호작용 개수 다름) |
| **Input Dropout** | 선택적 (DAE로 사용 가능) | 필수 (sparse data 일반화) |

### 왜 이런 차이가?

**MNIST VAE:**
- 픽셀들이 서로 **독립적으로** 켜지거나 꺼짐
- 각 픽셀: "이 위치가 검은색일 확률"
- 모든 샘플이 같은 차원 (28×28)

**MultiVAE:**
- Item들이 **상호 경쟁적** (user의 관심이 제한적)
- "User가 이 item들 중 어떤 것과 상호작용할 확률"
- User마다 상호작용 수가 크게 다름 → Normalization 필수

---
[목차](#목차)

## 8. 전체 Flow

### Training 시 Forward Pass

**입력 예시:**
```python
# User가 item 1, 5, 10과 상호작용 (총 100개 item)
x = [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, ..., 0]  # shape: (100,)
```

**단계별 처리:**

1. **Input → Encoder**
   ```python
   x = F.normalize(x, p=2, dim=1)  # L2 normalization
   x = F.dropout(x, self.dropout, training=True)
   h = self.encoder(x)  # Hidden layers
   ```

2. **Encoder → (mu, logvar)**
   ```python
   mu = self.mu(h)      # (batch_size, latent_dim)
   logvar = self.logvar(h)  # (batch_size, latent_dim)
   # 각 user를 N(mu, var) 분포로 인코딩
   ```

3. **(mu, logvar) → z**
   ```python
   std = torch.exp(0.5 * logvar)
   eps = torch.randn_like(std)  # eps ~ N(0, I)
   z = mu + eps * std  # Reparameterization trick
   ```

4. **z → Decoder**
   ```python
   logits = self.decoder(z)  # (batch_size, num_items)
   # 각 item별 선택 확률(점수)
   ```

5. **logits → Loss**
   ```python
   # Reconstruction Loss
   log_softmax = F.log_softmax(logits, dim=1)
   recon_loss = -(log_softmax * x).sum(dim=1).mean()
   # x 중 1인 item만 해당 log_softmax 값을 합산

   # KL Loss
   kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

   # Total Loss
   loss = recon_loss + kl_weight * kl_loss
   # Backpropagation
   ```

### Inference 시 추천 생성

```python
# 1. Forward pass (no sampling)
with torch.no_grad():
    logits, mu, logvar = model(x)  # z = mu (no randomness)
    scores = logits

# 2. 이미 상호작용한 item 제외
scores[x.bool()] = -float('inf')

# 3. Top-K 추천
top_k_items = torch.topk(scores, k=10).indices
```

**왜 Top-K 방식?**
- **Sampling**: 확률적으로 다양하지만 낮은 확률 item도 선택될 수 있음
- **Top-K**: 가장 높은 점수 K개 선택 → 실용적, 안정적

---
[목차](#목차)

## 핵심 개념 요약

✅ **Reparameterization trick** - Backpropagation을 위한 결정론적 변환

✅ **KL divergence** - Latent space 정규화로 일반화

✅ **Multinomial NLL** - Item 선택 확률 분포 모델링

✅ **KL Annealing** - Posterior collapse 방지

✅ **L2 normalization + Dropout** - Sparse data 특성 대응

✅ **MultiVAE vs MNIST VAE** - 출력 분포 차이 (Multinomial vs Bernoulli)

✅ **Binary Implicit Feedback** - Count가 아닌 0/1로 상호작용 표현

✅ **Top-K Inference** - 가장 높은 점수의 item 추천

---

## 참고 코드 위치

- MultiVAE 모델: [lightning/src/models/multi_vae.py](lightning/src/models/multi_vae.py)
- 데이터 모듈: [lightning/src/data/recsys_data.py](lightning/src/data/recsys_data.py)
- Reparameterization: [multi_vae.py:104-122](lightning/src/models/multi_vae.py#L104-L122)
- KL Annealing: [multi_vae.py:155-160](lightning/src/models/multi_vae.py#L155-L160)
- Reconstruction Loss: [multi_vae.py:172-173](lightning/src/models/multi_vae.py#L172-L173)
