# Multi-VAE (Collaborative Filtering VAE)

협업 필터링(Collaborative Filtering)을 위한 Variational AutoEncoder 구현입니다.

## 목차
- [개요](#개요)
- [Collaborative Filtering VAE의 이해](#collaborative-filtering-vae의-이해)
- [프로젝트 구조](#프로젝트-구조)
- [사용 방법](#사용-방법)
- [핵심 개념](#핵심-개념)
- [주요 질문과 답변](#주요-질문과-답변)

---

## 개요

Multi-VAE는 **암시적 피드백(Implicit Feedback)** 데이터를 위한 추천 시스템 모델입니다.

### 기본 아이디어

```
유저-아이템 상호작용 → VAE → 잠재 선호도 → 아이템 추천
```

**입력**: 유저의 클릭/시청 이력 (Binary Vector)
```python
user_123 = [1, 0, 1, 0, 0, 1, ...]  # 6931개 아이템
           ↑     ↑           ↑
        본 영화  안본 영화  본 영화
```

**출력**: 각 아이템에 대한 점수
```python
scores = [0.9, 0.1, 0.8, 0.7, 0.6, 0.3, ...]
          ↑                    ↑
      추천 1순위            추천 5순위
```

### MNIST VAE와의 차이점

| 특징 | MNIST VAE | Multi-VAE |
|-----|-----------|-----------|
| **입력** | 이미지 (784 픽셀) | 유저-아이템 벡터 (N개 아이템) |
| **출력 분포** | Bernoulli | Multinomial |
| **Loss** | BCE | Multinomial log-likelihood |
| **목적** | 이미지 생성 | 아이템 추천 |
| **Dropout** | ❌ 없음 | ✅ 입력에 적용 |
| **Normalization** | ❌ 없음 | ✅ L2 정규화 |
| **KL Annealing** | ❌ 고정 가중치 | ✅ 점진적 증가 |

---

## Collaborative Filtering VAE의 이해

### 1. 문제 정의

**목표**: 유저의 과거 상호작용을 바탕으로 미래 선호도 예측

```
입력 데이터:
┌──────────┬────────┬────────┬────────┬─────┐
│  User ID │ Item 1 │ Item 2 │ Item 3 │ ... │
├──────────┼────────┼────────┼────────┼─────┤
│   user_1 │   1    │   0    │   1    │ ... │
│   user_2 │   0    │   1    │   0    │ ... │
│   user_3 │   1    │   1    │   0    │ ... │
└──────────┴────────┴────────┴────────┴─────┘

1: 상호작용 있음 (클릭, 시청, 구매 등)
0: 상호작용 없음
```

### 2. 확률 모델

```
┌─────────────────────────────────────────────┐
│  Multi-VAE의 확률적 관점                      │
├─────────────────────────────────────────────┤
│                                             │
│  Prior (사전 분포):                          │
│    p(z_u) = N(0, I)                         │
│    "유저의 잠재 선호도는 표준 정규분포"        │
│                                             │
│  Encoder (추론 모델):                        │
│    q(z_u|x_u) = N(μ(x_u), σ²(x_u))         │
│    "유저 u의 상호작용 이력으로 선호도 추정"    │
│                                             │
│  Decoder (생성 모델):                        │
│    p(x_u|z_u) = Multinomial(π(z_u))        │
│    "선호도 z_u로부터 아이템 클릭 확률 예측"   │
│                                             │
└─────────────────────────────────────────────┘
```

### 3. Multinomial Likelihood

**MNIST VAE (Bernoulli)**:
```python
# 각 픽셀이 독립적인 베르누이 분포
p(x|z) = ∏ᵢ Bernoulli(xᵢ | πᵢ)

# 픽셀 i가 1일 확률
p(xᵢ=1|z) = πᵢ
```

**Multi-VAE (Multinomial)**:
```python
# 전체 클릭 횟수가 고정된 다항 분포
p(x|z) = Multinomial(x | π(z))

# 전체 클릭 중 아이템 i가 차지하는 비율
π(z) = softmax(f(z))

# 예: 유저가 총 10개 아이템 클릭
# π = [0.2, 0.1, 0.3, 0.4]
# → 아이템 1: 2번, 아이템 2: 1번, 아이템 3: 3번, 아이템 4: 4번
```

**왜 Multinomial인가?**
- 유저의 **총 상호작용 횟수가 중요한 정보**
- "10개 본 유저"와 "100개 본 유저"는 다르게 취급
- 아이템 간 **상대적 선호도**를 모델링

### 4. Loss Function

```python
# Reconstruction Loss (Negative Multinomial Log-Likelihood)
recon_loss = -(log_softmax(logits) * x_u).sum(dim=1).mean()

# KL Divergence
kl_loss = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))

# Total Loss with KL Annealing
loss = recon_loss + kl_weight * kl_loss
```

**KL Annealing**:
```python
# 학습 초기: kl_weight ≈ 0
# → Reconstruction에 집중
# → 유저 선호도를 잘 학습

# 학습 후반: kl_weight → 0.2 (kl_max_weight)
# → 잠재 공간 정규화
# → 일반화 성능 향상

kl_weight = min(kl_max_weight,
                kl_max_weight * global_step / kl_anneal_steps)
```

---

## 프로젝트 구조

```
lightning/
├── src/
│   ├── models/
│   │   └── multi_vae.py           # Multi-VAE 모델
│   ├── data/
│   │   └── recsys_data.py         # RecSys DataModule
│   └── utils/
│       ├── metrics.py             # Recall@K 메트릭
│       └── recommend.py           # Top-K 추천 생성
├── configs/
│   └── multi_vae.yaml             # Hydra 설정
├── train_multi_vae.py             # 학습 스크립트
└── docs/
    └── README_multi_vae.md        # 이 문서
```

---

## 사용 방법

### 1. 데이터 준비

```
~/data/train/train_ratings.csv

user,item,time
11,4643,1230768000
11,170,1230768000
11,531,1230768000
...
```

**필수 컬럼**:
- `user`: 유저 ID
- `item`: 아이템 ID
- `time`: 타임스탬프 (temporal split 사용 시 필수, 그 외는 선택)

### 2. 학습

```bash
python train_multi_vae.py
```

**Hydra Configuration 구조**:
```yaml
# configs/multi_vae.yaml
defaults:
  - logging: default
  - _self_

data:
  data_dir: "~/data/train/"
  batch_size: 512
  valid_ratio: 0.1
  min_interactions: 5
  split_strategy: "random"  # "random", "leave_one_out", "temporal_user", "temporal_global"
  temporal_split_ratio: 0.8  # temporal split용 (0.8 = 80% train, 20% valid)

model:
  hidden_dims: [600, 200]
  dropout: 0.5

training:
  max_epochs: 300
  lr: 1e-3
  weight_decay: 0.01
  kl_max_weight: 0.2
  kl_anneal_steps: 20000

trainer:
  devices: "auto"
  log_every_n_steps: 10
  val_check_interval: 1.0
  enable_progress_bar: true
  enable_model_summary: true

recommend:
  topk: 10
```

**주요 설정 변경**:
```bash
# 에포크 수
python train_multi_vae.py training.max_epochs=100

# KL annealing 속도
python train_multi_vae.py training.kl_anneal_steps=10000

# 최대 KL weight
python train_multi_vae.py training.kl_max_weight=0.5

# Hidden 레이어 크기
python train_multi_vae.py model.hidden_dims=[512,256,128]

# Dropout
python train_multi_vae.py model.dropout=0.3

# 데이터 디렉토리
python train_multi_vae.py data.data_dir=/path/to/data

# Batch size
python train_multi_vae.py data.batch_size=256

# Top-K 추천 개수
python train_multi_vae.py recommend.topk=20

# Validation split 전략
python train_multi_vae.py data.split_strategy=leave_one_out
python train_multi_vae.py data.split_strategy=temporal_user data.temporal_split_ratio=0.8
python train_multi_vae.py data.split_strategy=temporal_global data.temporal_split_ratio=0.8
```

### 3. 추천 생성

학습 완료 후 자동으로:
1. **Top-K 추천 생성** (기본: Top-10)
2. **Recall@K 계산** (Validation set)
3. **Submission 파일 생성** (`saved/hydra_logs/{run_timestamp}/submissions/submission.csv`)

**⚠️ Future Information Leakage 방지**: 추론 시 각 유저의 마지막 클릭 시점 이후 개봉한 영화는 자동으로 추천에서 제외됩니다.

```csv
user,item
11,123
11,456
11,789
...
```

### 4. 시각화

```bash
jupyter notebook notebooks/visualize_multi_vae.ipynb
```

**노트북 사용법**:
```python
# Cell 1: 모델 이름 상수 (이미 정의됨)
MODEL_NAME = "multi-vae"

# Cell 2: 실행할 run_timestamp 지정
run_timestamp = None  # 가장 최근 실행 자동 선택
# run_timestamp = "2025-12-20/16-44-09"  # 특정 실행 지정
```

노트북은 자동으로 Hydra 실행 디렉토리(`saved/hydra_logs/{MODEL_NAME}/{run_timestamp}/checkpoints/`)에서 최적 체크포인트를 로드합니다.

포함된 시각화:
1. **Input Data**: User-Item 상호작용 행렬 및 분포
2. **Data Split**: 4가지 분할 전략 비교 (random, leave-one-out, temporal_user, temporal_global)
3. **Prediction**: Top-K 추천 결과 및 Recall@K 분석
4. **Latent Space**: 유저 잠재 표현 3D 시각화 및 Multivariate Normal Distribution
5. **Output Distribution**: Multinomial 출력 분포 entropy 분석

결과는 `saved/hydra_logs/{run_timestamp}/visualizations/`에 저장됩니다.

### 5. 모델 로드 및 추론

```python
from src.models.multi_vae import MultiVAE
from src.utils.recommend import recommend_topk
import torch

# 체크포인트 로드
# Hydra 실행 디렉토리 기반 경로
model = MultiVAE.load_from_checkpoint(
    "saved/hydra_logs/2025-12-20/16-44-09/checkpoints/multi-vae-epoch=99-val_loss=1127.14.ckpt",
    weights_only=False  # OmegaConf ListConfig 로드를 위해 필요
)
model.eval()

# 유저-아이템 행렬 (CSR sparse matrix)
# user_item_matrix[u, i] = 1 if user u interacted with item i

# Top-10 추천 생성
recommendations = recommend_topk(
    model,
    user_item_matrix,
    k=10,
    device="cuda"
)

# recommendations[user_idx] = [item_1, item_2, ..., item_10]
```

---

## 핵심 개념

### 1. 모델 구조

```python
class MultiVAE(L.LightningModule):
    def __init__(
        self,
        num_items,              # 아이템 개수 (입력/출력 차원)
        hidden_dims=[600, 200], # Encoder/Decoder 히든 레이어
        dropout=0.5,            # Dropout 비율
        ...
    ):
        # Encoder: num_items → 600 → 200 (with Tanh)
        # mu, logvar: 200 → 200
        # Decoder: 200 → 600 → num_items (with Tanh)
```

**동적 레이어 생성**:
```python
# Encoder
for hidden_dim in hidden_dims:  # [600, 200]
    layers.append(Linear(input_dim, hidden_dim))
    layers.append(Tanh())
    input_dim = hidden_dim

# Decoder (역순)
for hidden_dim in reversed_dims:  # [200, 600]
    layers.append(Linear(input_dim, hidden_dim))
    layers.append(Tanh())
    input_dim = hidden_dim
```

### 2. Forward Pass

```python
def forward(self, x):
    # 1. L2 정규화
    x = F.normalize(x, p=2, dim=1)

    # 2. Dropout (학습 시에만)
    x = F.dropout(x, self.dropout, training=self.training)

    # 3. Encoder
    h = self.encoder(x)
    mu = self.mu(h)
    logvar = self.logvar(h)

    # 4. Reparameterization
    z = mu + eps * exp(0.5 * logvar)

    # 5. Decoder
    logits = self.decoder(z)

    return logits, mu, logvar
```

**L2 정규화**:
```python
# 각 유저 벡터를 단위 벡터로 변환
x = [10, 0, 5, 0, 3]  # 원본
x_norm = x / ||x||₂   # L2 norm으로 나눔
x_norm = [0.82, 0, 0.41, 0, 0.25]  # 정규화 후

# 목적: 상호작용 횟수가 많은/적은 유저를 동등하게 취급
```

**Dropout**:
```python
# 학습 시 입력의 일부를 0으로 만듦
x = [1, 0, 1, 1, 0]
x_dropped = [1, 0, 0, 1, 0]  # 50% dropout
            ↑        ↑
         keep    dropped

# 목적: 과적합 방지, 일반화 성능 향상
```

### 3. KL Annealing

```python
def _kl_weight(self):
    return min(
        self.kl_max_weight,
        self.kl_max_weight * self.global_step / self.kl_anneal_steps
    )

# global_step:     0     5000   10000   15000   20000
# kl_weight:      0.0    0.05    0.1    0.15     0.2
#                 ↓       ↓       ↓       ↓       ↓
#              재구성   점진적   균형    점진적   정규화
#              집중     증가     잡힘     증가     강화
```

**왜 필요한가?**
- 학습 초기: KL loss가 너무 강하면 posterior collapse
  - `q(z|x) ≈ p(z) = N(0,I)` (모든 유저가 같은 분포)
  - 유저별 차이 학습 실패
- 해결: 초기엔 reconstruction에 집중, 점진적으로 KL 추가

### 4. 추천 생성

```python
def recommend_topk(model, train_mat, k=10):
    # 1. 모델 추론
    logits, _, _ = model(user_item_vector)
    scores = logits  # (num_users, num_items)

    # 2. 이미 본 아이템 제외
    scores[train_mat.nonzero()] = -inf

    # 3. Top-K 추출
    topk_indices = argsort(-scores, axis=1)[:, :k]

    return topk_indices
```

### 5. 평가 메트릭: Recall@K

```python
def recall_at_k(actual_list, pred_list, k=10):
    """
    Recall@K = (추천한 K개 중 정답 개수) / min(K, 실제 정답 개수)
    """
    recalls = []
    for actual, pred in zip(actual_list, pred_list):
        actual_set = set(actual)
        pred_k = pred[:k]

        hits = len(actual_set.intersection(pred_k))
        denom = min(k, len(actual))

        recalls.append(hits / denom)

    return mean(recalls)
```

**예시**:
```python
# 유저 123의 validation ground truth
actual = [10, 25, 87, 99, 102]  # 5개 아이템

# 모델의 Top-10 추천
pred = [25, 33, 10, 44, 55, 66, 77, 88, 99, 100]
       ↑        ↑                        ↑
      정답     정답                     정답

# Recall@10 = 3 / min(10, 5) = 3 / 5 = 0.6
```

---

## 주요 질문과 답변

### Q1: Multinomial vs Bernoulli의 실제 차이는?

**A**: **총 클릭 횟수 정보**를 사용하는지 여부입니다.

**Bernoulli (MNIST VAE)**:
```python
user_a = [1, 0, 1, 0, 1]  # 3개 클릭
user_b = [1, 0, 1, 0, 1]  # 3개 클릭

# 두 유저가 동일하게 취급됨
p(x|z) = ∏ Bernoulli(xᵢ|πᵢ)
```

**Multinomial (Multi-VAE)**:
```python
user_a = [5, 0, 3, 0, 2]  # 총 10개 클릭
user_b = [1, 0, 1, 0, 1]  # 총 3개 클릭

# 두 유저가 다르게 취급됨
# user_a는 아이템 1을 50% 비율로 클릭
# user_b는 아이템 1을 33% 비율로 클릭

p(x|z) = Multinomial(x | π(z))
```

**Multi-VAE의 장점**:
- 활동적인 유저 vs 비활동적인 유저 구분
- 아이템 간 상대적 선호도 파악
- 더 풍부한 정보 활용

### Q2: L2 정규화는 왜 필요한가?

**A**: **유저 간 상호작용 횟수 차이를 보정**하기 위해서입니다.

```python
# 정규화 전
user_a = [100, 0, 50, 0, 30]  # 활동적인 유저 (180개 클릭)
user_b = [1, 0, 1, 0, 1]      # 비활동적인 유저 (3개 클릭)

# 정규화 후
user_a = [0.82, 0, 0.41, 0, 0.25]  # ||x|| = 1
user_b = [0.58, 0, 0.58, 0, 0.58]  # ||x|| = 1

# 모델이 "클릭 횟수"가 아닌 "선호 비율"을 학습
```

**효과**:
- 활동적인 유저가 학습을 지배하는 것 방지
- 모든 유저를 공평하게 취급
- 일반화 성능 향상

### Q3: Dropout은 왜 입력에만 적용하나?

**A**: **협업 필터링 특성상** 입력 노이즈가 가장 효과적이기 때문입니다.

```python
# 원본 입력
x = [1, 0, 1, 1, 0, 1, ...]

# Dropout 적용 (p=0.5)
x_dropped = [0, 0, 1, 1, 0, 0, ...]
             ↑              ↑
          dropped        dropped

# 모델은 "일부 상호작용이 가려진" 상태에서 학습
# → 보지 않은 아이템을 더 잘 예측하게 됨
```

**효과**:
- Denoising AutoEncoder처럼 동작
- 과적합 방지
- 일반화 성능 향상

**히든 레이어 Dropout과의 차이**:
- 히든 레이어 Dropout: 모델 복잡도 제어
- 입력 Dropout: 데이터 증강 효과 + Denoising

### Q4: KL Annealing 없이 학습하면?

**A**: **Posterior Collapse**가 발생할 수 있습니다.

```python
# Posterior Collapse:
q(z|x) ≈ p(z) = N(0, I)  # 모든 유저가 같은 분포

# 모든 유저의 mu ≈ 0, logvar ≈ 0
# → 유저별 차이를 학습하지 못함
# → 모델이 평균적인 추천만 생성
```

**KL Annealing의 효과**:
```python
# Step 1-5000: kl_weight ≈ 0
# → Reconstruction에 집중
# → 각 유저의 선호도를 mu, logvar에 학습

# Step 5000-20000: kl_weight 점진적 증가
# → 잠재 공간 정규화
# → 일반화 성능 향상
```

### Q5: hidden_dims를 어떻게 설정하나?

**A**: **아이템 개수와 Trade-off**를 고려합니다.

```python
# 기본 설정 (논문 추천)
hidden_dims: [600, 200]

# 아이템이 많으면 (10000개 이상)
hidden_dims: [1024, 512, 256]

# 아이템이 적으면 (1000개 이하)
hidden_dims: [256, 128]
```

**고려사항**:
- 너무 큰 히든 차원: 과적합, 느린 학습
- 너무 작은 히든 차원: 표현력 부족, 낮은 성능
- 일반적으로 `[입력 차원의 10%, 입력 차원의 3%]` 정도

### Q6: Validation split은 어떻게 하나?

**A**: **4가지 분할 전략**을 지원합니다.

#### 1. Random Split (기본값)
```python
# 유저별 랜덤하게 valid_ratio 비율만큼 분할
split_strategy: "random"
valid_ratio: 0.1

# 유저 123의 전체 상호작용
all_items = [10, 25, 33, 87, 99, 102, 150, 200, 301, 400]

# 10% validation split (랜덤)
train_items = [10, 33, 87, 99, 102, 150, 200, 301]  # 90%
valid_items = [25, 400]                             # 10%
```

**장점**:
- 유저별로 공정한 분할
- Cold-start 시뮬레이션
- 시간 정보 불필요

#### 2. Leave-One-Out
```python
# 유저별로 랜덤하게 1개만 validation
split_strategy: "leave_one_out"

# 유저 123의 전체 상호작용
all_items = [10, 25, 33, 87, 99, 102, 150, 200, 301, 400]

# 1개만 validation (랜덤)
train_items = [10, 25, 33, 87, 99, 102, 150, 200, 301]  # N-1개
valid_items = [400]                                      # 1개
```

**장점**:
- 추천시스템 연구에서 표준적으로 사용
- 빠른 평가
- 모든 유저가 동일한 validation 개수

**단점**:
- Validation set이 작음
- 분산이 클 수 있음

#### 3. Temporal User Split
```python
# 각 유저별로 시간순 정렬 후 temporal_split_ratio 기준 분할
split_strategy: "temporal_user"
temporal_split_ratio: 0.8

# 유저 123의 시간순 상호작용
all_items = [10, 25, 33, 87, 99, 102, 150, 200, 301, 400]
timestamps = [t1, t2, t3, t4, t5,  t6,  t7,  t8,  t9,  t10]
                                   ↑ 80% 지점

# 유저별 시간 기준 80% train, 20% valid
train_items = [10, 25, 33, 87, 99, 102, 150, 200]  # 시간순 앞 80%
valid_items = [301, 400]                           # 시간순 뒤 20%
```

**장점**:
- 유저별 시간적 패턴 학습 가능
- 모든 유저가 train/valid에 포함
- Cold-start 문제 없음

**단점**:
- 데이터에 `time` 컬럼 필요
- 유저 간 시간이 섞임 (유저 A의 미래가 유저 B의 과거일 수 있음)

#### 4. Temporal Global Split
```python
# 전체 데이터를 시간순 정렬 후 temporal_split_ratio 기준 분할
split_strategy: "temporal_global"
temporal_split_ratio: 0.8

# 전체 데이터 시간순 정렬
# 2024-01-01 ~ 2024-12-31

# 전역 시간 기준 80% train, 20% valid
train: 2024-01-01 ~ 2024-10-01 (80%)
valid: 2024-10-02 ~ 2024-12-31 (20%)
```

**장점**:
- 진짜 시간적 일반화 (과거→미래)
- 실제 서비스 환경과 가장 유사
- 더욱 현실적인 평가

**단점**:
- 데이터에 `time` 컬럼 필요
- Validation에 없는 신규 유저/아이템 발생 가능 (cold-start)
- 최근 활동이 적은 유저는 validation이 없을 수 있음

**주의**: Temporal split 사용 시 데이터에 `time` 컬럼이 반드시 있어야 합니다.

#### 설정 예시

```bash
# Random split (기본값)
python train_multi_vae.py data.split_strategy=random data.valid_ratio=0.1

# Leave-One-Out
python train_multi_vae.py data.split_strategy=leave_one_out

# Temporal User split (유저별 시간)
python train_multi_vae.py data.split_strategy=temporal_user data.temporal_split_ratio=0.8

# Temporal Global split (전역 시간)
python train_multi_vae.py data.split_strategy=temporal_global data.temporal_split_ratio=0.8
```

---

## 하이퍼파라미터 가이드

```yaml
# configs/multi_vae.yaml

defaults:
  - common: default_setup  # 공통 설정 로드
  - _self_                 # 현재 파일(autoencoder.yaml)의 설정이 우선함

# 모델 이름 (Hydra 로그 디렉토리 구조에 사용)
model_name: "multi-vae"

data:
  batch_size: 512           # 큰 배치가 안정적
  valid_ratio: 0.1          # 10% validation (random split용)
  min_interactions: 5       # 최소 상호작용 필터링
  data_file: "train_ratings.csv"
  split_strategy: "random"  # "random", "leave_one_out", "temporal_user", "temporal_global"
  temporal_split_ratio: 0.8 # 80% train, 20% valid (temporal split용)

model:
  hidden_dims: [600, 200]   # 히든 레이어 크기
  dropout: 0.5              # 강한 정규화

training:
  max_epochs: 300           # 충분한 학습
  lr: 1e-3                  # Adam 기본값
  weight_decay: 0.01        # L2 정규화
  kl_max_weight: 0.2        # 최대 KL 가중치
  kl_anneal_steps: 20000    # Annealing 스텝 수

recommend:
  topk: 10                  # Top-10 추천
```

**주요 파라미터 조정**:

1. **dropout** (0.3 ~ 0.7)
   - 높을수록: 강한 정규화, 낮은 train loss
   - 낮을수록: 약한 정규화, 과적합 위험

2. **kl_max_weight** (0.1 ~ 0.5)
   - 높을수록: 규칙적인 잠재 공간, 낮은 재구성 품질
   - 낮을수록: 좋은 재구성, 덜 규칙적인 잠재 공간

3. **kl_anneal_steps** (10000 ~ 50000)
   - 작을수록: 빠른 annealing, posterior collapse 위험
   - 클수록: 느린 annealing, 안정적 학습

---

## 성능 향상 팁

### 1. 데이터 전처리

```python
# 너무 적은 상호작용을 가진 유저 제거
min_interactions: 5  # → 10

# 너무 인기 없는 아이템 제거
# (별도 전처리 스크립트 필요)
```

### 2. 모델 용량 증가

```yaml
model:
  hidden_dims: [1024, 512, 256]  # [600, 200] → 더 깊게
  dropout: 0.3                    # 0.5 → 약하게 (용량 증가 시)
```

### 3. KL Annealing 조정

```yaml
training:
  kl_max_weight: 0.1     # 0.2 → 낮추기 (재구성 우선)
  kl_anneal_steps: 30000 # 20000 → 늘리기 (천천히)
```

### 4. 학습률 스케줄링

```python
# ReduceLROnPlateau 사용 (코드에 이미 적용됨)
scheduler = ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,    # 절반으로 감소
    patience=15    # 15 epoch 개선 없으면
)
```

---

## 트러블슈팅

### Recall@K가 너무 낮음 (< 0.1)

**원인 및 해결**:

1. **KL weight가 너무 큼**
   ```yaml
   training:
     kl_max_weight: 0.1  # 0.2 → 0.1
   ```

2. **Dropout이 너무 강함**
   ```yaml
   model:
     dropout: 0.3  # 0.5 → 0.3
   ```

3. **학습 부족**
   ```yaml
   training:
     max_epochs: 500  # 300 → 500
   ```

### Loss가 수렴하지 않음

**원인 및 해결**:

1. **Learning rate가 너무 높음**
   ```yaml
   training:
     lr: 5e-4  # 1e-3 → 5e-4
   ```

2. **Batch size 조정**
   ```yaml
   data:
     batch_size: 256  # 512 → 256 (작게)
   ```

### Validation loss는 좋은데 Recall이 낮음

**원인**: Validation loss와 Recall@K는 다른 목적 함수

**해결**:
```yaml
# Reconstruction에 더 집중
training:
  kl_max_weight: 0.05  # 낮추기

# Dropout 약화
model:
  dropout: 0.2
```

### OOM (Out of Memory) 에러

**해결**:
```yaml
data:
  batch_size: 256  # 512 → 256

# 또는 작은 모델
model:
  hidden_dims: [256, 128]  # [600, 200] → 작게
```

---

## 참고 자료

- 원논문: [Variational Autoencoders for Collaborative Filtering (Liang et al., 2018)](https://arxiv.org/abs/1802.05814)
- RecBole 구현: https://github.com/RUCAIBox/RecBole
- Cornac 구현: https://github.com/PreferredAI/cornac

---

## Future Information Leakage 방지

### 개요

영화 추천 시스템에서 **시간적 정보 유출(Future Information Leakage)** 문제를 방지하기 위해, 사용자가 아직 알 수 없는 미래 정보(개봉 전 영화)를 추천에서 자동으로 제외합니다.

### 동작 원리

```
사용자의 마지막 클릭: 2018년 5월
↓
개봉년도가 2019년 이후인 영화들은 추천에서 제외
(사용자가 2018년에 2019년 개봉 영화를 알 수 없음)
```

### 구현 세부사항

#### 1. 데이터 로딩 ([src/data/recsys_data.py](../src/data/recsys_data.py))

```python
# years.tsv에서 영화 개봉년도 로드
self.item_years = {}  # Dict[item_idx, year]

# 사용자별 마지막 클릭 년도 계산
self.user_last_click_years = {}  # Dict[user_idx, year]
```

**필요한 파일**:
- `years.tsv`: 영화 ID와 개봉년도 매핑 (탭 구분)
- `train_ratings.csv`: `time` 컬럼 필수 (Unix timestamp)

#### 2. Future Items 필터링

```python
def get_future_item_sequences(self):
    """
    각 유저별로 추천에서 제외할 future items 반환

    Returns:
        Dict[user_idx, Set[item_idx]]
    """
    future_items = {}
    for user_idx in range(num_users):
        last_click_year = self.user_last_click_years[user_idx]
        # 개봉년도 > 마지막 클릭 년도인 영화 필터링
        future_items[user_idx] = {
            item_idx for item_idx in range(num_items)
            if self.item_years[item_idx] > last_click_year
        }
    return future_items
```

#### 3. 추천 함수에 적용 ([src/utils/recommend.py](../src/utils/recommend.py))

```python
def recommend_topk(model, train_mat, k=10, exclude_items=None):
    """
    Args:
        exclude_items: Dict[user_idx, Set[item_idx]]
                      추천에서 제외할 아이템 (future items 등)
    """
    # 모델 추론
    logits, _, _ = model(batch_tensor)
    scores = logits.cpu().numpy()

    # 이미 본 아이템 제외
    scores[batch_mat.nonzero()] = -np.inf

    # Future items 제외 (year filtering)
    if exclude_items is not None:
        for batch_idx, user_idx in enumerate(range(start_idx, end_idx)):
            if user_idx in exclude_items:
                future_items = list(exclude_items[user_idx])
                if future_items:
                    scores[batch_idx, future_items] = -np.inf

    # Top-K 추출
    topk_indices = np.argsort(-scores, axis=1)[:, :k]
    return topk_indices
```

#### 4. 추론 시 자동 적용 ([predict_multi_vae.py](../predict_multi_vae.py))

```python
# Future items 가져오기
future_item_sequences = datamodule.get_future_item_sequences()

# Validation 추천 (future items 제외)
recommendations_valid = recommend_topk(
    model,
    train_mat,
    k=topk,
    device=device,
    batch_size=batch_size,
    exclude_items=future_item_sequences  # 제외 목록
)

# Submission 추천 (future items 제외)
recommendations_submission = recommend_topk(
    model,
    full_mat,
    k=topk,
    device=device,
    batch_size=batch_size,
    exclude_items=future_item_sequences  # 제외 목록
)
```

### 로그 예시

```
[INFO] Step 5/5: Loading item metadata...
[INFO] Loaded 6807 items with release year info
[INFO] Mapped 6807 items to release years
[INFO] Calculated last click year for 31360 users
[INFO] Getting future item sequences for year filtering...
[INFO] Future items to filter: 289456 items across 28934 users
```

### 예시

**사용자 A**:
- 마지막 클릭: `2018-05-15` → 년도: `2018`
- 추천 후보: `[어벤져스: 엔드게임 (2019), 겨울왕국 2 (2019), ...]`
- **결과**: 2019년 이후 개봉 영화 모두 제외 ✅

**사용자 B**:
- 마지막 클릭: `2020-12-01` → 년도: `2020`
- 추천 후보: `[어벤져스: 엔드게임 (2019), 겨울왕국 2 (2019), ...]`
- **결과**: 2019년 영화는 추천 가능 ✅

### 비활성화 방법

Future information leakage 방지를 비활성화하려면:

```python
# predict_multi_vae.py 수정
future_item_sequences = datamodule.get_future_item_sequences()

# 모든 유저에 대해 빈 set으로 변경
future_item_sequences = {u: set() for u in range(datamodule.num_users)}
```

### 데이터 요구사항

1. **years.tsv** (필수):
   ```tsv
   item	year
   1	1995
   2	1995
   3	1995
   ```

2. **train_ratings.csv** (time 컬럼 필수):
   ```csv
   user,item,time
   1,31,1260759144
   1,1029,1260759179
   ```

`time` 컬럼이 없으면 year filtering이 작동하지 않습니다.

---

## 관련 문서

- [README_mnist_vae.md](./README_mnist_vae.md) - 이미지 생성용 VAE
- [README_autoencoder.md](./README_autoencoder.md) - 기본 AutoEncoder

---

## 라이센스

MIT License
