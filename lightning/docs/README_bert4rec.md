# BERT4Rec Implementation Guide

## 개요

BERT4Rec (Bidirectional Encoder Representations from Transformers for Sequential Recommendation)의 PyTorch Lightning 구현입니다.

**논문**: [BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer](https://arxiv.org/abs/1904.06690)

## 파일 구조

```
lightning/
├── configs/
│   └── bert4rec.yaml              # Hydra 설정 파일
├── src/
│   ├── models/
│   │   ├── bert4rec.py            # BERT4Rec 모델 (LightningModule)
│   │   └── __init__.py
│   └── data/
│       ├── bert4rec_data.py       # DataModule
│       └── __init__.py
├── docs/
│   └── README_bert4rec.md         # 이 문서
├── train_bert4rec.py              # 학습 스크립트
├── predict_bert4rec.py            # 추론 스크립트
└── run_bert4rec.sh                # 실행 스크립트
```

## 사용법

### 1. 학습

```bash
# 기본 설정으로 학습
python train_bert4rec.py

# 하이퍼파라미터 오버라이드
python train_bert4rec.py model.hidden_units=128 training.lr=0.001

# 논문 기준 학습 (lr=0.001)
python train_bert4rec.py training.lr=0.001

# 여러 파라미터 동시 변경
python train_bert4rec.py \
    model.hidden_units=128 \
    model.num_heads=8 \
    model.num_layers=4 \
    training.num_epochs=500 \
    training.lr=0.001
```

### 2. 추론

```bash
# 마지막 체크포인트로 추론
python predict_bert4rec.py

# 특정 체크포인트 사용
python predict_bert4rec.py inference.checkpoint_path=saved/bert4rec/checkpoints/best.ckpt

# Top-K 변경
python predict_bert4rec.py inference.topk=20
```

### 3. 스크립트로 실행

```bash
# 학습 + 추론
./run_bert4rec.sh both

# 학습만
./run_bert4rec.sh train

# 추론만
./run_bert4rec.sh predict
```

## 설정 (configs/bert4rec.yaml)

### 데이터 설정

```yaml
data:
  data_dir: "~/data/train/"        # 데이터 디렉토리
  data_file: "train_ratings.csv"   # CSV 파일명
  batch_size: 128                  # 배치 크기
  min_interactions: 3              # 최소 interaction 수
  num_workers: 4                   # DataLoader workers
```

**데이터 포맷**: CSV with columns `user`, `item`, `time` (optional)

### 모델 설정

```yaml
model:
  hidden_units: 64        # Hidden dimension (논문: dataset-dependent)
  num_heads: 4            # Attention heads 수 (논문: dataset-dependent)
  num_layers: 3           # Transformer blocks 수 (논문: 2 for most datasets)
  max_len: 50             # 최대 시퀀스 길이 (논문: 200 for ML-1M)
  dropout_rate: 0.3       # Dropout 확률 (논문: 0.2~0.5)
  mask_prob: 0.15         # Masking 확률 (논문: 0.15, BERT와 동일)
  share_embeddings: true  # Output layer와 embedding 공유 (논문: Yes)
```

**논문의 하이퍼파라미터**:
- Hidden units: 64~256 (dataset-dependent)
- Attention heads: 2~8
- Transformer layers: 2~4
- Max sequence length: 200 (ML-1M), 50 (Steam)
- Dropout: 0.2~0.5
- Masking probability: 0.15 (BERT 논문과 동일)

**중요**: `hidden_units`는 `num_heads`로 나누어떨어져야 합니다.

### 학습 설정

```yaml
training:
  num_epochs: 300                  # 최대 epoch 수 (논문: early stopping 사용)
  lr: 0.0015                       # Learning rate (논문: 0.001)
  weight_decay: 0.0                # L2 regularization (논문: 명시 안됨)
  monitor_metric: "val_ndcg@10"    # 체크포인트 저장 기준
  early_stopping: false            # Early stopping (논문: 사용)
  early_stopping_patience: 20      # Patience
  accelerator: "auto"              # GPU/CPU 자동 선택
  precision: "32-true"             # 정밀도 (논문: 32-bit)
```

## 모델 아키텍처

### BERT4Rec 구조 (논문 Figure 2 기준)

```
Input Sequence: [item₁, item₂, [MASK], item₄, item₅]
    ↓
[Item Embedding] + [Position Embedding]
    ↓
    Dropout + LayerNorm
    ↓
┌─────────────────────────────────┐
│  Transformer Block 1            │
│  ┌──────────────────────────┐   │
│  │ Multi-Head Attention     │   │
│  │ (Bidirectional)          │   │
│  └──────────────────────────┘   │
│          ↓                      │
│    Residual + LayerNorm         │
│          ↓                      │
│  ┌──────────────────────────┐   │
│  │ Feed-Forward Network     │   │
│  │ (4x expansion, GELU)     │   │
│  └──────────────────────────┘   │
│          ↓                      │
│    Residual + LayerNorm         │
└─────────────────────────────────┘
    ↓
    ... (repeat N times)
    ↓
Output Projection (shared with item embedding)
    ↓
Predictions: [logits for all items]
```

### Transformer Block 상세

**Multi-Head Attention**:
```
Q, K, V = Linear(x)
Q, K, V = split_heads(Q, K, V)  # [batch, num_heads, seq_len, head_dim]
attn = softmax(Q·K^T / √head_dim)  # Scaled Dot-Product
output = attn · V
output = concat_heads(output)
output = Linear(output)
```

**Feed-Forward Network**:
```
FFN(x) = W₂(GELU(W₁(x)))
where W₁: hidden → 4*hidden
      W₂: 4*hidden → hidden
```

### Special Tokens

- `0`: Padding token
- `1 ~ num_items`: Item indices
- `num_items + 1`: [MASK] token

### Masking Strategy (논문 Section 3.2)

학습 시 각 아이템은 **15% 확률**로 마스킹:
- **80%**: `[MASK]` 토큰으로 대체
- **10%**: 랜덤 아이템으로 대체
- **10%**: 원본 유지

추론 시:
- 마지막 위치에 `[MASK]` 추가
- 모델이 다음 아이템 예측

## 평가 메트릭

### HIT@K (Hit Ratio)
```
HIT@K = 1 if ground_truth in top_K else 0
```

### NDCG@K (Normalized Discounted Cumulative Gain)
```
NDCG@K = 1 / log₂(rank + 2)  if rank < K else 0
```

### MRR (Mean Reciprocal Rank) - 미구현
```
MRR = 1 / rank
```

## PyTorch Lightning 기능

### 자동 제공되는 기능

1. **분산 학습**: Multi-GPU, TPU 자동 지원
2. **체크포인트 관리**:
   - `last.ckpt`: 마지막 epoch
   - `best.ckpt`: 최고 성능 모델
3. **Early Stopping**: Validation 성능 개선 없으면 중단
4. **로깅**: TensorBoard 자동 로깅
5. **재현성**: Seed 설정으로 재현 가능

### 체크포인트 로드

```python
from src.models.bert4rec import BERT4Rec

# 체크포인트에서 모델 로드
model = BERT4Rec.load_from_checkpoint("path/to/checkpoint.ckpt")
model.eval()

# 추론
predictions = model.predict(
    user_sequences=[[1, 2, 3, 4], [5, 6, 7]],
    topk=10,
    exclude_items=[set([1,2,3,4]), set([5,6,7])]
)
```

## 논문 구현 사항

### ✅ 논문과 동일하게 구현된 부분

| 구성 요소 | 논문 명세 | 구현 상태 |
|-----------|-----------|-----------|
| **Architecture** | Bidirectional Transformer | ✅ 구현 |
| **Attention** | Multi-Head Self-Attention | ✅ 구현 |
| **Scaling** | Q·K^T / √d_k (d_k = head_dim) | ✅ 구현 |
| **FFN** | 4x hidden dimension expansion | ✅ 구현 |
| **Activation** | GELU | ✅ 구현 |
| **Normalization** | Layer Normalization | ✅ 구현 |
| **Residual** | Residual Connection | ✅ 구현 |
| **Position Encoding** | Learnable positional embeddings | ✅ 구현 |
| **Masking Strategy** | 15% masking (80% [MASK], 10% random, 10% keep) | ✅ 구현 |
| **Loss** | Cross-Entropy on masked positions | ✅ 구현 |
| **Padding** | Zero padding with padding_idx=0 | ✅ 구현 |

### 📝 논문과 다르거나 선택 가능한 부분

#### 1. Output Layer Weight Sharing

**논문**:
> "we tie the item embedding matrix with the final output projection matrix to reduce parameters"

**구현**:
```python
share_embeddings: bool = True  # 기본값: True (논문과 동일)
```

- ✅ **True** (기본값): 논문과 동일, item embedding 재사용
- ❌ **False**: 별도의 Linear layer 사용 (더 많은 파라미터)

**설정 위치**: `configs/bert4rec.yaml`
```yaml
model:
  share_embeddings: true  # 권장 (논문과 동일)
```

#### 2. Validation 전략

**논문**:
> "we randomly select 100 negative items and rank these 101 items"

**구현**:
- **전체 아이템 ranking** 방식 구현 (더 정확하지만 느림)
- 샘플링 기반 평가는 미구현

**선택 이유**:
- 정확한 메트릭 측정을 위해 전체 ranking 선택
- 필요시 샘플링 방식 추가 가능

#### 3. Optimizer

**논문**:
> "We use Adam optimizer with learning rate of 0.001"

**구현**:
```python
optimizer = torch.optim.Adam(
    self.parameters(),
    lr=self.lr,  # 기본값: 0.0015
    weight_decay=self.weight_decay  # 기본값: 0.0
)
```

**차이점**:
- 논문 기본값: `lr=0.001`
- 구현 기본값: `lr=0.0015` (실험적으로 조정 가능)

#### 4. Learning Rate Scheduler

**논문**:
> No explicit mention of learning rate scheduling

**구현**:
- 현재 미구현 (constant learning rate)
- Lightning의 `configure_optimizers()`에서 쉽게 추가 가능

**추가 방법**:
```python
def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    return [optimizer], [scheduler]
```

#### 5. Dropout 위치

**논문**:
> "we apply dropout on all intermediate layers including the embedding layer"

**구현**:
- ✅ Embedding layer 이후
- ✅ Attention distribution
- ✅ Feed-forward network
- ✅ Residual connection 이후

**모두 논문과 동일**

#### 6. Attention Mask

**논문**:
> "bidirectional model... each position can attend to all positions"

**구현**:
```python
# Padding만 마스킹 (bidirectional attention 유지)
mask = (log_seqs > 0).unsqueeze(1).unsqueeze(2)
mask = mask.expand(-1, -1, seq_len, -1)  # [batch, 1, seq_len, seq_len]
```

**논문과 동일**: 패딩 위치만 마스킹, 모든 유효한 위치는 서로 볼 수 있음

#### 7. 평가 메트릭

**논문**:
> "HIT@K, NDCG@K, MRR"

**구현**:
- ✅ HIT@10
- ✅ NDCG@10
- ❌ MRR (미구현)

**MRR 추가 가능**:
```python
# validation_step에 추가
mrr = 1.0 / (rank + 1)  # rank는 0-based
self.log('val_mrr', mrr, ...)
```

## 성능 최적화 팁

### 1. GPU 메모리 부족 시

```yaml
# 배치 크기 줄이기
data:
  batch_size: 64

# Mixed precision 사용
training:
  precision: "16-mixed"
```

### 2. 학습 속도 향상

```yaml
# DataLoader workers 증가
data:
  num_workers: 8

# 검증 빈도 줄이기 (2 epoch마다)
training:
  val_check_interval: 2.0
```

### 3. 데이터셋별 권장 설정

**작은 데이터셋** (< 10K users):
```yaml
model:
  hidden_units: 32
  num_heads: 2
  num_layers: 2
  max_len: 30
```

**중간 데이터셋** (10K ~ 100K users):
```yaml
model:
  hidden_units: 64
  num_heads: 4
  num_layers: 3
  max_len: 50
```

**큰 데이터셋** (> 100K users):
```yaml
model:
  hidden_units: 128
  num_heads: 8
  num_layers: 4
  max_len: 200
```

## 논문 재현을 위한 설정

### MovieLens-1M (논문 Table 2)

```yaml
model:
  hidden_units: 256
  num_heads: 4
  num_layers: 2
  max_len: 200
  dropout_rate: 0.2
  mask_prob: 0.15

training:
  lr: 0.001
  num_epochs: 200
```

### Steam (논문 Table 2)

```yaml
model:
  hidden_units: 256
  num_heads: 4
  num_layers: 2
  max_len: 50
  dropout_rate: 0.5
  mask_prob: 0.15

training:
  lr: 0.001
  num_epochs: 200
```

## 문제 해결

### Import Error

```bash
# PYTHONPATH 설정
export PYTHONPATH=/data/ephemeral/home/juik/lightning:$PYTHONPATH
```

### CUDA Out of Memory

```yaml
# 배치 크기 줄이기
data:
  batch_size: 64

# 시퀀스 길이 줄이기
model:
  max_len: 30

# Mixed precision 사용
training:
  precision: "16-mixed"
```

### 학습이 느린 경우

```yaml
# 1. Workers 증가
data:
  num_workers: 8

# 2. 검증 빈도 줄이기
training:
  val_check_interval: 2.0

# 3. Pin memory 활성화 (DataLoader에서 자동)
```

### Validation 성능이 낮은 경우

1. **Masking 확률 조정**:
   ```yaml
   model:
     mask_prob: 0.2  # 0.15에서 증가
   ```

2. **Dropout 줄이기**:
   ```yaml
   model:
     dropout_rate: 0.2  # 0.3에서 감소
   ```

3. **모델 크기 증가**:
   ```yaml
   model:
     hidden_units: 128
     num_layers: 4
   ```

## 구현 세부 사항

### Weight Initialization (논문 명시 안됨)

```python
# Normal distribution (mean=0, std=0.02)
nn.init.normal_(module.weight, mean=0.0, std=0.02)

# Padding embedding은 0으로 초기화
if module.padding_idx is not None:
    nn.init.zeros_(module.weight[module.padding_idx])
```

### Loss Computation

```python
# Cross-entropy on masked positions only
criterion = nn.CrossEntropyLoss(ignore_index=0)

# ignore_index=0: 패딩 및 비마스킹 위치 무시
```

### Attention Mask Shape

```python
# Input: [batch, seq_len]
# Mask: [batch, 1, seq_len, seq_len]
#
# mask[b, 0, i, j] = 1 if position j is valid (not padding)
#                  = 0 if position j is padding
```

## 참고 자료

- [BERT4Rec 논문](https://arxiv.org/abs/1904.06690)
- [BERT 원본 논문](https://arxiv.org/abs/1810.04805)
- [PyTorch Lightning 문서](https://lightning.ai/docs/pytorch/stable/)
- [Hydra 문서](https://hydra.cc/)

## 라이선스

MIT License
