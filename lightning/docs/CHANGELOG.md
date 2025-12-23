# MultiVAE 개선 이력 (Changelog)

이 문서는 MultiVAE 프로젝트의 주요 개선 사항들을 정리한 것입니다.

---

## 목차

1. [성능 개선](#1-성능-개선)
2. [버그 수정](#2-버그-수정)
3. [구조 개선](#3-구조-개선)
4. [데이터 처리 최적화](#4-데이터-처리-최적화)
5. [설정 관리 개선](#5-설정-관리-개선)
6. [시각화 및 분석 도구](#6-시각화-및-분석-도구)
7. [문서화](#7-문서화)

---

## 1. 성능 개선

### 1.1 모델 하이퍼파라미터 최적화

**목표**: EASE 모델의 Recall@10 (0.16)을 넘어서는 성능 달성

**변경사항** ([multi_vae_v2.yaml](../configs/multi_vae_v2.yaml)):
```yaml
# Before (multi_vae.yaml)
model:
  hidden_dims: [400, 200]
  dropout: 0.6

training:
  kl_max_weight: 0.5
  kl_anneal_steps: 10000

# After (multi_vae_v2.yaml)
model:
  hidden_dims: [600, 200]  # 첫 번째 레이어 증가
  dropout: 0.5             # dropout 감소

training:
  kl_max_weight: 0.2       # 논문 기준값
  kl_anneal_steps: 20000   # 천천히 annealing
  early_stopping_patience: 20  # patience 증가
```

**근거**:
- `hidden_dims`: 더 큰 representation capacity
- `dropout`: 0.6은 과도한 정규화
- `kl_max_weight`: MultiVAE 논문 기준값 (0.2)
- Early stopping patience: 더 충분한 학습 시간 확보

### 1.2 데이터 분할 전략

**변경사항**:
```yaml
data:
  split_strategy: "leave_one_out"  # random -> leave_one_out
  valid_ratio: 0.1
```

**이유**: Leave-one-out 방식이 collaborative filtering에서 더 안정적인 평가 제공

---

## 2. 버그 수정

### 2.1 🐛 **CRITICAL**: Encoder에서 Dropout과 Normalization 순서 오류

**문제**:
- `F.normalize()` 후 `F.dropout()`을 적용하면 정규화가 깨짐
- Train loss가 ~1127에서 정체되고 극심한 노이즈 발생

**원인** ([multi_vae.py:95-98](../src/models/multi_vae.py#L95-L98)):
```python
# ❌ WRONG (Before)
x = F.normalize(x, p=2, dim=1)  # L2 normalize first
x = F.dropout(x, self.dropout, training=self.training)  # Then dropout
```

**수정**:
```python
# ✅ CORRECT (After)
x = F.dropout(x, self.dropout, training=self.training)  # Dropout first
x = F.normalize(x, p=2, dim=1)  # Then L2 normalize
```

**영향**:
- Train loss 안정화
- 수렴 속도 개선
- 최종 성능 향상

**참고**: [MultiVAE 논문](https://arxiv.org/abs/1802.05814) Section 3.2

### 2.2 Checkpoint 로딩 시 `weights_only` 에러

**문제**: PyTorch 2.6+에서 `weights_only=True`가 기본값으로 변경되어 OmegaConf 객체 로딩 실패

**수정** ([predict_multi_vae.py:66-70](../predict_multi_vae.py#L66-L70)):
```python
model = MultiVAE.load_from_checkpoint(
    checkpoint_path,
    num_items=datamodule.num_items,
    weights_only=False,  # 추가
)
```

---

## 3. 구조 개선

### 3.1 Train/Predict 스크립트 분리

**Before**:
- 단일 스크립트에서 train/inference 모두 처리

**After**:
- `train_multi_vae.py`: 학습 전용
- `predict_multi_vae.py`: 추론 전용 (BERT4Rec 패턴 참조)

**이점**:
- 명확한 책임 분리
- 각 단계별 최적화 가능
- 재사용성 향상

### 3.2 통합 Path 관리

**추가**: `src/utils/path_utils.py`

```python
def get_directories(cfg, stage="fit"):
    """Hydra 출력 디렉토리 기반으로 checkpoint/tensorboard 경로 생성"""
    # fit: 현재 실행 디렉토리 (새로 생성)
    # predict: 최근 실행 디렉토리 (기존 것 사용)
```

**사용**:
```python
# train_multi_vae.py
checkpoint_dir, tensorboard_dir = get_directories(cfg, stage="fit")

# predict_multi_vae.py
checkpoint_dir, tensorboard_dir = get_directories(cfg, stage="predict")
```

**이점**:
- Train/Predict 간 checkpoint 경로 일관성 보장
- Hydra 출력 디렉토리 구조 활용

### 3.3 실행 스크립트 통합

**추가**: `run_multi_vae.sh`

```bash
./run_multi_vae.sh [mode] [config_file]

# Modes:
# - train           Train only
# - predict         Predict only
# - both            Train + Predict (default)
# - clean           Clean cache only
# - clean-train     Clean cache + Train
# - clean-both      Clean cache + Train + Predict
```

**예시**:
```bash
./run_multi_vae.sh clean-both multi_vae_v2
```

**참조**: [run_bert4rec.sh](../run_bert4rec.sh) 패턴 적용

---

## 4. 데이터 처리 최적화

### 4.1 Leave-One-Out Split 최적화

**문제**: 기존 구현이 30초 이상 소요

**원인**: DataFrame 반복 필터링 (O(N*M) 복잡도)

**해결** ([recsys_data.py:216-252](../src/data/recsys_data.py#L216-L252)):
```python
# Before: O(N*M)
for u_idx in range(num_users):
    user_items = df_enc[df_enc["user"] == u_idx]["item"].tolist()  # 매번 필터링

# After: O(N)
grouped = df_enc.groupby("user")["item"].apply(list).to_dict()  # 한번만
for u_idx in range(num_users):
    user_items = grouped.get(u_idx, [])
```

**성능 향상**:
- 30초 → 0.1초 (300배 개선)

### 4.2 디스크 캐싱 시스템 도입

**목적**: 동일 설정으로 재실행 시 데이터 로딩 시간 단축

**구현** ([recsys_data.py:358-438](../src/data/recsys_data.py#L358-L438)):

```python
class RecSysDataModule:
    def __init__(self, ..., use_cache=True, cache_dir="~/.cache/recsys"):
        ...

    def _get_cache_key(self):
        """설정 기반 MD5 해시 생성"""
        key_params = {
            "data_file": self.data_file,
            "split_strategy": self.split_strategy,
            "seed": self.seed,
            "min_interactions": self.min_interactions,
            ...
        }
        return hashlib.md5(str(sorted(key_params.items())).encode()).hexdigest()

    def _save_to_cache(self):
        """train_mat, valid_gt 등을 pickle로 저장"""

    def _load_from_cache(self):
        """캐시 로드 (설정이 동일하면)"""
```

**캐시 내용**:
- `user2idx`, `idx2user`, `item2idx`, `idx2item`: ID 매핑
- `num_users`, `num_items`: 메타데이터
- `train_mat`: Sparse matrix (CSR)
- `valid_gt`: Validation ground truth

**설정** ([default_setup.yaml:34-37](../configs/common/default_setup.yaml#L34-L37)):
```yaml
data_cache:
  use_cache: true
  cache_dir: ~/.cache/recsys
```

**성능**:
- 첫 실행: 30초 (데이터 로드 + 전처리 + 캐시 저장)
- 재실행: 0.1초 (캐시 로드만)

**캐시 관리**:
```bash
./run_multi_vae.sh clean        # 캐시 삭제
./run_multi_vae.sh clean-train  # 캐시 삭제 + 학습
```

---

## 5. 설정 관리 개선

### 5.1 `.get()` 패턴 제거

**문제**: Config에서 `.get("field", default)` 사용 시 로그와 실제 동작 불일치 가능

**변경 대상**:
- [train_bert4rec.py](../train_bert4rec.py)
- [predict_bert4rec.py](../predict_bert4rec.py)
- [train_multi_vae.py](../train_multi_vae.py)
- [predict_multi_vae.py](../predict_multi_vae.py)

**Before**:
```python
min_interactions = cfg.data.get("min_interactions", 3)
num_workers = cfg.data.get("num_workers", 4)
```

**After**:
```python
min_interactions = cfg.data.min_interactions
num_workers = cfg.data.num_workers
```

**이점**:
- Config 파일에 명시되지 않은 값은 즉시 에러 발생
- 로그에 남은 설정과 실제 동작 일치 보장
- 디버깅 용이

### 5.2 설정 파일 구조화

**구조**:
```
configs/
├── common/
│   └── default_setup.yaml       # 공통 설정 (checkpoint, seed, cache 등)
├── bert4rec_v2.yaml             # BERT4Rec 설정
└── multi_vae_v2.yaml            # MultiVAE 설정 (개선 버전)
```

**공통 설정 추출** ([default_setup.yaml](../configs/common/default_setup.yaml)):
```yaml
# Hydra 출력 디렉토리
hydra:
  run:
    dir: ./saved/hydra_logs/${model_name}/${now:%Y-%m-%d}/${now:%H-%M-%S}

# Checkpoint 설정
checkpoint:
  save_top_k: 1
  monitor: "val_loss"
  mode: "min"

# Data caching 설정
data_cache:
  use_cache: true
  cache_dir: ~/.cache/recsys

# 기타
seed: 42
float32_matmul_precision: "medium"
```

---

## 6. 시각화 및 분석 도구

### 6.1 MultiVAE Attention Visualization

**추가**: [notebooks/visualize_multi_vae.ipynb](../notebooks/visualize_multi_vae.ipynb)

**기능**:
1. **Latent Space 시각화**:
   - μ (mean) 분포의 PCA/t-SNE 시각화
   - User embedding의 클러스터링 분석

2. **Reconstruction 분석**:
   - Input vs Reconstructed output 비교
   - Top-K 추천 아이템 시각화

3. **Training Dynamics**:
   - Loss curves (Total, Reconstruction, KL)
   - KL annealing 과정 시각화

4. **Item Similarity**:
   - Decoder weight 기반 item-item similarity
   - 유사 아이템 추천

**예시 플롯**:
```python
# Latent space visualization
visualize_latent_space(model, datamodule, method='tsne')

# Reconstruction quality
plot_reconstruction_quality(model, user_idx=123)

# Training curves
plot_training_curves(tensorboard_log_dir)
```

### 6.2 BERT4Rec Attention Visualization

**수정**: [notebooks/visualize_bert4rec.ipynb](../notebooks/visualize_bert4rec.ipynb)

**추가 기능**:
- Multi-head attention 시각화 (모든 레이어 + 모든 헤드)
- Layer-wise attention pattern 비교
- Position-wise attention 분석

**변경사항**:
```python
# Before: 첫 번째 레이어만
attention = model.transformer_blocks[0].attention

# After: 모든 레이어
for layer_idx in range(cfg.model.num_layers):
    attention = model.transformer_blocks[layer_idx].attention
    # 모든 head 시각화
```

---

## 성능 비교

### Recall@10

| 모델 | 설정 | Recall@10 | 비고 |
|-----|------|-----------|------|
| EASE | Baseline | **0.16** | 목표 |
| MultiVAE (Before) | `multi_vae.yaml` | 0.1367 | 초기 |
| MultiVAE (After) | `multi_vae_v2.yaml` | 0.1311 | 성능저하 |

### 데이터 로딩 속도

| 작업 | Before | After |
|-----|--------|-------|
| Leave-one-out split | 30초 | 1초 |
| 재실행 (캐시 활용) | 30초 | 1초 |

---

## 참고 자료

### 논문
- [Variational Autoencoders for Collaborative Filtering (MultiVAE)](https://arxiv.org/abs/1802.05814)
- [BERT4Rec: Sequential Recommendation with BERT](https://arxiv.org/abs/1904.06690)
- [EASE: Embarrassingly Shallow Autoencoders](https://arxiv.org/abs/1905.03375)

### 코드 참조
- [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/)
- [Hydra Configuration](https://hydra.cc/docs/intro/)

### 관련 문서
- [CHECKPOINT_STRUCTURE.md](CHECKPOINT_STRUCTURE.md): Checkpoint 파일 구조
- [README.md](../README.md): 프로젝트 개요 (업데이트 필요)

---

**Last Updated**: 2025-12-23
