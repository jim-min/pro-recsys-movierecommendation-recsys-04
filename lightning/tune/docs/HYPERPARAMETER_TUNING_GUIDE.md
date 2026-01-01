# BERT4Rec Optuna 하이퍼파라미터 튜닝 가이드

Optuna를 사용하여 BERT4Rec 모델의 하이퍼파라미터를 자동으로 최적화하는 방법을 설명합니다.

## 📋 목차

- [설치](#설치)
- [빠른 시작](#빠른-시작)
- [사용 방법](#사용-방법)
- [튜닝 모드](#튜닝-모드)
- [고급 사용법](#고급-사용법)
- [결과 분석](#결과-분석)
- [FAQ](#faq)

---

## 🚀 설치

### 필요한 패키지

```bash
# Optuna 설치
pip install optuna

# 시각화를 위한 추가 패키지 (선택사항)
pip install plotly kaleido

# Optuna Dashboard (선택사항)
pip install optuna-dashboard
```

### 파일 확인

튜닝에 필요한 파일들:
```
lightning/
├── tune/                          # Optuna 튜닝 디렉토리
│   ├── quick_tune.py              # 빠른 실행 스크립트 (권장)
│   ├── tune_bert4rec_optuna.py   # 메인 튜닝 스크립트
│   ├── tune_bert4rec_optuna_monitored.py  # 모니터링 강화 버전
│   ├── docs/                      # 문서
│   │   ├── README_optuna.md       # 이 파일
│   │   └── MONITORING_GUIDE_optuna.md
│   └── results/                   # 튜닝 결과
│       └── bert4rec_*_best_config.yaml
├── src/
│   ├── models/bert4rec.py
│   └── data/bert4rec_data.py
└── configs/
    └── bert4rec_v2.yaml
```

---

## ⚡ 빠른 시작

### 0. 디렉토리 이동

```bash
cd tune
```

### 1. 테스트 실행 (2 trials, ~5분)

```bash
python quick_tune.py --mode test
```

### 2. 빠른 튜닝 (10 trials, ~2-3시간)

```bash
python quick_tune.py --mode quick
```

### 3. 추천 방법 (30 trials, ~8-12시간)

```bash
python quick_tune.py --mode medium
```

### 4. 최고 성능 (100 trials, ~1-2일)

```bash
python quick_tune.py --mode full
```

---

## 📖 사용 방법

### 기본 사용법

```bash
cd tune

# 기본 실행 (50 trials, 50 epochs per trial)
python tune_bert4rec_optuna.py

# 커스텀 설정
python tune_bert4rec_optuna.py \
    --n_trials 30 \
    --n_epochs 40 \
    --study_name my_bert4rec_tuning
```

### 주요 파라미터

| 파라미터 | 설명 | 기본값 | 예시 |
|---------|------|--------|------|
| `--n_trials` | 실행할 trial 수 | 50 | `--n_trials 100` |
| `--n_epochs` | Trial당 최대 epoch | 50 | `--n_epochs 30` |
| `--study_name` | Study 이름 | bert4rec_study | `--study_name my_study` |
| `--data_dir` | 데이터 디렉토리 | ~/data/train/ | `--data_dir /path/to/data` |
| `--n_jobs` | 병렬 실행 수 | 1 | `--n_jobs 2` |
| `--no_pruning` | Pruning 비활성화 | False | `--no_pruning` |
| `--resume` | 기존 study 재개 | False | `--resume` |

---

## 🎯 튜닝 모드

### Test Mode (스크립트 테스트)

```bash
cd tune
python quick_tune.py --mode test
```

**설정:**
- Trials: 2
- Epochs per trial: 2
- 예상 시간: 5분
- 목적: 스크립트 동작 확인

### Quick Mode (빠른 탐색)

```bash
cd tune
python quick_tune.py --mode quick
```

**설정:**
- Trials: 10
- Epochs per trial: 20
- 예상 시간: 2-3시간
- 예상 개선: +0.002~0.005 NDCG@10

**적합한 경우:**
- 빠른 proof-of-concept
- 초기 탐색
- 시간이 제한적일 때

### Medium Mode (균형잡힌 튜닝) ⭐ 추천

```bash
cd tune
python quick_tune.py --mode medium
```

**설정:**
- Trials: 30
- Epochs per trial: 50
- 예상 시간: 8-12시간
- 예상 개선: +0.005~0.010 NDCG@10

**적합한 경우:**
- 실전 사용
- 좋은 성능과 시간의 균형
- 대부분의 경우 추천

### Full Mode (최고 성능)

```bash
cd tune
python quick_tune.py --mode full
```

**설정:**
- Trials: 100
- Epochs per trial: 100
- 예상 시간: 1-2일
- 예상 개선: +0.010~0.015 NDCG@10

**적합한 경우:**
- 최종 제출
- 최고 성능이 필요한 경우
- 충분한 시간이 있을 때

---

## 🔧 고급 사용법

### 1. 병렬 실행 (GPU 2개 이상)

```bash
cd tune

# 2개의 GPU로 병렬 실행
python tune_bert4rec_optuna.py \
    --n_trials 50 \
    --n_jobs 2
```

⚠️ **주의:** `n_jobs` 값은 사용 가능한 GPU 수와 같거나 작아야 합니다.

### 2. Study 재개 (중단 후 재시작)

```bash
cd tune

# Study가 중단된 경우
python tune_bert4rec_optuna.py \
    --study_name bert4rec_medium \
    --resume \
    --n_trials 20  # 추가로 20 trials 실행
```

### 3. Pruning 비활성화

```bash
cd tune

# 모든 trial을 끝까지 실행
python tune_bert4rec_optuna.py \
    --no_pruning \
    --n_trials 30
```

**언제 사용:**
- Trial이 너무 일찍 종료되는 경우
- 느린 수렴이 예상되는 경우

### 4. 커스텀 데이터 경로

```bash
cd tune

python tune_bert4rec_optuna.py \
    --data_dir /custom/path/to/data \
    --n_trials 50
```

### 5. 실시간 모니터링

터미널을 하나 더 열어서:

```bash
cd tune

# Optuna Dashboard 실행
optuna-dashboard sqlite:///bert4rec_medium.db
```

브라우저에서 http://127.0.0.1:8080 열기

자세한 내용은 [MONITORING_GUIDE_optuna.md](MONITORING_GUIDE_optuna.md) 참고

---

## 📊 결과 분석

### 1. 최고 설정 확인

튜닝 완료 후 콘솔에 출력:

```
======================================================================
OPTIMIZATION COMPLETE
======================================================================

Best trial: 23
Best NDCG@10: 0.1024

Best hyperparameters:
  hidden_units: 256
  num_heads: 4
  num_layers: 2
  max_len: 150
  dropout_rate: 0.2
  lr: 0.0008
  weight_decay: 0.01
  batch_size: 256
  random_mask_prob: 0.15
  last_item_mask_ratio: 0.1

✅ Best config saved to: results/bert4rec_medium_best_config.yaml
```

### 2. 결과 파일

```
tune/
├── bert4rec_medium.db                     # SQLite 데이터베이스
└── results/
    ├── bert4rec_medium_best_config.yaml   # 최고 설정 (YAML)
    ├── bert4rec_medium_history.html       # 최적화 히스토리 그래프
    ├── bert4rec_medium_importance.html    # 파라미터 중요도 분석
    └── bert4rec_medium_parallel.html      # 병렬 좌표 플롯
```

### 3. 시각화 확인

```bash
cd tune/results

# 브라우저로 열기
firefox bert4rec_medium_history.html

# 또는
google-chrome bert4rec_medium_importance.html
```

**주요 시각화:**

- **History**: Trial별 성능 변화 추이
- **Importance**: 어떤 파라미터가 성능에 가장 영향을 미치는지 (2+ trials 필요)
- **Parallel Coordinate**: 파라미터 조합과 성능의 관계 (2+ trials 필요)

### 4. 최고 설정으로 학습

```bash
# 1. 설정 파일 확인
cat tune/results/bert4rec_medium_best_config.yaml

# 2. 수동으로 configs/bert4rec_v2.yaml 파일 수정
```

수동으로 `configs/bert4rec_v2.yaml` 파일을 수정:

```yaml
model:
  hidden_units: 256        # tune/results에서 복사
  num_heads: 4
  num_layers: 2
  max_len: 150
  dropout_rate: 0.2
  random_mask_prob: 0.15
  last_item_mask_ratio: 0.1

training:
  lr: 0.0008
  weight_decay: 0.01

data:
  batch_size: 256
```

그 후 정상 학습:

```bash
# 프로젝트 루트로 이동
cd ..

# 최적화된 설정으로 학습
python train_bert4rec.py
```

---

## 🎛️ 튜닝 가능한 하이퍼파라미터

### 모델 아키텍처

| 파라미터 | 탐색 범위 | 설명 |
|---------|----------|------|
| `hidden_units` | [64, 128, 256] | Hidden layer 차원 |
| `num_heads` | [2, 4, 8] | Attention head 수 |
| `num_layers` | [1, 2, 3] | Transformer layer 수 |
| `max_len` | [50, 100, 150, 200] | 최대 시퀀스 길이 |
| `dropout_rate` | 0.1 ~ 0.5 | Dropout 비율 |

### 학습 설정

| 파라미터 | 탐색 범위 | 설명 |
|---------|----------|------|
| `lr` | 1e-4 ~ 1e-2 (log) | Learning rate |
| `weight_decay` | 0.0 ~ 0.1 | Weight decay (L2 regularization) |
| `batch_size` | [128, 256, 512] | Batch size |

### 마스킹 전략

| 파라미터 | 탐색 범위 | 설명 |
|---------|----------|------|
| `random_mask_prob` | 0.1 ~ 0.3 | Random masking 확률 |
| `last_item_mask_ratio` | 0.0 ~ 0.5 | Last item masking 비율 |

---

## 💡 성능 최적화 팁

### 1. Trial 수 vs Epochs

**적은 Trial, 긴 Epochs:**
```bash
cd tune
python tune_bert4rec_optuna.py --n_trials 20 --n_epochs 100
```
- 장점: 각 설정을 깊게 탐색
- 단점: 탐색 공간이 좁음

**많은 Trial, 짧은 Epochs:**
```bash
cd tune
python tune_bert4rec_optuna.py --n_trials 100 --n_epochs 30
```
- 장점: 넓은 탐색 공간
- 단점: 각 설정이 충분히 학습되지 않을 수 있음

**추천 (균형):**
```bash
cd tune
python tune_bert4rec_optuna.py --n_trials 50 --n_epochs 50
```

### 2. Pruning 활용

Pruning은 성능이 낮은 trial을 일찍 종료하여 시간을 절약합니다.

```python
# 자동으로 활성화됨 (기본값)
# 비활성화하려면: --no_pruning
```

**Pruning이 작동하는 방식:**
- 첫 10 epochs: 모든 trial 실행
- 10 epochs 이후: Median 성능보다 낮으면 종료

**Pruned 상태:**
- "Pruned"는 에러가 아닌 정상적인 조기 종료
- 성능이 낮다고 판단되어 중단된 것
- 시간을 절약하기 위한 효율적인 동작

### 3. 메모리 관리

GPU 메모리가 부족한 경우:

```python
# tune_bert4rec_optuna.py 수정
# Line 71: batch_size 범위 조정
batch_size = trial.suggest_categorical('batch_size', [128, 256])  # 512 제거
```

---

## 🔍 Study 관리

### SQLite 직접 조회

```bash
cd tune

# 상위 5개 trial 확인
sqlite3 bert4rec_medium.db \
  'SELECT number, value FROM trials ORDER BY value DESC LIMIT 5;'

# Trial 상태 확인
sqlite3 bert4rec_medium.db \
  'SELECT state, COUNT(*) FROM trials GROUP BY state;'
```

### Study 삭제

```bash
cd tune

# 데이터베이스 파일 삭제
rm bert4rec_medium.db
rm -rf results/*
```

---

## 🐛 문제 해결

### 1. Out of Memory (OOM)

**증상:**
```
RuntimeError: CUDA out of memory
```

**해결:**
```python
# tune_bert4rec_optuna.py 수정
# Line 71: batch_size 범위 줄이기
batch_size = trial.suggest_categorical('batch_size', [64, 128])

# 또는 Line 62: hidden_units 줄이기
hidden_units = trial.suggest_categorical('hidden_units', [64, 128])
```

### 2. Trial이 너무 일찍 종료됨

**증상:**
```
Trial 5 pruned.
Trial 6 pruned.
...
```

**해결:**
```bash
cd tune

# Pruning 비활성화
python tune_bert4rec_optuna.py --no_pruning
```

또는 warmup 기간 늘리기:
```python
# tune_bert4rec_optuna.py Line 233-236 수정
pruner=optuna.pruners.MedianPruner(
    n_startup_trials=10,  # 5 → 10
    n_warmup_steps=20,    # 10 → 20
)
```

### 3. Study 재개 안됨

**증상:**
```
KeyError: 'Record does not exist.'
```

**해결:**
```bash
cd tune

# 올바른 study 이름 확인
ls -la *.db

# 재개 시 정확한 이름 사용
python tune_bert4rec_optuna.py \
    --study_name bert4rec_medium \
    --resume
```

### 4. 시각화 파일 생성 안됨

**증상:**
```
ValueError: Cannot evaluate parameter importances with only a single trial.
```

**원인:**
- Parameter importance와 parallel plot은 **2개 이상의 completed trial** 필요
- History plot은 1개 trial로도 생성됨

**해결:**
- Test 모드 대신 Quick 모드 이상 사용
- 또는 2개 이상의 trial 실행

---

## 📚 참고 자료

### Optuna 공식 문서
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [Optuna Tutorial](https://optuna.readthedocs.io/en/stable/tutorial/index.html)
- [PyTorch Lightning Integration](https://optuna.readthedocs.io/en/stable/reference/integration.html#pytorch-lightning)

### BERT4Rec 논문
- [BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer](https://arxiv.org/abs/1904.06690)

---

## ❓ FAQ

### Q1: 튜닝에 얼마나 걸리나요?

**A:** 모드에 따라 다릅니다:
- Test: 5분
- Quick: 2-3시간
- Medium: 8-12시간
- Full: 1-2일

GPU 성능과 데이터셋 크기에 따라 달라질 수 있습니다.

### Q2: 튜닝 중에 중단해도 되나요?

**A:** 네! `Ctrl+C`로 중단 후 `--resume` 옵션으로 재개할 수 있습니다.

```bash
cd tune
python tune_bert4rec_optuna.py --study_name bert4rec_medium --resume
```

### Q3: 여러 GPU에서 병렬로 실행할 수 있나요?

**A:** 네, `--n_jobs` 옵션을 사용하세요:

```bash
cd tune
python tune_bert4rec_optuna.py --n_jobs 2  # GPU 2개 사용
```

### Q4: logger 경고가 나타납니다

**증상:**
```
You called `self.log('val_hit@10', ..., logger=True)` but have no logger configured.
```

**A:** 이것은 정보성 경고로, 튜닝 동작에는 영향을 주지 않습니다. Optuna 튜닝 시에는 `logger=False`로 설정되어 있어서 나타나는 것이며 무시해도 됩니다.

### Q5: Pruned는 에러인가요?

**A:** 아닙니다! Pruned는 정상적인 동작입니다.
- 성능이 낮은 trial을 조기 종료하여 시간 절약
- 더 유망한 hyperparameter 조합 탐색에 집중
- 전체 튜닝 효율성을 높이는 기능

### Q6: Test 모드에서 시각화 파일이 일부만 생성됩니다

**A:** Parameter importance와 parallel plot은 2개 이상의 completed trial이 필요합니다. Test 모드(2 trials, 2 epochs)는 스크립트 테스트용이므로, 실제 튜닝은 Quick 모드 이상을 사용하세요.

---

## 🎉 빠른 참조

```bash
# 디렉토리 이동
cd tune

# 🧪 스크립트 테스트
python quick_tune.py --mode test

# 🚀 빠른 시작
python quick_tune.py --mode quick

# 📊 추천 설정
python quick_tune.py --mode medium

# 🔄 중단 후 재개
python tune_bert4rec_optuna.py --study_name bert4rec_medium --resume

# 💪 병렬 실행 (GPU 2개)
python tune_bert4rec_optuna.py --n_jobs 2

# 📈 결과 확인
firefox results/bert4rec_medium_history.html

# 🔍 실시간 모니터링
optuna-dashboard sqlite:///bert4rec_medium.db

# ✅ 최고 설정 확인
cat results/bert4rec_medium_best_config.yaml
```

---

**Happy Tuning! 🎯**
