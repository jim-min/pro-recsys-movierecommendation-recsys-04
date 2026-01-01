# BERT4Rec Hyperparameter Tuning with Optuna

Optuna 기반의 BERT4Rec 하이퍼파라미터 자동 튜닝 도구입니다.

## 디렉토리 구조

```
tune/
├── README.md                          # 이 파일
├── quick_tune.py                      # 빠른 튜닝 스크립트 (권장)
├── tune_bert4rec_optuna.py            # 기본 튜닝 스크립트
├── tune_bert4rec_optuna_monitored.py  # 모니터링 강화 버전
├── bert4rec_*.db                      # Optuna study 데이터베이스
├── docs/                              # 문서
│   ├── README_optuna.md               # Optuna 튜닝 상세 가이드
│   └── MONITORING_GUIDE_optuna.md     # 모니터링 가이드
└── results/                           # 튜닝 결과
    ├── bert4rec_*_best_config.yaml    # 최적 하이퍼파라미터
    ├── bert4rec_*_history.html        # 최적화 히스토리
    ├── bert4rec_*_importance.html     # 파라미터 중요도
    └── bert4rec_*_parallel.html       # 병렬 좌표 플롯
```

## 빠른 시작

### 1. 기본 사용법 (권장)

```bash
cd tune

# Quick 모드 (10 trials, 20 epochs)
python quick_tune.py --mode quick

# Medium 모드 (30 trials, 50 epochs)
python quick_tune.py --mode medium

# Full 모드 (100 trials, 100 epochs)
python quick_tune.py --mode full
```

### 2. 직접 실행

```bash
cd tune

# 기본 튜닝
python tune_bert4rec_optuna.py --n_trials 50 --n_epochs 50

# 모니터링 강화 버전
python tune_bert4rec_optuna_monitored.py --n_trials 50
```

### 3. Study 재개

```bash
cd tune

python tune_bert4rec_optuna.py \
    --study_name bert4rec_study \
    --resume
```

## 병렬 실행 (Multi-GPU)

GPU가 여러 개 있는 경우:

```bash
cd tune

# GPU 2개 사용
python tune_bert4rec_optuna.py --n_trials 50 --n_jobs 2

# GPU 4개 사용
python quick_tune.py --mode medium --n_jobs 4
```

**주의**: `n_jobs` 값만큼 GPU가 필요합니다.

## 실시간 모니터링

터미널을 하나 더 열어서:

```bash
cd tune

# Optuna Dashboard 실행
optuna-dashboard sqlite:///bert4rec_medium.db
```

브라우저에서 http://127.0.0.1:8080 열기

## 결과 확인

### 1. 최적 하이퍼파라미터

```bash
cd tune

cat results/bert4rec_medium_best_config.yaml
```

### 2. 시각화 (HTML)

```bash
cd tune/results

# 브라우저로 열기
open bert4rec_medium_history.html      # 최적화 히스토리
open bert4rec_medium_importance.html   # 파라미터 중요도
open bert4rec_medium_parallel.html     # 병렬 좌표 플롯
```

### 3. SQLite 직접 조회

```bash
cd tune

# 상위 5개 trial 확인
sqlite3 bert4rec_medium.db \
  'SELECT number, value FROM trials ORDER BY value DESC LIMIT 5;'
```

## 주요 파라미터

### quick_tune.py

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| `--mode` | 튜닝 모드 (quick/medium/full) | quick |
| `--data_dir` | 데이터 디렉토리 | ~/data/train/ |

### tune_bert4rec_optuna.py

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| `--n_trials` | Trial 개수 | 50 |
| `--n_epochs` | Trial당 최대 epoch | 50 |
| `--study_name` | Study 이름 | bert4rec_study |
| `--n_jobs` | 병렬 실행 개수 | 1 |
| `--no_pruning` | Pruning 비활성화 | False |
| `--resume` | Study 재개 | False |

## 튜닝 대상 하이퍼파라미터

1. **모델 아키텍처**
   - `hidden_units`: [64, 128, 256]
   - `num_heads`: [2, 4, 8]
   - `num_layers`: [1, 2, 3]
   - `max_len`: [50, 100, 150, 200]
   - `dropout_rate`: [0.1 ~ 0.5]

2. **학습 파라미터**
   - `lr`: [1e-4 ~ 1e-2] (log scale)
   - `weight_decay`: [0.0 ~ 0.1]
   - `batch_size`: [128, 256, 512]

3. **마스킹 전략**
   - `random_mask_prob`: [0.1 ~ 0.3]
   - `last_item_mask_ratio`: [0.0 ~ 0.5]

## Trial 상태

- **Complete**: 정상 완료
- **Pruned**: 성능이 낮아 조기 종료 (정상)
- **Running**: 실행 중
- **Failed**: 에러 발생

## 더 알아보기

- **하이터파라미터 튜닝 가이드**: [docs/HYPERPARAMETER_TUNING_GUIDE.md](docs/HYPERPARAMETER_TUNING_GUIDE.md)
- **모니터링 가이드**: [docs/MONITORING_GUIDE_optuna.md](docs/MONITORING_GUIDE_optuna.md)

## 문제 해결

### Study가 없다는 에러

```bash
# resume=False로 변경하거나
python tune_bert4rec_optuna.py --study_name new_study

# 또는 기존 study 삭제 후 재시작
rm bert4rec_*.db
```

### GPU 메모리 부족

```bash
# batch_size 범위 축소 (tune_bert4rec_optuna.py 수정)
batch_size = trial.suggest_categorical('batch_size', [128, 256])  # 512 제거
```

### Pruning 너무 공격적

```bash
# Pruning 비활성화
python tune_bert4rec_optuna.py --no_pruning
```
