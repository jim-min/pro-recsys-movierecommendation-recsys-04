# Optuna 실시간 모니터링 가이드

Optuna 튜닝 과정을 실시간으로 모니터링하는 방법을 설명합니다.

---

## 📊 모니터링 방법 비교

| 방법 | 실시간 | 시각화 | 설치 필요 | 난이도 | 추천도 |
|------|--------|--------|----------|--------|--------|
| **Optuna Dashboard** | ✅ | ✅ | ✅ | ⭐ | ⭐⭐⭐⭐⭐ |
| **Enhanced Script** | ✅ | ❌ | ❌ | ⭐⭐ | ⭐⭐⭐⭐ |
| **Progress Bar** | ✅ | ❌ | ❌ | ⭐ | ⭐⭐⭐ |
| **SQLite Query** | ⚠️ | ❌ | ❌ | ⭐⭐⭐ | ⭐⭐ |
| **TensorBoard** | ✅ | ✅ | ❌ | ⭐⭐ | ⭐⭐⭐ |

---

## 1. Optuna Dashboard (가장 추천! 🌟)

### 설치

```bash
pip install optuna-dashboard
```

### 사용 방법

#### Step 1: 튜닝 시작

```bash
# 터미널 1
cd tune
python tune_bert4rec_optuna.py --n_trials 50
```

#### Step 2: 대시보드 실행 (동시에)

```bash
# 터미널 2 (새 터미널 열기)
cd tune
optuna-dashboard sqlite:///bert4rec_study.db
```

#### Step 3: 브라우저 접속

```
http://127.0.0.1:8080
```

### Dashboard 기능

**실시간으로 확인 가능:**

1. **Study List** - 모든 study 목록
2. **Optimization History** - Trial별 성능 그래프
3. **Parameter Importances** - 어떤 파라미터가 중요한지
4. **Parallel Coordinate Plot** - 파라미터 조합 시각화
5. **Slice Plot** - 개별 파라미터 효과
6. **Contour Plot** - 파라미터 간 상호작용
7. **Intermediate Values** - Epoch별 성능 변화
8. **Trial Table** - 모든 trial 상세 정보

### 스크린샷 예시

```
┌──────────────────────────────────────────────────────┐
│ Optuna Dashboard                                     │
├──────────────────────────────────────────────────────┤
│                                                      │
│ Study: bert4rec_study                               │
│ Best Value: 0.1024 (Trial #23)                     │
│                                                      │
│ ┌────────────────────────────────────┐             │
│ │ Optimization History                │             │
│ │                                      │             │
│ │   Score                              │             │
│ │   0.11 ┤                        ●   │             │
│ │   0.10 ┤              ●     ●      │             │
│ │   0.09 ┤      ●   ●               │             │
│ │   0.08 ┤  ●                       │             │
│ │        └──────────────────────────┘│             │
│ │          Trial Number               │             │
│ └────────────────────────────────────┘             │
│                                                      │
│ Running Trials: 1                                   │
│ Completed Trials: 15                                │
│ Pruned Trials: 3                                    │
└──────────────────────────────────────────────────────┘
```

---

## 2. Enhanced Monitoring Script

상세한 로그와 실시간 피드백을 제공하는 스크립트입니다.

### 사용 방법

```bash
cd tune
python tune_bert4rec_optuna_monitored.py --n_trials 30
```

### 출력 예시

```
================================================================================
TRIAL 5 START (Total: 5)
================================================================================

Trial 5 Hyperparameters:
  Model: hidden=256, heads=4, layers=2, max_len=150
  Training: lr=0.000842, weight_decay=0.0234, batch=256
  Masking: random=0.18, last_item=0.12

2024-01-15 10:23:45 - Training started
Trial 5 | Epoch 0 | NDCG@10: 0.0823
Trial 5 | Epoch 1 | NDCG@10: 0.0891
Trial 5 | Epoch 2 | NDCG@10: 0.0942
Trial 5 | Epoch 3 | NDCG@10: 0.0978
Trial 5 | Epoch 4 | NDCG@10: 0.0995
Trial 5 | Epoch 5 | NDCG@10: 0.1012

🎉 NEW BEST SCORE: 0.1012

================================================================================
TRIAL 5 COMPLETE
  Score: 0.1012
  Duration: 15.3 minutes
  Current Best: 0.1012
================================================================================

✓ Trial 5 completed with score: 0.1012
  Current best: 0.1012 (Trial 5)
```

---

## 3. Progress Bar (기본 제공)

### 자동으로 표시되는 정보

```
[I 2024-01-15 10:00:00,000] Trial 3 finished with value: 0.0956
[I 2024-01-15 10:15:23,456] Trial 4 finished with value: 0.0989
[I 2024-01-15 10:30:45,123] Trial 5 finished with value: 0.1012 and parameters:
    {'hidden_units': 256, 'num_heads': 4, 'lr': 0.000842}
[I 2024-01-15 10:31:00,000] Trial 5 is the new best trial.

Progress: 10%|████▌                                    | 5/50 [2:30:15<22:45:45, 30.00s/trial]
```

---

## 4. SQLite 직접 쿼리

### 실시간 상태 확인

```bash
# Best trial 확인
sqlite3 bert4rec_study.db \
  "SELECT number, value FROM trials
   WHERE state = 'COMPLETE'
   ORDER BY value DESC LIMIT 5;"

# 출력:
# 23|0.1024
# 18|0.1019
# 31|0.1015
# 12|0.1008
# 7|0.0998
```

### 진행 상황 확인

```bash
# Trial 통계
sqlite3 bert4rec_study.db \
  "SELECT state, COUNT(*) as count
   FROM trials
   GROUP BY state;"

# 출력:
# COMPLETE|15
# RUNNING|1
# PRUNED|3
# FAIL|0
```

### 평균 소요 시간

```bash
sqlite3 bert4rec_study.db \
  "SELECT
     AVG((julianday(datetime_complete) - julianday(datetime_start)) * 24 * 60) as avg_minutes
   FROM trials
   WHERE state = 'COMPLETE';"

# 출력:
# 18.5
```

---

## 5. Python 스크립트로 모니터링

### 실시간 모니터링 스크립트 (`monitor_optuna.py`)

```python
#!/usr/bin/env python3
"""
Real-time Optuna monitoring script

Usage:
    python monitor_optuna.py bert4rec_study.db
"""

import sys
import time
import optuna
from datetime import datetime

def monitor_study(db_path, refresh_interval=10):
    """Monitor Optuna study in real-time"""

    study_name = db_path.replace('.db', '')
    storage = f'sqlite:///{db_path}'

    print(f"Monitoring study: {study_name}")
    print(f"Press Ctrl+C to stop\n")

    try:
        while True:
            # Load study
            study = optuna.load_study(
                study_name=study_name,
                storage=storage
            )

            # Clear screen
            print("\033[H\033[J", end='')

            # Header
            print("=" * 80)
            print(f"Optuna Study Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 80)

            # Statistics
            trials = study.trials
            completed = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
            running = [t for t in trials if t.state == optuna.trial.TrialState.RUNNING]
            pruned = [t for t in trials if t.state == optuna.trial.TrialState.PRUNED]

            print(f"\nTotal Trials: {len(trials)}")
            print(f"  Completed: {len(completed)}")
            print(f"  Running: {len(running)}")
            print(f"  Pruned: {len(pruned)}")

            # Best trial
            if completed:
                best = study.best_trial
                print(f"\n🏆 Best Trial: #{best.number}")
                print(f"   Score: {best.value:.4f}")
                print(f"   Params: {best.params}")

            # Top 5 trials
            if len(completed) >= 5:
                print("\n📊 Top 5 Trials:")
                top5 = sorted(completed, key=lambda t: t.value, reverse=True)[:5]
                for i, trial in enumerate(top5, 1):
                    print(f"   {i}. Trial {trial.number}: {trial.value:.4f}")

            # Recent trials
            print("\n📝 Recent Trials:")
            recent = sorted(trials, key=lambda t: t.number, reverse=True)[:5]
            for trial in recent:
                state = trial.state.name
                value = f"{trial.value:.4f}" if trial.value else "N/A"
                print(f"   Trial {trial.number}: {state:12s} | Score: {value}")

            print("\n" + "=" * 80)
            print(f"Refreshing every {refresh_interval}s... (Ctrl+C to stop)")

            time.sleep(refresh_interval)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
        sys.exit(0)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python monitor_optuna.py <study_db_file>")
        sys.exit(1)

    monitor_study(sys.argv[1])
```

**사용:**

```bash
# 터미널 1: 튜닝 실행
python tune_bert4rec_optuna.py --n_trials 50

# 터미널 2: 모니터링
python monitor_optuna.py bert4rec_study.db
```

---

## 6. TensorBoard 통합

Optuna는 TensorBoard와도 통합 가능합니다.

### 코드 수정

```python
# tune_bert4rec_optuna.py에 추가
from optuna.integration import TensorBoardCallback

# Study 생성 시 callback 추가
tensorboard_callback = TensorBoardCallback(
    "optuna_logs/",
    metric_name="val_ndcg@10"
)

study.optimize(
    objective,
    n_trials=n_trials,
    callbacks=[tensorboard_callback]
)
```

### TensorBoard 실행

```bash
# 터미널에서
tensorboard --logdir optuna_logs/
```

브라우저에서 `http://localhost:6006` 접속

---

## 🎯 추천 모니터링 조합

### 초보자
```bash
# 단순하게
python tune_bert4rec_optuna_monitored.py
```

### 중급자
```bash
# 터미널 1: 튜닝
python tune_bert4rec_optuna.py --n_trials 50

# 터미널 2: Dashboard
optuna-dashboard sqlite:///bert4rec_study.db
```

### 고급자
```bash
# 터미널 1: 튜닝
python tune_bert4rec_optuna.py --n_trials 50

# 터미널 2: Dashboard
optuna-dashboard sqlite:///bert4rec_study.db

# 터미널 3: Custom monitor
python monitor_optuna.py bert4rec_study.db

# 터미널 4: SQLite watch
watch -n 5 'sqlite3 bert4rec_study.db "SELECT number, value FROM trials ORDER BY value DESC LIMIT 5;"'
```

---

## 💡 모니터링 팁

### 1. 알람 설정

새로운 best trial 발견 시 알람:

```python
def slack_notify(study, trial):
    """Slack에 알림 전송"""
    if trial.value and trial.value > study.best_value:
        # Slack webhook으로 메시지 전송
        message = f"New best trial: {trial.number} with score {trial.value:.4f}"
        # ... slack API 호출 ...

study.optimize(
    objective,
    callbacks=[slack_notify]
)
```

### 2. 자동 체크포인트

일정 trial마다 중간 결과 저장:

```python
def checkpoint_callback(study, trial):
    """10 trial마다 결과 저장"""
    if trial.number % 10 == 0:
        # 현재까지 최고 설정 저장
        best_params = study.best_params
        # ... 파일로 저장 ...

study.optimize(
    objective,
    callbacks=[checkpoint_callback]
)
```

### 3. 진행 상황 이메일

```python
import smtplib
from email.mime.text import MIMEText

def email_progress(study, trial):
    """25%, 50%, 75% 완료 시 이메일"""
    progress = trial.number / total_trials
    if progress in [0.25, 0.5, 0.75]:
        msg = MIMEText(f"Tuning {progress*100}% complete. Best: {study.best_value:.4f}")
        # ... 이메일 전송 ...

study.optimize(
    objective,
    callbacks=[email_progress]
)
```

---

## 🔧 트러블슈팅

### Dashboard가 안 열릴 때

```bash
# 포트 변경
optuna-dashboard sqlite:///bert4rec_study.db --port 8888

# 외부 접속 허용
optuna-dashboard sqlite:///bert4rec_study.db --host 0.0.0.0
```

### Database가 잠긴 경우

```bash
# Study 로드 시 timeout 설정
storage = optuna.storages.RDBStorage(
    url="sqlite:///bert4rec_study.db",
    engine_kwargs={"connect_args": {"timeout": 30}}
)
```

### Real-time update 안될 때

```bash
# Dashboard auto-refresh 확인
# 브라우저에서 F5 또는 자동 새로고침 설정
```

---

## 📊 모니터링 체크리스트

- [ ] Optuna Dashboard 설치됨
- [ ] Dashboard 실행 중
- [ ] Progress bar 활성화
- [ ] 로그 파일 확인 가능
- [ ] Best trial 자동 저장 설정
- [ ] 알람/통지 설정 (선택)
- [ ] 백업 모니터링 방법 준비

---

## 🎉 빠른 시작

```bash
# 1. Dashboard 설치
pip install optuna-dashboard

# 2. 튜닝 시작 (터미널 1)
python tune_bert4rec_optuna.py --n_trials 30

# 3. Dashboard 실행 (터미널 2)
optuna-dashboard sqlite:///bert4rec_study.db

# 4. 브라우저 열기
open http://127.0.0.1:8080
```

**Happy Monitoring! 📊**
