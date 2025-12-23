# PyTorch Lightning Checkpoint 자료구조

PyTorch Lightning의 `.ckpt` 파일은 PyTorch의 pickle 포맷으로 저장되며, 모델 가중치뿐만 아니라 학습 상태를 완전히 재현하기 위한 모든 정보를 포함합니다.

## 파일 구조

```
checkpoint_file.ckpt (PyTorch pickle format, ~101MB)
│
├─ 📋 hyper_parameters          # 모델 하이퍼파라미터
│   ├─ num_items: 6807
│   ├─ hidden_dims: [600, 200]
│   ├─ dropout: 0.5
│   ├─ lr: 0.001
│   ├─ weight_decay: 0.0
│   ├─ kl_max_weight: 0.2
│   └─ kl_anneal_steps: 20000
│
├─ 🔧 state_dict               # 모델 가중치 (12개 텐서)
│   ├─ encoder.0.weight        [600, 6807]   # Input → Hidden1
│   ├─ encoder.0.bias          [600]
│   ├─ encoder.2.weight        [200, 600]    # Hidden1 → Hidden2
│   ├─ encoder.2.bias          [200]
│   ├─ mu.weight               [200, 200]    # Hidden2 → μ
│   ├─ mu.bias                 [200]
│   ├─ logvar.weight           [200, 200]    # Hidden2 → log(σ²)
│   ├─ logvar.bias             [200]
│   ├─ decoder.0.weight        [600, 200]    # Latent → Hidden1
│   ├─ decoder.0.bias          [600]
│   ├─ decoder.2.weight        [6807, 600]   # Hidden1 → Output
│   └─ decoder.2.bias          [6807]
│
├─ ⚙️  optimizer_states         # Optimizer 상태 (list[1])
│   └─ [0] Adam
│       ├─ state               # 각 파라미터별 momentum 등
│       │   └─ {12개 파라미터의 상태}
│       └─ param_groups        # Optimizer 설정
│           ├─ lr: 1.5625e-05  (현재 learning rate)
│           ├─ betas: (0.9, 0.999)
│           ├─ weight_decay: 0.0
│           └─ ...
│
├─ 🔄 lr_schedulers            # Learning Rate Scheduler (list[1])
│   └─ [0] scheduler 상태
│
├─ 📊 callbacks                # Callback 메타데이터
│   ├─ EarlyStopping
│   │   ├─ best_score: 1064.78
│   │   ├─ wait_count: 0
│   │   └─ patience: 20
│   └─ ModelCheckpoint
│       ├─ best_model_path: "...epoch=435-val_loss=1064.7800.ckpt"
│       ├─ best_model_score: 1064.78
│       ├─ last_model_path: ".../last.ckpt"
│       ├─ kth_best_model_path: "..."
│       └─ best_k_models: {path: score}
│
├─ 🔁 loops                    # Training loop 상태
│   └─ epoch/batch 진행 상태
│
└─ 🎯 메타데이터
    ├─ epoch: 435              # 저장 시점의 epoch
    ├─ global_step: 106820     # 전체 step 수
    └─ pytorch-lightning_version: "2.6.0"
```

## 주요 구성요소 설명

### 1. `hyper_parameters` (dict)

모델 초기화에 필요한 하이퍼파라미터들이 저장됩니다.

```python
{
    'num_items': 6807,
    'hidden_dims': [600, 200],
    'dropout': 0.5,
    'lr': 0.001,
    'weight_decay': 0.0,
    'kl_max_weight': 0.2,
    'kl_anneal_steps': 20000
}
```

**중요**:
- `load_from_checkpoint()` 호출 시 이 값들이 기본값으로 사용됩니다
- 파라미터로 전달하면 저장된 값을 오버라이드할 수 있습니다
- MultiVAE는 `num_items`가 모델 구조를 결정하므로 반드시 전달해야 합니다

### 2. `state_dict` (dict)

실제 학습된 모델 가중치(텐서)들이 저장됩니다.

**MultiVAE 구조**:
- **Encoder**: items(6807) → 600 → 200
  - `encoder.0.*`: 첫 번째 fully-connected layer
  - `encoder.2.*`: 두 번째 fully-connected layer

- **Latent Variables**: 200 → 200
  - `mu.*`: 평균(μ) 계산 레이어
  - `logvar.*`: 로그분산(log σ²) 계산 레이어

- **Decoder**: 200 → 600 → items(6807)
  - `decoder.0.*`: 첫 번째 fully-connected layer
  - `decoder.2.*`: 출력 레이어

총 **12개의 파라미터 텐서**로 구성됩니다.

### 3. `optimizer_states` (list)

Adam optimizer의 내부 상태를 저장합니다.

```python
[
    {
        'state': {
            # 각 파라미터별 momentum, variance 등
            0: {'step': ..., 'exp_avg': ..., 'exp_avg_sq': ...},
            1: {...},
            ...
        },
        'param_groups': [
            {
                'lr': 1.5625e-05,  # 현재 학습률
                'betas': (0.9, 0.999),
                'eps': 1e-08,
                'weight_decay': 0.0,
                ...
            }
        ]
    }
]
```

**용도**:
- 학습 재개(resume training) 시 필요
- Inference 시에는 불필요

### 4. `callbacks` (dict)

PyTorch Lightning callback들의 상태를 저장합니다.

#### EarlyStopping
```python
{
    'best_score': tensor(1064.7800),
    'wait_count': 0,
    'patience': 20,
    'stopped_epoch': 0
}
```

#### ModelCheckpoint
```python
{
    'best_model_path': '/path/to/checkpoints/multi-vae-epoch=435-val_loss=1064.7800.ckpt',
    'best_model_score': tensor(1064.7800),
    'last_model_path': '/path/to/checkpoints/last.ckpt',
    'best_k_models': {
        '/path/to/checkpoints/multi-vae-epoch=435-val_loss=1064.7800.ckpt': tensor(1064.7800)
    }
}
```

**주의사항**:
- `last_model_path`가 checkpoint에 저장되어 있어도 실제 파일이 없을 수 있습니다
- `save_last=False` 설정 시 `last.ckpt`가 생성되지 않습니다

### 5. 메타데이터

```python
{
    'epoch': 435,                          # 저장 시점 epoch
    'global_step': 106820,                 # 전체 training step
    'pytorch-lightning_version': '2.6.0',  # Lightning 버전
    'hparams_name': 'kwargs',
    'loops': {...}                         # Training loop 상태
}
```

## Checkpoint 로딩 방법

### 기본 로딩 (PyTorch 2.6+)

```python
import torch

# PyTorch 2.6부터는 weights_only=False 명시 필요
checkpoint = torch.load(
    'checkpoint.ckpt',
    map_location='cpu',
    weights_only=False  # OmegaConf 등 커스텀 객체 포함
)

# 주요 키 확인
print(checkpoint.keys())
# dict_keys(['epoch', 'global_step', 'pytorch-lightning_version',
#            'state_dict', 'loops', 'callbacks', 'optimizer_states',
#            'lr_schedulers', 'hparams_name', 'hyper_parameters'])
```

### Lightning 모델로 로딩

```python
from src.models.multi_vae import MultiVAE

# 방법 1: checkpoint의 hyper_parameters 사용
model = MultiVAE.load_from_checkpoint(
    'checkpoint.ckpt',
    num_items=6807,  # 필수! (모델 구조 결정)
    weights_only=False
)

# 방법 2: 모든 hyperparameter 명시
model = MultiVAE.load_from_checkpoint(
    'checkpoint.ckpt',
    num_items=6807,
    hidden_dims=[600, 200],
    dropout=0.5,
    lr=0.001,
    weights_only=False
)
```

### BERT4Rec vs MultiVAE 차이점

**BERT4Rec**:
```python
# num_items가 모델 구조에 영향을 주지 않음
model = BERT4Rec.load_from_checkpoint(checkpoint_path)
```

**MultiVAE**:
```python
# num_items가 첫 번째 레이어 크기를 결정
# 반드시 명시해야 함!
model = MultiVAE.load_from_checkpoint(
    checkpoint_path,
    num_items=datamodule.num_items,
    weights_only=False
)
```

**이유**:
- BERT4Rec: `num_items`는 최종 prediction head에서만 사용 (동적 처리 가능)
- MultiVAE: `num_items`가 encoder의 첫 번째 layer 크기 결정 (고정됨)

## 파일 크기

- **MultiVAE checkpoint**: ~101MB
  - state_dict가 대부분의 용량 차지
  - 가장 큰 텐서: `encoder.0.weight` [600, 6807]와 `decoder.2.weight` [6807, 600]

## 체크포인트 저장 설정

### train_multi_vae.py
```python
checkpoint_callback = ModelCheckpoint(
    dirpath=checkpoint_dir,
    filename="multi-vae-{epoch:02d}-{val_loss:.4f}",
    monitor="val_loss",
    mode="min",
    save_top_k=1,      # 최고 성능 모델 1개만 저장
    save_last=False,   # ⚠️ last.ckpt 저장 안 함
    verbose=True,
)
```

**주의**: `save_last=False`로 설정되어 있어 `last.ckpt`가 생성되지 않습니다.

## 문제 해결

### 1. "FileNotFoundError: last.ckpt"

**원인**: `save_last=False` 설정

**해결책**:
```python
# Option 1: save_last=True로 변경
checkpoint_callback = ModelCheckpoint(..., save_last=True)

# Option 2: best model 사용
checkpoint_path = get_latest_checkpoint(checkpoint_dir)  # best model 반환
```

### 2. "num_items mismatch"

**원인**: checkpoint의 `num_items`와 현재 데이터의 `num_items` 불일치

**해결책**:
```python
model = MultiVAE.load_from_checkpoint(
    checkpoint_path,
    num_items=datamodule.num_items,  # 현재 데이터 기준으로 오버라이드
    weights_only=False
)
```

### 3. "Weights only load failed"

**원인**: PyTorch 2.6+ 버전에서 `weights_only=True`가 기본값

**해결책**:
```python
# torch.load 사용 시
checkpoint = torch.load(path, weights_only=False)

# load_from_checkpoint 사용 시
model = MultiVAE.load_from_checkpoint(path, weights_only=False)
```

## 참고 자료

- [PyTorch Lightning Checkpointing](https://lightning.ai/docs/pytorch/stable/common/checkpointing.html)
- [torch.load Documentation](https://pytorch.org/docs/stable/generated/torch.load.html)
- [ModelCheckpoint API](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html)
