# predict_bert4rec.py 주요 변수 형태 분석

이 문서는 `predict_bert4rec.py`에서 사용되는 주요 변수들의 형태와 데이터 구조를 상세히 설명합니다.

---

## ⚠️ 중요: 버그 수정 (2025-12-25)

### 발견된 문제 1: validation 아이템 제외 누락
초기 구현에서 **validation 아이템이 `batch_exclude`에 포함되지 않아**, 이미 본 validation 아이템을 다시 추천할 수 있는 버그가 있었습니다.

### 발견된 문제 2: 모델 입력 시퀀스 오류
**더 심각한 문제**: 모델 입력으로 train 시퀀스만 사용하면, **validation 아이템 이전 시점을 기준으로 예측**하게 됩니다.

#### 잘못된 예시
```python
# User의 실제 시퀀스: [1, 2, 3, ..., 99, 100]
# train: [1, 2, 3, ..., 99]
# valid: [100]

# ❌ 잘못된 방식 (train만 사용)
input_seq = [1, 2, 3, ..., 99, MASK]
# → "99 다음에 올 아이템은?" 예측
# → 정답은 100이지만, 100은 exclude되어서 추천 안됨
# → 실제로는 100 이후를 예측해야 하는데, 99 이후를 예측함!

# ✅ 올바른 방식 (train + valid 사용)
input_seq = [1, 2, 3, ..., 99, 100, MASK]
# → "100 다음에 올 아이템은?" 예측
# → 올바르게 미래 아이템 예측!
```

### 수정 내용
1. **bert4rec_data.py**에 `get_full_sequences()` 메서드 추가
   - train + validation 아이템을 모두 포함하는 시퀀스 반환
2. **predict_bert4rec.py** 수정
   - **모델 입력**과 **제외 리스트** 모두 `full_sequences` 사용
   - `user_sequences` 변수 제거 (불필요)

### Before (버그)
```python
user_sequences = datamodule.get_all_sequences()  # train only
batch_seqs.append(seq)  # ❌ train만 사용 → 과거 시점 예측
batch_exclude.append(set(seq))  # ❌ validation 제외 안됨
```

### After (수정)
```python
full_sequences = datamodule.get_full_sequences()  # train + valid
batch_seqs.append(full_seq)  # ✅ 전체 사용 → 최신 시점 예측
batch_exclude.append(set(full_seq))  # ✅ validation 포함
```

---

## 목차
1. [full_sequences](#1-full_sequences)
2. [user_indices](#2-user_indices)
3. [batch_seqs](#3-batch_seqs)
4. [batch_exclude](#4-batch_exclude)
5. [기타 주요 변수](#5-기타-주요-변수)
6. [전체 데이터 흐름](#6-전체-데이터-흐름)

---

## 1. full_sequences

### 정의 위치
[predict_bert4rec.py:95](../predict_bert4rec.py#L95)
```python
full_sequences = datamodule.get_full_sequences()  # train + valid (for both input and exclusion)
```

### 데이터 타입
```python
Dict[int, List[int]]
```

### 구조
```python
{
    user_idx: [item_idx_1, item_idx_2, ..., item_idx_n],  # train + valid
    ...
}
```

### 상세 설명
- **Key**: `user_idx` (int) - 인코딩된 유저 인덱스 (0부터 시작)
- **Value**: `List[item_idx]` - 해당 유저의 **전체** 시간순 상호작용 아이템 시퀀스
  - 아이템 인덱스는 1부터 시작 (0은 padding token)
  - **train + validation 아이템 모두 포함**
  - 시간순으로 정렬되어 있음
  - **모델 입력**과 **제외 리스트** 모두에 사용됨

### 예시
```python
full_sequences = {
    0: [15, 342, 88, 201, 456, 789, 999],     # 999 = validation 아이템
    1: [23, 45, 67, 89, 123, 888],             # 888 = validation 아이템
    2: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 777],  # 777 = validation 아이템
    ...
}
```

### 생성 과정
[bert4rec_data.py:359-371](../src/data/bert4rec_data.py#L359-L371)
```python
def get_full_sequences(self):
    """
    Get full sequences including validation items (for inference)
    """
    full_sequences = {}
    for user_idx in self.user_train.keys():
        # train + validation 아이템 모두 포함
        full_seq = self.user_train[user_idx] + [self.user_valid[user_idx]]
        full_sequences[user_idx] = full_seq
    return full_sequences
```

### 중요성
이 시퀀스를 모델 입력으로 사용해야 하는 이유:
```python
# ❌ train만 사용하면
input: [1, 2, 3, ..., 99, MASK]
예측: "99 다음은?" → 100 (하지만 100은 exclude됨)
문제: 과거 시점을 기준으로 예측

# ✅ full 사용하면
input: [1, 2, 3, ..., 99, 100, MASK]
예측: "100 다음은?" → 실제 미래 아이템
올바름: 최신 시점을 기준으로 예측
```

### 특징
- **시퀀스 길이**: 가변적 (유저마다 다름)
- **최소 길이**: `min_interactions` (train + valid 합친 것)
- **아이템 범위**: `1 ~ num_items`
- **마지막 아이템**: validation 아이템

---

## 2. user_indices

### 정의 위치
[predict_bert4rec.py:96](../predict_bert4rec.py#L96)
```python
user_indices = list(full_sequences.keys())
```

### 데이터 타입
```python
List[int]
```

### 구조
```python
[user_idx_1, user_idx_2, user_idx_3, ...]
```

### 상세 설명
- `full_sequences`의 모든 key를 리스트로 변환
- 배치 처리를 위한 순회 가능한 리스트
- 유저 인덱스는 0부터 시작하는 연속적인 정수

### 예시
```python
user_indices = [0, 1, 2, 3, 4, ..., 31359]
```

### 용도
```python
# 배치 단위로 순회하기 위한 인덱스 리스트
for start_idx in range(0, len(user_indices), batch_size):
    end_idx = min(start_idx + batch_size, len(user_indices))
    batch_users = user_indices[start_idx:end_idx]  # 현재 배치의 유저들
```

### 특징
- **길이**: `num_users` (필터링 후 유저 수)
- **순서**: dictionary의 key 순서 (Python 3.7+는 insertion order 보장)

---

## 3. batch_seqs

### 정의 위치
[predict_bert4rec.py:109-111](../predict_bert4rec.py#L109-L111)
```python
batch_seqs = []
for user_idx in batch_users:
    full_seq = full_sequences[user_idx]
    batch_seqs.append(full_seq)  # Use FULL sequence (train + valid)
```

### 데이터 타입
```python
List[List[int]]
```

### 구조
```python
[
    [item_idx_1, item_idx_2, ...],  # user 1의 시퀀스
    [item_idx_1, item_idx_2, ...],  # user 2의 시퀀스
    ...
]
```

### 상세 설명
- 현재 배치에 포함된 유저들의 **전체 시퀀스** 리스트 (train + valid)
- 각 시퀀스는 가변 길이
- 모델의 `predict()` 메서드에 입력으로 전달됨
- **중요**: validation 아이템까지 포함해야 최신 시점 기준 예측 가능

### 예시
```python
# batch_size = 3인 경우
batch_seqs = [
    [15, 342, 88, 201, 456, 789, 999],      # 길이 7 (마지막 999는 valid)
    [23, 45, 67, 89, 123, 888],              # 길이 6 (마지막 888은 valid)
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 777],   # 길이 11 (마지막 777은 valid)
]
```

### 모델 내부 처리
[bert4rec.py:445-453](../src/models/bert4rec.py#L445-L453)
```python
# 1. 마지막에 mask token 추가 및 max_len으로 자르기
seqs = []
for seq in user_sequences:  # user_sequences == batch_seqs
    masked_seq = (list(seq) + [self.mask_token])[-self.max_len:]

    # 2. Padding (앞쪽에 pad_token 추가)
    if len(masked_seq) < self.max_len:
        masked_seq = [self.pad_token] * (self.max_len - len(masked_seq)) + masked_seq

    seqs.append(masked_seq)

seqs = np.array(seqs, dtype=np.int64)  # shape: (batch_size, max_len)
```

### 변환 예시 (max_len=50, mask_token=6808)
```python
# 입력
batch_seqs = [
    [15, 342, 88, 201, 456, 789],  # 길이 6
]

# Step 1: mask token 추가 후 max_len으로 자르기
[15, 342, 88, 201, 456, 789, 6808]  # 길이 7

# Step 2: padding (앞쪽)
[0, 0, 0, ..., 0, 15, 342, 88, 201, 456, 789, 6808]  # 길이 50
#<---- 43개 --->  <-------- 원본 + mask -------->

# 최종 numpy array
shape: (batch_size, 50)
```

### 특징
- **길이**: `batch_size` (마지막 배치는 작을 수 있음)
- **각 시퀀스 길이**: 가변적
- **처리 방식**: 모델 내부에서 padding 및 truncation 수행

---

## 4. batch_exclude

### 정의 위치
[predict_bert4rec.py:105-113](../predict_bert4rec.py#L105-L113)
```python
batch_exclude = []
for user_idx in batch_users:
    seq = user_sequences[user_idx]  # train sequence for model input
    full_seq = full_sequences[user_idx]  # train + valid for exclusion
    batch_seqs.append(seq)
    batch_exclude.append(set(full_seq))  # Exclude ALL already interacted items (train + valid)
```

### 데이터 타입
```python
List[Set[int]]
```

### 구조
```python
[
    {item_idx_1, item_idx_2, ...},  # user 1이 이미 본 아이템들 (train + valid)
    {item_idx_1, item_idx_2, ...},  # user 2가 이미 본 아이템들 (train + valid)
    ...
]
```

### 상세 설명
- 각 유저가 이미 상호작용한 **모든** 아이템들의 집합 (train + validation)
- 추천 시 이미 본 아이템을 제외하기 위해 사용
- Set 자료구조를 사용하여 빠른 조회 (O(1))
- **중요**: validation 아이템도 포함되어야 함 (이미 본 아이템을 다시 추천하면 안됨)

### 예시
```python
# user_sequences (train only - 모델 입력용)
user_sequences = {
    0: [15, 342, 88, 201, 456, 789],          # validation 제외
    1: [23, 45, 67, 89, 123],                  # validation 제외
    2: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],       # validation 제외
}

# full_sequences (train + valid - 제외용)
full_sequences = {
    0: [15, 342, 88, 201, 456, 789, 999],     # 999 = validation 아이템
    1: [23, 45, 67, 89, 123, 888],             # 888 = validation 아이템
    2: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 777],  # 777 = validation 아이템
}

# batch_seqs (모델 입력)
batch_seqs = [
    [15, 342, 88, 201, 456, 789],          # validation 제외
    [23, 45, 67, 89, 123],
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
]

# batch_exclude (제외용 - train + valid 모두 포함!)
batch_exclude = [
    {15, 342, 88, 201, 456, 789, 999},     # validation(999) 포함!
    {23, 45, 67, 89, 123, 888},             # validation(888) 포함!
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 777},  # validation(777) 포함!
]
```

### 모델 내부 사용
[bert4rec.py:463-467](../src/models/bert4rec.py#L463-L467)
```python
# Exclude already seen items
if exclude_items is not None:  # exclude_items == batch_exclude
    for i, items in enumerate(exclude_items):
        if items:
            # 해당 아이템들의 score를 매우 낮게 설정 (-1e9)
            # 이렇게 하면 top-k 선택 시 제외됨
            scores[i, list(items)] = -1e9
```

### 처리 흐름
```python
# 1. 모델 출력 (scores)
scores.shape = (batch_size, num_items + 2)  # +2: pad_token, mask_token

# 2. Invalid token masking
scores[:, self.pad_token] = -1e9   # padding token (0) 제외
scores[:, self.mask_token] = -1e9  # mask token (6808) 제외

# 3. 이미 본 아이템 masking
for i, items in enumerate(batch_exclude):
    scores[i, list(items)] = -1e9  # 각 유저가 본 아이템 제외

# 4. Top-K 선택
_, top_items = torch.topk(scores, k=topk, dim=1)
# shape: (batch_size, topk)
```

### 예시 (실제 동작)
```python
# batch_size=2, num_items=6807, topk=10

# 초기 scores (batch_size=2, num_items+2=6809)
scores = [
    [s0, s1, s2, ..., s6808],  # user 0
    [s0, s1, s2, ..., s6808],  # user 1
]

# batch_exclude = [{15, 342, 88}, {23, 45}]

# Masking 후
scores[0, [0, 6808, 15, 342, 88]] = -1e9   # user 0
scores[1, [0, 6808, 23, 45]] = -1e9        # user 1

# Top-10 추출 → 이미 본 아이템과 invalid token은 제외됨
top_items = [
    [501, 234, 789, 456, 123, 678, 901, 345, 567, 890],  # user 0의 top-10
    [456, 789, 123, 456, 678, 234, 567, 890, 345, 678],  # user 1의 top-10
]
```

### 특징
- **길이**: `batch_size`
- **각 Set 크기**: 유저의 interaction 수만큼 (가변적)
- **목적**: 이미 본 아이템을 추천에서 제외

---

## 5. 기타 주요 변수

### 5.1 batch_users
```python
# predict_bert4rec.py:101
batch_users = user_indices[start_idx:end_idx]
```
- **타입**: `List[int]`
- **구조**: `[user_idx_1, user_idx_2, ...]`
- **예시**: `[0, 1, 2]` (batch_size=3일 때)
- **설명**: 현재 배치에 포함된 유저 인덱스 리스트

### 5.2 top_items (모델 출력)
```python
# bert4rec.py:470
_, top_items = torch.topk(scores, k=topk, dim=1)
return top_items.cpu().numpy()
```
- **타입**: `numpy.ndarray`
- **Shape**: `(batch_size, topk)`
- **예시**:
```python
array([
    [501, 234, 789, 456, 123, 678, 901, 345, 567, 890],  # user 0의 top-10
    [456, 789, 123, 456, 678, 234, 567, 890, 345, 678],  # user 1의 top-10
    [123, 456, 789, 234, 567, 890, 345, 678, 901, 234],  # user 2의 top-10
])
```
- **설명**: 각 유저별 추천된 top-k 아이템 인덱스

### 5.3 results
```python
# predict_bert4rec.py:94, 131
results = []
results.append({"user": original_user_id, "item": original_item_id})
```
- **타입**: `List[Dict[str, int]]`
- **구조**:
```python
[
    {"user": 11, "item": 4643},
    {"user": 11, "item": 170},
    {"user": 11, "item": 531},
    ...
]
```
- **설명**: 최종 추천 결과 (원본 ID로 변환된 상태)

### 5.4 datamodule 매핑
```python
# bert4rec_data.py:248-253
self.item2idx = pd.Series(data=np.arange(len(item_ids)) + 1, index=item_ids)
self.user2idx = pd.Series(data=np.arange(len(user_ids)), index=user_ids)
self.idx2item = pd.Series(index=self.item2idx.values, data=self.item2idx.index)
self.idx2user = pd.Series(index=self.user2idx.values, data=self.user2idx.index)
```
- **item2idx**: `original_item_id → item_idx` (1부터 시작)
- **idx2item**: `item_idx → original_item_id`
- **user2idx**: `original_user_id → user_idx` (0부터 시작)
- **idx2user**: `user_idx → original_user_id`

---

## 6. 전체 데이터 흐름

```python
# ============================================================
# Step 1: 전체 시퀀스 가져오기
# ============================================================
user_sequences = datamodule.get_all_sequences()  # train only (모델 입력용)
# {
#     0: [15, 342, 88, 201, 456, 789],          # validation 제외
#     1: [23, 45, 67, 89, 123],                  # validation 제외
#     2: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],       # validation 제외
#     ...
# }

full_sequences = datamodule.get_full_sequences()  # train + valid (제외용)
# {
#     0: [15, 342, 88, 201, 456, 789, 999],     # validation(999) 포함
#     1: [23, 45, 67, 89, 123, 888],             # validation(888) 포함
#     2: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 777],  # validation(777) 포함
#     ...
# }

user_indices = list(user_sequences.keys())
# [0, 1, 2, 3, ..., 31359]

# ============================================================
# Step 2: 배치 단위로 처리
# ============================================================
for start_idx in range(0, len(user_indices), batch_size):
    # 현재 배치의 유저 선택
    batch_users = user_indices[start_idx:end_idx]
    # [0, 1, 2]  (batch_size=3)

    # 배치 시퀀스 준비
    batch_seqs = []
    batch_exclude = []

    for user_idx in batch_users:
        seq = user_sequences[user_idx]      # train only
        full_seq = full_sequences[user_idx]  # train + valid
        batch_seqs.append(seq)
        batch_exclude.append(set(full_seq))  # validation 포함!

    # batch_seqs = [
    #     [15, 342, 88, 201, 456, 789],          # validation 제외
    #     [23, 45, 67, 89, 123],                  # validation 제외
    #     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],       # validation 제외
    # ]

    # batch_exclude = [
    #     {15, 342, 88, 201, 456, 789, 999},     # validation(999) 포함!
    #     {23, 45, 67, 89, 123, 888},             # validation(888) 포함!
    #     {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 777},  # validation(777) 포함!
    # ]

    # ============================================================
    # Step 3: 모델 추론
    # ============================================================
    top_items = model.predict(
        user_sequences=batch_seqs,
        topk=topk,
        exclude_items=batch_exclude
    )
    # shape: (3, 10)  # 3명의 유저, 각 10개 추천
    # [
    #     [501, 234, 789, 456, 123, 678, 901, 345, 567, 890],
    #     [456, 789, 123, 456, 678, 234, 567, 890, 345, 678],
    #     [123, 456, 789, 234, 567, 890, 345, 678, 901, 234],
    # ]

    # ============================================================
    # Step 4: ID 변환 및 결과 저장
    # ============================================================
    for i, user_idx in enumerate(batch_users):
        original_user_id = datamodule.idx2user[user_idx]
        # user_idx: 0 → original_user_id: 11

        item_indices = top_items[i]
        # [501, 234, 789, 456, 123, 678, 901, 345, 567, 890]

        for item_idx in item_indices:
            if item_idx > 0 and item_idx <= datamodule.num_items:
                original_item_id = datamodule.idx2item[item_idx]
                # item_idx: 501 → original_item_id: 4643

                results.append({
                    "user": original_user_id,
                    "item": original_item_id
                })

# ============================================================
# Step 5: 결과 저장
# ============================================================
# results = [
#     {"user": 11, "item": 4643},
#     {"user": 11, "item": 170},
#     {"user": 11, "item": 531},
#     ...
# ]
pred_df = pd.DataFrame(results)
pred_df.to_csv(output_path, index=False)
```

---

## 7. 모델 내부 상세 처리

### 7.1 시퀀스 전처리 (predict 메서드 내부)
```python
# 입력: batch_seqs
# [
#     [15, 342, 88, 201, 456, 789],          # 길이 6
#     [23, 45, 67],                           # 길이 3
# ]

# Step 1: mask token 추가 후 max_len으로 자르기
# max_len = 50, mask_token = 6808
masked_seqs = [
    [15, 342, 88, 201, 456, 789, 6808],     # 길이 7 → 50으로 자르기 불필요
    [23, 45, 67, 6808],                      # 길이 4 → 50으로 자르기 불필요
]

# Step 2: Padding (앞쪽에 pad_token=0 추가)
padded_seqs = [
    [0, 0, 0, ...(43개), 0, 15, 342, 88, 201, 456, 789, 6808],  # 길이 50
    [0, 0, 0, ...(46개), 0, 23, 45, 67, 6808],                    # 길이 50
]

# Step 3: numpy array 변환
seqs = np.array(padded_seqs, dtype=np.int64)
# shape: (2, 50)

# Step 4: 모델 forward
logits = self(seqs)
# shape: (2, 50, 6809)  # (batch, seq_len, vocab_size)

# Step 5: 마지막 위치의 logits 추출
scores = logits[:, -1, :]
# shape: (2, 6809)  # 마지막 mask token 위치의 예측

# Step 6: Invalid token masking
scores[:, 0] = -1e9      # pad_token
scores[:, 6808] = -1e9   # mask_token

# Step 7: 이미 본 아이템 masking
# batch_exclude = [{15, 342, 88, 201, 456, 789}, {23, 45, 67}]
scores[0, [15, 342, 88, 201, 456, 789]] = -1e9
scores[1, [23, 45, 67]] = -1e9

# Step 8: Top-K 선택
_, top_items = torch.topk(scores, k=10, dim=1)
# shape: (2, 10)

return top_items.cpu().numpy()
```

### 7.2 인덱스 범위 정리
```python
# Token 인덱스
pad_token = 0           # Padding
item_idx = 1 ~ 6807     # 실제 아이템 (num_items = 6807)
mask_token = 6808       # num_items + 1

# Vocabulary size
vocab_size = 6809       # num_items + 2 (pad + mask)

# User 인덱스
user_idx = 0 ~ 31359    # 0부터 시작 (num_users = 31360)
```

---

## 8. 시각화

### 8.1 데이터 구조 다이어그램
```
user_sequences (Dict)
├── 0: [15, 342, 88, 201, 456, 789]
├── 1: [23, 45, 67, 89, 123]
├── 2: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
└── ...
    ↓
user_indices (List)
[0, 1, 2, 3, ..., 31359]
    ↓ (batch_size=3)
batch_users: [0, 1, 2]
    ↓
batch_seqs                        batch_exclude
[                                 [
  [15, 342, 88, 201, 456, 789],    {15, 342, 88, 201, 456, 789},
  [23, 45, 67, 89, 123],            {23, 45, 67, 89, 123},
  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
]                                 ]
    ↓
model.predict()
    ↓
top_items (numpy array, shape: 3 x 10)
[
  [501, 234, 789, 456, 123, 678, 901, 345, 567, 890],
  [456, 789, 123, 456, 678, 234, 567, 890, 345, 678],
  [123, 456, 789, 234, 567, 890, 345, 678, 901, 234]
]
    ↓
results (List of Dict)
[
  {"user": 11, "item": 4643},
  {"user": 11, "item": 170},
  ...
]
```

### 8.2 배치 처리 타임라인
```
Time ────────────────────────────────────────────────>

Batch 0: users [0, 1, 2]
│
├─ batch_seqs: 3개 시퀀스
├─ batch_exclude: 3개 Set
├─ model.predict() → top_items: (3, 10)
└─ results += 30개 추천

Batch 1: users [3, 4, 5]
│
├─ batch_seqs: 3개 시퀀스
├─ batch_exclude: 3개 Set
├─ model.predict() → top_items: (3, 10)
└─ results += 30개 추천

...

Batch N: users [31357, 31358, 31359]
│
├─ batch_seqs: 3개 시퀀스
├─ batch_exclude: 3개 Set
├─ model.predict() → top_items: (3, 10)
└─ results += 30개 추천

Final: results → DataFrame → CSV
```

---

## 9. 요약

| 변수 | 타입 | Shape/구조 | 설명 |
|------|------|-----------|------|
| `full_sequences` | `Dict[int, List[int]]` | `{user_idx: [item_idx, ...]}` | 전체 유저의 train + valid 시퀀스 (입력 & 제외용) |
| `user_indices` | `List[int]` | `[user_idx, ...]` | 배치 순회용 유저 인덱스 리스트 |
| `batch_seqs` | `List[List[int]]` | `[[item_idx, ...], ...]` | 현재 배치의 시퀀스 리스트 (train + valid) |
| `batch_exclude` | `List[Set[int]]` | `[{item_idx, ...}, ...]` | 현재 배치의 제외 아이템 집합 (train + valid) |
| `top_items` | `numpy.ndarray` | `(batch_size, topk)` | 모델 예측 결과 (아이템 인덱스) |
| `results` | `List[Dict]` | `[{"user": id, "item": id}, ...]` | 최종 추천 결과 (원본 ID) |

---

**Last Updated**: 2025-12-25
