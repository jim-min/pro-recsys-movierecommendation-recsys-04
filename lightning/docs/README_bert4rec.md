# BERT4Rec Implementation Guide

## 개요

BERT4Rec (Bidirectional Encoder Representations from Transformers for Sequential Recommendation)의 PyTorch Lightning 구현입니다.

**논문**: [BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer](https://arxiv.org/abs/1904.06690)

## ✨ 최신 개선사항 (2025-12-29)

### Masking Strategy 개선: Random + Boost 방식

기존의 last-item masking 전략을 개선하여 **데이터 효율성과 성능을 동시에 향상**시켰습니다.

**기존 방식의 문제점**:
- `last_item_mask_ratio` 비율의 샘플만 마지막 아이템만 마스킹
- 이 샘플들은 매 epoch마다 **같은 마스킹 패턴** 반복
- 데이터 다양성 부족으로 학습 비효율

**개선된 방식** (Random + Boost):
```python
# 모든 샘플에 대해:
1. Random masking 적용 (매 epoch 다른 패턴)
2. last_item_mask_ratio 확률로 마지막 아이템 추가 마스킹
```

**개선 효과**:
| 항목 | 기존 방식 | 개선 방식 |
|------|----------|----------|
| 데이터 다양성 | ❌ 일부 샘플 고정 패턴 | ✅ 모든 샘플 매 epoch 변화 |
| Random masking | 90% 샘플만 | ✅ 100% 샘플 |
| Next-item focus | ✅ 학습됨 | ✅✅ 더 강화됨 |
| 학습 효율 | ⚠️ 중복 학습 | ✅ 효율적 |

**사용 방법**:
```yaml
model:
  random_mask_prob: 0.2        # 모든 샘플에 랜덤 마스킹
  last_item_mask_ratio: 0.05   # 5% 확률로 마지막 아이템 추가 마스킹
```

자세한 내용은 [Masking Strategy](#masking-strategy-논문-section-32) 섹션을 참조하세요.

---

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
# 기본 설정으로 학습 (validation 포함)
python train_bert4rec.py

# Full data 학습 (validation 없이 전체 데이터 사용)
python train_bert4rec.py data.use_full_data=true

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

**⚠️ Future Information Leakage 방지**: 추론 시 각 유저의 마지막 클릭 시점 이후 개봉한 영화는 자동으로 추천에서 제외됩니다.

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
  batch_size: 512                  # 배치 크기
  min_interactions: 3              # 최소 interaction 수
  num_workers: 4                   # DataLoader workers
  use_full_data: false             # Full data 학습 (validation 없음)
```

**데이터 포맷**: CSV with columns `user`, `item`, `time` (optional)

**Full Data 학습 모드** (`use_full_data: true`):
- Validation split 없이 전체 데이터로 학습
- Early stopping 자동 비활성화
- Checkpoint는 `train_loss` 기준으로 저장
- 최종 제출용 모델 학습 시 권장

### 모델 설정

```yaml
model:
  hidden_units: 256            # Hidden dimension (논문: dataset-dependent)
  num_heads: 4                 # Attention heads 수 (논문: dataset-dependent)
  num_layers: 2                # Transformer blocks 수 (논문: 2 for most datasets)
  max_len: 200                 # 최대 시퀀스 길이 (논문: 200 for ML-1M)
  dropout_rate: 0.2            # Dropout 확률 (논문: 0.2~0.5)
  random_mask_prob: 0.2        # 랜덤 마스킹 확률 (논문: 0.15, BERT와 동일)
  last_item_mask_ratio: 0.05   # 랜덤 마스킹 후 추가로 마지막 아이템을 마스킹할 확률 (next-item prediction 강화)
  share_embeddings: true       # Output layer와 embedding 공유 (논문: Yes)

  # 메타데이터 설정 (확장 기능)
  use_genre_emb: false      # 장르 임베딩 사용
  use_director_emb: false   # 감독 임베딩 사용
  use_writer_emb: false     # 작가 임베딩 사용
  use_title_emb: true       # 제목 임베딩 사용 (사전 계산된 임베딩)
  metadata_fusion: "gate"   # 융합 방법: "concat", "add", "gate"
  metadata_dropout: 0.1     # 메타데이터 임베딩 dropout
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
  num_epochs: 500                  # 최대 epoch 수 (논문: early stopping 사용)
  lr: 0.001                        # Learning rate (논문: 0.001)
  weight_decay: 0.0                # L2 regularization (논문: 명시 안됨)
  monitor_metric: "val_ndcg@10"    # 체크포인트 저장 기준
  early_stopping: true             # Early stopping (논문: 사용)
  early_stopping_patience: 20      # Patience
  accelerator: "auto"              # GPU/CPU 자동 선택
  precision: "32-true"             # 32-bit precision (안정성 우선) / "16-mixed": V100 최적화 (속도 우선)
```

## 아이템 메타데이터 기능 (확장)

BERT4Rec의 기본 item 임베딩에 추가로 **아이템 메타데이터**를 활용하여 성능을 향상시킬 수 있습니다.

### 지원하는 메타데이터 종류

1. **장르 (Genre)**: 영화의 장르 정보
   - 파일: `genres.tsv` (item, genre 컬럼)
   - 다중 장르 지원 (예: 액션, 드라마)
   - 평균 풀링으로 집계

2. **감독 (Director)**: 영화 감독 정보
   - 파일: `directors.tsv` (item, director 컬럼)
   - 단일 감독만 지원

3. **작가 (Writer)**: 영화 각본가 정보
   - 파일: `writers.tsv` (item, writer 컬럼)
   - 다중 작가 지원
   - 평균 풀링으로 집계

4. **제목 (Title)**: 사전 계산된 제목 임베딩
   - 파일: `titles.tsv` (item, title 컬럼)
   - BERT/Sentence-BERT 등으로 사전 계산된 768차원 벡터 사용

### 제목 임베딩 생성 방법

제목 임베딩은 영화 제목의 의미론적 정보를 벡터로 변환하여 추천 성능을 향상시킵니다.

#### 방법 0: 자동 스크립트 사용 (제목 + 장르, 가장 권장) ⭐

**특징**:
- 제목과 장르를 결합하여 더 풍부한 의미 정보 제공
- `mxbai-embed-large-v1` 모델 사용 (1024-dim, MTEB 최상위)
- 원클릭 실행으로 전처리 완료
- 장르 정보가 없는 아이템은 제목만 사용

**사용법**:

```bash
# 기본 사용 (mxbai-embed-large-v1, 제목 + 장르)
python scripts/preprocess_title_genre_embeddings.py

# 백업 생성 후 실행
python scripts/preprocess_title_genre_embeddings.py --backup-original

# 다른 모델 사용
python scripts/preprocess_title_genre_embeddings.py \
    --model-name sentence-transformers/all-mpnet-base-v2

# 배치 크기 조정 (GPU 메모리 부족 시)
python scripts/preprocess_title_genre_embeddings.py --batch-size 16

# 커스텀 출력 경로 지정
python scripts/preprocess_title_genre_embeddings.py \
    --data-dir ~/data/train \
    --output-path ~/data/train/custom_output/titles.tsv
```

**결합 텍스트 예시**:
```
원본 제목: "The Matrix"
장르: ["Action", "Sci-Fi"]
→ 결합 텍스트: "The Matrix. Genres: Action, Sci-Fi"

원본 제목: "Inception"
장르: []
→ 결합 텍스트: "Inception"
```

**장점**:
- 제목과 장르의 의미를 동시에 임베딩에 반영
- 장르 정보가 모델의 semantic understanding을 향상
- 별도의 장르 임베딩 없이도 장르 정보 활용 가능
- MTEB 최상위 모델로 최고 품질 보장

**출력 예시**:
```bash
2025-12-27 12:00:00 - INFO - Loading model: mixedbread-ai/mxbai-embed-large-v1
2025-12-27 12:00:05 - INFO - Model loaded successfully
2025-12-27 12:00:05 - INFO - Loading titles from: ~/data/train/titles.tsv
2025-12-27 12:00:05 - INFO - Loaded 6807 titles
2025-12-27 12:00:05 - INFO - Loading genres from: ~/data/train/genres.tsv
2025-12-27 12:00:05 - INFO - Loaded 20414 genre entries
2025-12-27 12:00:05 - INFO - Aggregated genres for 6807 items
2025-12-27 12:00:05 - INFO - Creating combined title + genre texts...

Example combined texts:
  [318] Shawshank Redemption, The (1994). Genres: Crime, Drama
  [2571] Matrix, The (1999). Genres: Action, Sci-Fi, Thriller
  [2959] Fight Club (1999). Genres: Action, Crime, Drama, Thriller
  [296] Pulp Fiction (1994). Genres: Comedy, Crime, Drama, Thriller
  [356] Forrest Gump (1994). Genres: Comedy, Drama, Romance, War

2025-12-27 12:00:05 - INFO - Generating embeddings...
Processing batches: 100%|████████████| 213/213 [02:15<00:00,  1.57it/s]
2025-12-27 12:02:20 - INFO - Generated embeddings shape: (6807, 1024)
2025-12-27 12:02:20 - INFO - Embedding dimension: 1024
2025-12-27 12:02:20 - INFO - Saving embeddings to: ~/data/train/title_embeddings/titles.tsv
2025-12-27 12:02:21 - INFO - Saved successfully!
================================================================================
Title + Genre Embedding Generation Complete!
  Model: mixedbread-ai/mxbai-embed-large-v1
  Total items: 6807
  Embedding dimension: 1024
  Items with genres: 6807
  Items without genres: 0
  Output file: ~/data/train/title_embeddings/titles.tsv
================================================================================
```

#### 방법 1: Sentence-BERT (수동 구현)

**특징**:
- 의미론적으로 유사한 문장이 가까운 벡터로 매핑됨
- 사전학습된 모델로 고품질 임베딩 생성
- 추천 시스템에서 가장 효과적

**구현 예시**:

```python
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np

# 1. Sentence-BERT 모델 로드
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# 또는 한국어: SentenceTransformer('jhgan/ko-sroberta-multitask')

# 2. 영화 제목 데이터 로드
titles_df = pd.read_csv('titles.tsv', sep='\t')  # columns: item, title

# 3. 제목 임베딩 생성
embeddings = model.encode(
    titles_df['title'].tolist(),
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True  # L2 정규화
)

# 4. TSV 파일로 저장
# 형식: item \t embedding_dim_0 embedding_dim_1 ... embedding_dim_767
with open('titles.tsv', 'w') as f:
    f.write('item\ttitle\n')
    for idx, (item_id, emb) in enumerate(zip(titles_df['item'], embeddings)):
        emb_str = ' '.join(map(str, emb))
        f.write(f'{item_id}\t{emb_str}\n')

print(f"Generated embeddings with shape: {embeddings.shape}")
# 출력: (num_items, 384) for all-MiniLM-L6-v2
```

**추천 모델**:
| 모델 | 차원 | 언어 | MTEB | 특징 |
|------|------|------|------|------|
| **`mixedbread-ai/mxbai-embed-large-v1`** ⭐ | **1024** | 영어 | **~65** | **MTEB 최상위권, 추천 시스템 최적화 (권장)** |
| `all-mpnet-base-v2` | 768 | 영어 | ~63 | 고품질 |
| `all-MiniLM-L6-v2` | 384 | 영어 | ~56 | 빠르고 효율적 (베이스라인) |
| `jhgan/ko-sroberta-multitask` | 768 | 한국어 | - | 한국어 최적화 |
| `paraphrase-multilingual-MiniLM-L12-v2` | 384 | 다국어 | - | 50개 언어 지원 |

#### 방법 2: BERT 평균 풀링

**특징**:
- BERT의 토큰 임베딩을 평균하여 문장 임베딩 생성
- Sentence-BERT보다 품질이 낮을 수 있음
- HuggingFace Transformers 사용

**구현 예시**:

```python
from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd

# 1. BERT 모델과 토크나이저 로드
model_name = 'bert-base-uncased'  # 또는 'bert-base-multilingual-cased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

# 2. 제목 데이터 로드
titles_df = pd.read_csv('titles.tsv', sep='\t')

def get_bert_embedding(text):
    """BERT 평균 풀링으로 문장 임베딩 생성"""
    # 토큰화
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)

    # Forward pass (gradient 계산 안함)
    with torch.no_grad():
        outputs = model(**inputs)

    # 마지막 hidden state: [batch=1, seq_len, hidden_dim=768]
    last_hidden = outputs.last_hidden_state

    # 평균 풀링 (CLS 토큰 제외)
    # Attention mask를 사용하여 패딩 제외
    mask = inputs['attention_mask'].unsqueeze(-1)  # [1, seq_len, 1]
    masked_embeddings = last_hidden * mask
    sum_embeddings = masked_embeddings.sum(dim=1)  # [1, hidden_dim]
    count = mask.sum(dim=1)  # [1, 1]
    mean_embedding = sum_embeddings / count  # [1, hidden_dim]

    return mean_embedding.squeeze().numpy()

# 3. 모든 제목에 대해 임베딩 생성
embeddings = []
for title in titles_df['title']:
    emb = get_bert_embedding(title)
    embeddings.append(emb)

embeddings = np.array(embeddings)

# 4. 저장 (방법 1과 동일)
with open('titles.tsv', 'w') as f:
    f.write('item\ttitle\n')
    for item_id, emb in zip(titles_df['item'], embeddings):
        emb_str = ' '.join(map(str, emb))
        f.write(f'{item_id}\t{emb_str}\n')

print(f"Generated BERT embeddings: {embeddings.shape}")
# 출력: (num_items, 768)
```

#### 방법 3: TF-IDF (경량화)

**특징**:
- 사전학습 모델 불필요
- 빠르고 메모리 효율적
- 의미론적 정보는 부족하지만 키워드 기반 매칭에는 효과적

**구현 예시**:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import pandas as pd
import numpy as np

# 1. 제목 데이터 로드
titles_df = pd.read_csv('titles.tsv', sep='\t')

# 2. TF-IDF 벡터화
vectorizer = TfidfVectorizer(
    max_features=5000,      # 상위 5000개 단어만 사용
    ngram_range=(1, 2),     # Unigram + Bigram
    min_df=2,               # 최소 2번 이상 등장한 단어
    stop_words='english'    # 불용어 제거
)

tfidf_matrix = vectorizer.fit_transform(titles_df['title'])
print(f"TF-IDF shape: {tfidf_matrix.shape}")  # (num_items, 5000)

# 3. 차원 축소 (SVD)
svd = TruncatedSVD(n_components=384, random_state=42)
embeddings = svd.fit_transform(tfidf_matrix)

# L2 정규화
from sklearn.preprocessing import normalize
embeddings = normalize(embeddings, norm='l2')

# 4. 저장 (방법 1과 동일)
with open('titles.tsv', 'w') as f:
    f.write('item\ttitle\n')
    for item_id, emb in zip(titles_df['item'], embeddings):
        emb_str = ' '.join(map(str, emb))
        f.write(f'{item_id}\t{emb_str}\n')

print(f"Generated TF-IDF embeddings: {embeddings.shape}")
# 출력: (num_items, 384)
```

#### 파일 형식 (titles.tsv)

생성된 임베딩 파일은 다음 형식을 따라야 합니다:

```tsv
item	title
1	0.123 -0.456 0.789 ... (768개 또는 384개 값)
2	-0.234 0.567 -0.890 ...
3	0.345 -0.678 0.123 ...
```

**중요 사항**:
- 첫 번째 줄: 헤더 (`item\ttitle`)
- 두 번째 줄부터: `item_id\t임베딩값1 임베딩값2 ... 임베딩값N`
- 구분자: 탭(`\t`)
- 임베딩 값: 공백으로 구분된 float 값들

#### 성능 비교

| 방법 | 차원 | 품질 | 속도 | 메모리 | 추천 용도 |
|------|------|------|------|--------|-----------|
| **Title + Genre (자동)** | **1024** | **⭐⭐⭐⭐⭐** | **⭐⭐⭐⭐** | **⭐⭐⭐** | **프로덕션 (최고 권장)** ⭐ |
| Sentence-BERT | 384-768 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | 프로덕션 |
| BERT 평균 풀링 | 768 | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | 실험 단계 |
| TF-IDF + SVD | 100-500 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 베이스라인 |

#### 데이터 로딩 (자동)

BERT4RecDataModule이 `titles.tsv` 파일을 자동으로 로드합니다:

```python
# src/data/bert4rec_data.py에서 자동 처리
def load_metadata(self):
    titles_path = os.path.join(self.data_dir, 'titles.tsv')
    if os.path.exists(titles_path):
        # TSV 파일 로드
        with open(titles_path, 'r') as f:
            next(f)  # Skip header
            for line in f:
                parts = line.strip().split('\t')
                item_id = int(parts[0])
                emb = np.array([float(x) for x in parts[1].split()])
                self.item_title_embs[item_id] = emb
```

**자동 감지**:
- 임베딩 차원은 첫 번째 아이템에서 자동으로 감지됨
- `title_embedding_dim` 파라미터가 모델에 자동으로 전달됨

#### 팁과 Best Practices

1. **L2 정규화 권장**:
   ```python
   # 모든 임베딩을 L2 정규화하여 벡터 크기를 1로 맞춤
   from sklearn.preprocessing import normalize
   embeddings = normalize(embeddings, norm='l2')
   ```

2. **배치 처리로 속도 향상**:
   ```python
   # Sentence-BERT는 배치 처리 지원
   embeddings = model.encode(titles, batch_size=64, show_progress_bar=True)
   ```

3. **GPU 활용**:
   ```python
   # GPU가 있으면 자동으로 사용
   model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
   ```

4. **차원 조정**:
   ```python
   # BERT4Rec 모델에서 title 임베딩은 Linear로 hidden_dim에 맞춰짐
   # title_projection: Linear(title_emb_dim, hidden_units)
   # 따라서 임베딩 차원은 384, 768 등 자유롭게 선택 가능
   ```

### 메타데이터 융합 전략

세 가지 융합 방법을 지원합니다:

#### 1. Concatenation (`metadata_fusion: "concat"`)

```
item_emb + genre_emb + director_emb + ... → [hidden_dim * N]
        ↓
   Linear Layer
        ↓
   [hidden_dim]
```

- 모든 임베딩을 연결하여 Linear layer로 축소
- 단순하지만 효과적
- 모든 메타데이터가 동등한 가중치

#### 2. Addition (`metadata_fusion: "add"`)

```
item_emb + genre_emb + director_emb + ... = final_emb
```

- Element-wise 덧셈으로 융합
- 파라미터 수 증가 없음
- 메타데이터 간 가중치 조절 불가

#### 3. Gate Fusion (`metadata_fusion: "gate"`) ⭐ 권장

```
features = [item_emb, genre_emb, director_emb, writer_emb, title_emb]
gates = softmax(Linear(concat(features)))  # [N] (합이 1)
        ↓
final_emb = Σ (gate_i * feature_i)
```

- **학습 가능한 가중치**로 각 메타데이터의 중요도 자동 조절
- Softmax로 정규화 (합이 1)
- TensorBoard로 gate 값 모니터링 가능 (아래 섹션 참고)

**예시**:
```
Gate values: item=0.54, title=0.36, writer=0.06, director=0.02, genre=0.01
→ item과 title이 가장 중요하게 학습됨
```

### 설정 방법

[bert4rec_v2.yaml](../configs/bert4rec_v2.yaml)에서 메타데이터 옵션 설정:

```yaml
model:
  # 사용할 메타데이터 선택
  use_genre_emb: true       # 장르
  use_director_emb: true    # 감독
  use_writer_emb: true      # 작가
  use_title_emb: true       # 제목

  # 융합 방법 선택
  metadata_fusion: "gate"   # "concat", "add", "gate" 중 선택
  metadata_dropout: 0.1     # Dropout 비율
```

**필요한 데이터 파일**:
- `~/data/train/genres.tsv`
- `~/data/train/directors.tsv`
- `~/data/train/writers.tsv`
- `~/data/train/titles.tsv`

파일이 없으면 해당 메타데이터는 자동으로 비활성화됩니다.

### 메타데이터 없이 사용

기본 BERT4Rec (논문 구현)만 사용하려면:

```yaml
model:
  use_genre_emb: false
  use_director_emb: false
  use_writer_emb: false
  use_title_emb: false
```

또는 간단히 메타데이터 파일을 제공하지 않으면 됩니다.

## Gate Fusion 모니터링 (TensorBoard)

Gate fusion 방법을 사용할 때, 각 메타데이터의 중요도를 TensorBoard로 실시간 모니터링할 수 있습니다.

### 사용 방법

#### 1. Gate fusion으로 학습 시작

```bash
python train_bert4rec.py model.metadata_fusion=gate
```

#### 2. TensorBoard 실행

```bash
# 학습 디렉토리에서 실행
tensorboard --logdir=saved/tensorboard_logs/bert4rec
```

브라우저에서 `http://localhost:6006` 접속

#### 3. Gate 값 확인

TensorBoard에서 다음 메트릭을 확인할 수 있습니다:

**SCALARS 탭**:
- `val_gate/item`: Item 임베딩의 가중치
- `val_gate/genre`: Genre 임베딩의 가중치
- `val_gate/director`: Director 임베딩의 가중치
- `val_gate/writer`: Writer 임베딩의 가중치
- `val_gate/title`: Title 임베딩의 가중치

### Gate 값 해석

Gate 값은 **softmax로 정규화**되어 합이 1입니다:

```
Epoch 50:
  val_gate/item     = 0.54  (54%)
  val_gate/title    = 0.36  (36%)
  val_gate/writer   = 0.06  (6%)
  val_gate/director = 0.02  (2%)
  val_gate/genre    = 0.01  (1%)
```

**해석**:
- **Item 임베딩(54%)**: 협업 필터링 신호가 가장 중요
- **Title 임베딩(36%)**: 콘텐츠 기반 신호 (의미론적 유사성)가 두 번째로 중요
- **기타 메타데이터(10%)**: 보조적인 역할

### 활용 사례

#### 1. 불필요한 메타데이터 제거

특정 메타데이터의 gate 값이 지속적으로 낮다면 (< 0.05), 해당 메타데이터를 비활성화하여 학습 속도를 향상시킬 수 있습니다:

```yaml
# Genre의 gate 값이 계속 0.01이라면
model:
  use_genre_emb: false  # 비활성화
```

#### 2. 중요 메타데이터 집중

Gate 값이 높은 메타데이터의 품질을 개선하여 성능을 향상시킬 수 있습니다:

```
Title gate가 0.36으로 높음
→ Title 임베딩 품질 개선 (더 나은 사전학습 모델 사용)
```

#### 3. 학습 안정성 확인

Gate 값이 epoch마다 크게 변동하면 학습이 불안정한 신호입니다:

```yaml
# Learning rate 감소 또는 dropout 조정
training:
  lr: 0.0005  # 낮춤
model:
  metadata_dropout: 0.2  # 증가
```

### TensorBoard 하이퍼파라미터 추적

모든 설정 파라미터가 **HPARAMS 탭**에 자동으로 기록됩니다:

**Data 설정**:
- `data/batch_size`
- `data/use_full_data`
- `data/seed`

**Model 설정**:
- `model/hidden_units`
- `model/num_layers`
- `model/metadata_fusion`
- `model/use_genre_emb`, `model/use_director_emb`, 등

**Training 설정**:
- `training/lr`
- `training/num_epochs`
- `training/early_stopping_patience`

**Metadata 차원**:
- `metadata/num_genres`
- `metadata/num_directors`
- `metadata/num_items`

이를 통해 여러 실험을 비교하고 최적의 하이퍼파라미터를 찾을 수 있습니다.

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

학습 시 두 가지 마스킹 전략을 혼합하여 사용합니다:

#### 1. Random Masking (기본 전략)

각 아이템은 **`random_mask_prob` 확률**로 랜덤하게 마스킹:
- **80%**: `[MASK]` 토큰으로 대체
- **10%**: 랜덤 아이템으로 대체
- **10%**: 원본 유지

#### 2. Last Item Masking Boost (추가 전략)

**모든 샘플**에 랜덤 마스킹을 적용한 후, **`last_item_mask_ratio` 확률**로 마지막 아이템을 추가로 마스킹:
- 랜덤 마스킹으로 다양한 패턴 학습 (매 epoch 다른 mask pattern)
- 추가로 마지막 아이템을 강제 마스킹하여 next-item prediction 강화
- 추론 시나리오와 일치하는 패턴 학습

**설정 예시** ([bert4rec_v2.yaml](../configs/bert4rec_v2.yaml)):
```yaml
model:
  random_mask_prob: 0.2        # 랜덤 마스킹 확률
  last_item_mask_ratio: 0.05   # 랜덤 마스킹 후 추가로 마지막 아이템을 마스킹할 확률
```

**동작 방식** (개선됨):
```
모든 샘플에 대해:
1. Random masking 적용 (20% 확률로 각 아이템)
2. 5% 확률로 마지막 아이템을 추가로 마스킹

매 epoch마다 다른 mask pattern 생성 → 데이터 다양성 향상
```

**장점**:
- ✅ **매 epoch 다양한 학습 데이터** (기존: 같은 last-item mask 반복)
- ✅ **Random masking + Last-item masking 효과 동시 획득**
- ✅ 추론 시와 동일한 마스킹 패턴으로 학습하여 성능 향상
- ✅ 시퀀스의 마지막 아이템 예측 능력 강화

**구현 상세** ([bert4rec_data.py:81-90](../src/data/bert4rec_data.py#L81-L90)):
```python
# Always apply random masking first (for data diversity)
tokens, labels = self._random_mask_sequence(seq)

# Additionally mask the last item with probability last_item_mask_ratio
if len(seq) > 0 and np.random.random() < self.last_item_mask_ratio:
    # Force mask the last item (overwrite if already masked)
    last_idx = len(seq) - 1
    tokens[last_idx] = self.mask_token
    labels[last_idx] = seq[last_idx]  # Original item as label
```

**추론 시**:
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

## 학습 모드

### Standard 모드 (기본값: `use_full_data: false`)

```bash
python train_bert4rec.py
```

**특징**:
- ✅ Train/Validation split 자동 생성 (마지막 아이템)
- ✅ Validation 메트릭 계산 (Hit@10, NDCG@10)
- ✅ Early stopping 활성화
- ✅ Checkpoint: `val_ndcg@10` 기준 저장
- ✅ 하이퍼파라미터 튜닝 시 권장

**체크포인트 파일명**: `bert4rec-{epoch:02d}-{val_ndcg@10:.4f}.ckpt`

### Full Data 모드 (`use_full_data: true`)

```bash
python train_bert4rec.py data.use_full_data=true
```

**특징**:
- ✅ 전체 데이터로 학습 (validation split 없음)
- ✅ Early stopping 자동 비활성화
- ✅ Checkpoint: `train_loss` 기준 저장
- ✅ 최종 제출용 모델 학습 시 권장
- ⚠️ Overfitting 확인 불가 (epoch 수 조절 필요)

**체크포인트 파일명**: `bert4rec-{epoch:02d}-{train_loss:.4f}.ckpt`

**자동 조정 사항**:
| 설정 | Standard 모드 | Full Data 모드 |
|------|--------------|----------------|
| Train data | `seq[:-1]` | `seq` (전체) |
| Validation | 활성화 | 비활성화 |
| Early stopping | 활성화 | 비활성화 |
| Checkpoint metric | `val_ndcg@10` | `train_loss` |
| Checkpoint mode | `max` | `min` |

## PyTorch Lightning 기능

### 자동 제공되는 기능

1. **분산 학습**: Multi-GPU, TPU 자동 지원
2. **체크포인트 관리**:
   - `last.ckpt`: 마지막 epoch
   - `best.ckpt`: 최고 성능 모델
3. **Early Stopping**: Validation 성능 개선 없으면 중단 (Standard 모드)
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

# 학습 안정성을 위해 32-bit precision 사용 (또는 "16-mixed"로 속도 향상)
training:
  precision: "32-true"
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

# 학습 안정성을 위해 32-bit precision 사용 (또는 "16-mixed"로 속도 향상)
training:
  precision: "32-true"
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

#### 1. 데이터 로딩 ([src/data/bert4rec_data.py](../src/data/bert4rec_data.py))

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
    for user_idx in users:
        last_click_year = self.user_last_click_years[user_idx]
        # 개봉년도 > 마지막 클릭 년도인 영화 필터링
        future_items[user_idx] = {
            item_idx for item_idx in items
            if self.item_years[item_idx] > last_click_year
        }
    return future_items
```

#### 3. 추론 시 자동 적용 ([predict_bert4rec.py](../predict_bert4rec.py))

```python
# Future items 가져오기
future_item_sequences = datamodule.get_future_item_sequences()

# 추천에서 제외 (train + valid + future items)
for user_idx in batch_users:
    exclude_set = set(full_seq)  # 이미 본 영화
    exclude_set.update(future_item_sequences[user_idx])  # Future 영화

    top_items = model.predict(
        user_sequences=batch_seqs,
        topk=topk,
        exclude_items=exclude_set  # 제외 목록
    )
```

### 로그 예시

```
[INFO] Loading item metadata...
[INFO] Loaded 6807 items with release year info
[INFO] Calculated last click year for 31360 users
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
# predict_bert4rec.py 수정
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

## 테스트 (pytest)

BERT4Rec 모델의 품질을 보장하기 위해 pytest 기반 단위 테스트를 제공합니다.

### 테스트 환경 설정

```bash
# 테스트 의존성 설치
pip install -r lightning/requirements-dev.txt

# 또는 개별 설치
pip install pytest pytest-cov
```

### 테스트 실행

#### 1. 모든 테스트 실행

```bash
cd lightning
pytest
```

#### 2. 특정 테스트 파일 실행

```bash
# BERT4Rec 모델 테스트만 실행
pytest tests/unit/test_bert4rec_model.py

# 특정 테스트 클래스만 실행
pytest tests/unit/test_bert4rec_model.py::TestBERT4RecForwardPass

# 특정 테스트 함수만 실행
pytest tests/unit/test_bert4rec_model.py::TestBERT4RecForwardPass::test_forward_pass_without_metadata
```

#### 3. 상세 출력 모드

```bash
# 상세 로그 출력
pytest -v

# 더 상세한 출력 (테스트 진행 상황 표시)
pytest -vv

# 표준 출력도 표시 (print 문 출력)
pytest -s
```

#### 4. 커버리지 확인

```bash
# 커버리지와 함께 테스트 실행
pytest --cov=src --cov-report=html

# HTML 리포트 확인
open htmlcov/index.html
```

#### 5. 마커(Marker)로 필터링

```bash
# Unit 테스트만 실행
pytest -m unit

# GPU 테스트만 실행 (GPU 있을 때)
pytest -m gpu

# GPU 테스트 제외
pytest -m "not gpu"
```

### 테스트 구조

```
lightning/tests/
├── conftest.py                    # 공통 fixtures (샘플 데이터, 모델 설정)
├── unit/
│   ├── test_bert4rec_model.py     # 모델 단위 테스트
│   └── TEST_SUMMARY.md            # 테스트 요약 문서
└── integration/                    # 통합 테스트 (예정)
```

### 주요 테스트 카테고리

#### 1. 모델 초기화 테스트 (`TestBERT4RecModelInitialization`)

```python
# 모델이 올바르게 생성되는지 확인
def test_model_creates_successfully()
def test_model_has_correct_attributes()
def test_special_tokens_initialized()
def test_metadata_embeddings_created()
```

#### 2. Forward Pass 테스트 (`TestBERT4RecForwardPass`)

```python
# Forward pass가 올바른 형태의 출력을 생성하는지 확인
def test_forward_pass_without_metadata()
def test_forward_pass_with_metadata()
def test_forward_output_dtype()
def test_forward_no_nan_or_inf()
```

#### 3. Metadata Fusion 테스트 (`TestBERT4RecMetadataFusion`)

```python
# 세 가지 융합 전략 테스트
def test_concat_fusion()
def test_add_fusion()
def test_gate_fusion()
def test_invalid_fusion_raises_error()
```

#### 4. Gate Values 테스트 (`TestBERT4RecGateValues`)

```python
# Gate 값이 올바르게 반환되고 정규화되는지 확인
def test_gate_fusion_returns_values()
def test_gate_values_sum_to_one()        # Softmax 검증
def test_gate_values_are_positive()       # [0,1] 범위 검증
def test_gate_num_features_matches_enabled_embeddings()
```

#### 5. Training/Validation Step 테스트

```python
# 학습과 검증이 올바르게 동작하는지 확인
def test_training_step_without_metadata()
def test_training_step_with_metadata()
def test_validation_step_logs_gate_values()
```

#### 6. Edge Cases 테스트 (`TestBERT4RecSpecialCases`)

```python
# 극단적인 경우 처리 확인
def test_all_padding_sequence()
def test_all_mask_tokens()
def test_single_item_sequence()
def test_max_length_sequence()
```

### 샘플 Fixtures (conftest.py)

테스트에서 사용 가능한 공통 fixtures:

```python
@pytest.fixture
def sample_config():
    """기본 BERT4Rec 설정"""

@pytest.fixture
def sample_config_no_metadata():
    """메타데이터 없는 설정"""

@pytest.fixture
def bert4rec_model():
    """메타데이터 포함 모델"""

@pytest.fixture
def bert4rec_model_no_metadata():
    """메타데이터 없는 모델"""

@pytest.fixture
def sample_batch():
    """샘플 배치 데이터 (sequences, labels)"""

@pytest.fixture
def sample_metadata():
    """샘플 메타데이터 딕셔너리"""

@pytest.fixture
def temp_data_dir():
    """임시 데이터 디렉토리 (자동 정리)"""
```

### 새로운 테스트 추가

새로운 기능을 추가할 때 테스트도 함께 작성하세요:

```python
# tests/unit/test_bert4rec_model.py

import pytest
from src.models.bert4rec import BERT4Rec

@pytest.mark.unit
class TestMyNewFeature:
    """새로운 기능 테스트"""

    def test_feature_works_correctly(self, bert4rec_model):
        """기능이 올바르게 동작하는지 확인"""
        result = bert4rec_model.my_new_method()
        assert result is not None
        assert result.shape == (expected_shape)

    def test_feature_handles_edge_case(self):
        """엣지 케이스 처리 확인"""
        # 테스트 코드
        pass
```

### CI/CD 통합

GitHub Actions 등의 CI/CD 파이프라인에서 자동 테스트:

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
      - run: pip install -r requirements-dev.txt
      - run: pytest -v --cov=src
```

### 테스트 베스트 프랙티스

1. **작은 단위로 테스트**: 각 테스트는 하나의 기능만 검증
2. **독립성 유지**: 테스트 간 의존성 없이 독립적으로 실행 가능
3. **빠른 실행**: 단위 테스트는 빠르게 실행되어야 함
4. **명확한 이름**: 테스트 함수 이름만으로 무엇을 테스트하는지 알 수 있게
5. **Fixtures 활용**: 중복 코드 제거를 위해 공통 설정은 fixtures로
6. **예외 테스트**: 정상 케이스뿐만 아니라 예외 상황도 테스트

## 참고 자료

- [BERT4Rec 논문](https://arxiv.org/abs/1904.06690)
- [BERT 원본 논문](https://arxiv.org/abs/1810.04805)
- [PyTorch Lightning 문서](https://lightning.ai/docs/pytorch/stable/)
- [Hydra 문서](https://hydra.cc/)

## 라이선스

MIT License
