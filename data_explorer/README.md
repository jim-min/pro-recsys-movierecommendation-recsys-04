# RecSys Data Explorer

RecSys 데이터(`train_ratings.csv` 등)를 시각적으로 탐색하고 분석하기 위한 Streamlit 기반 대시보드입니다.

## 1. 사용법 (Usage)

### 설치 및 실행
필요한 라이브러리가 설치되어 있어야 합니다 (`streamlit`, `pandas`, `plotly`).

```bash
# 앱 실행
streamlit run data_explorer/app.py
```

### 주요 기능
탭 구조 (4개의 독립적인 분석 탭)
#### 1. 👥 User Subset Analysis
특정 사용자들에 대한 심층 분석
- **User Selection**: Specific Users / User ID Range / Random Sample
- **시각화**: 사용자별 인터랙션 타임라인
- **통계**: 사용자별 평가 수, 고유 아이템 수, 활동 기간
- **시간대별 분포**: Day/Week/Month/Year 단위 집계

#### 2. 🌐 All Users Analysis  
전체 사용자에 대한 패턴 분석
- **Early Account Activity Analysis** (핵심 기능)
  - 30분 단위로 초기 평가 패턴 분석
  - Bulk Rater 자동 탐지 및 통계
  - 평가 속도(ratings/minute), 초기 평가 비율 등
  - 개별 사용자 타임라인 시각화
- **User Rankings**: Top/Bottom K 사용자

#### 3. 🎬 Item Subset Analysis
특정 아이템들에 대한 심층 분석
- **Item Selection**: Specific Items / Item ID Range / Random Sample  
- **통계**: 아이템별 평가 수, 고유 사용자 수
- **인기도 추이**: 시간대별 아이템 평가 패턴
- **상세 정보**: 제목, 장르, 감독, 작가, 연도 등

#### 4. 📊 All Items Analysis
전체 아이템에 대한 패턴 분석
- **Item Rankings**: Top/Bottom K 아이템
- **분포 분석**: 아이템별 평가 수 분포 히스토그램
- **Interaction Matrix**: User-Item 조합 통계

---

## 2. 필요한 데이터 파일

데이터는 `data/train/` 디렉토리에 다음 파일들이 있어야 합니다:

```
data/train/
├── train_ratings.csv       # 필수: user, item, time 컬럼
├── titles.tsv              # 선택: item, title 컬럼
├── genres.tsv              # 선택: item, genre 컬럼
├── directors.tsv           # 선택: item, director 컬럼
├── writers.tsv             # 선택: item, writer 컬럼
├── years.tsv               # 선택: item, year 컬럼
└── Ml_item2attributes.json # 선택: {item_id: [genre_ids]} 매핑
```
---

## 3. 구현상 주요 고려사항 (Implementation Considerations)

### 데이터 처리
*   **Implicit Feedback 가정**: `train_ratings.csv`에 명시적인 평점(score) 컬럼이 없을 경우, 인터랙션 발생 여부(Event)로 간주하여 시각화했습니다.
*   **대용량 데이터 Caching**: 매번 CSV를 로드하지 않도록 Streamlit의 `@st.cache_data`를 사용하여 데이터 로딩 속도를 최적화했습니다.

### 시각화 및 UX
*   **시간 축 정렬 (Temporal Axis)**:
    *   데이터가 없는 기간(Empty Periods)도 명확히 표현하기 위해 Pandas `resample` 기능을 사용하여 0값을 채웠습니다.
    *   Plotly Bar Chart에서 막대와 눈금의 정확한 정렬을 위해, 리샘플링 후 날짜를 Categorical String 포맷(예: `YYYY`, `YYYY-MM`)으로 변환하여 시각화했습니다.
*   **Session State 동기화**:
    *   통계 탭에서 'Add to Selection' 버튼 클릭 시, Sidebar의 선택 상태를 즉시 업데이트하기 위해 `st.session_state`와 콜백 함수(`on_click`)를 활용했습니다.
    *   이는 Streamlit의 리렌더링 사이클에서 상태 충돌을 방지합니다.

---

## 3. 수정 히스토리 (Modification History)

### v1.0.0
*   기초 기능 구현: 데이터 로더, 유저 인터랙션 Scatter Plot, 아이템 정보 조회.

### v1.1.0
*   **통계 기능 강화**: Top/Bottom K 유저/아이템 랭킹 기능 추가.
*   **필터 UI 개선**: Time Range Slider 추가, User Range(범위) 선택 기능 추가.

### v1.2.0
*   **Temporal Chart 개선 (Bug Fix)**:
    *   기간별 통계(Temporal Distribution)에서 X축이 연속적이지 않던 문제 수정.
    *   `resample`을 도입하여 데이터가 없는 연도/월도 0으로 표시되도록 개선.
    *   X축 눈금(Ticks) 및 그리드 표시 추가.

### v1.3.0
*   **헤더 메트릭 추가**: 앱 상단에 Total Users, Total Movies, Total Interactions 표시.
*   **UX 개선**: 중복된 로딩 성공 메시지 제거.
*   **Session State 버그 수정**: Sidebar 유저 선택과 통계 탭 간의 싱크 문제 해결 (Callback 방식 도입).

### v2.0.0
*   **탭 구분**: 4개의 대시보드 탭 생성. 각각 일부유저 / 전체유저 / 일부아이템 / 전체아이템 데이터에 관한 EDA를 몰아 볼 수 있도록 함.
    *   일부 데이터를 선정할 때, 1)특정 2)범위 3)랜덤샘플링 세 가지 방법을 설정

### v2.1.0
*   **탭 구분**: 6개의 대시보드 탭 생성. 각각 퀄리티 체크 / 일부유저 / 전체유저 / 일부아이템 / 전체아이템 데이터 / 추가 eda 에 관한 분석 내용을 몰아 볼 수 있도록 함.
    *   팀원들의 EDA를 합친 버전