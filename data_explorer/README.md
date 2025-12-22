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
1.  **Configuration (Sidebar)**
    *   **Data Directory**: 데이터 파일이 위치한 경로를 지정합니다. (기본값: `data/train`)
    *   **Time Range**: 전체 데이터 중 분석하고 싶은 기간을 슬라이더로 조정합니다.
    *   **User Selection**: 특정 유저를 직접 선택하거나(Specific Users), ID 범위로 선택합니다(User ID Range).
2.  **Dashboard Tabs**
    *   **User Interactions & Stats**: 선택한 유저의 시간대별 인터랙션(Scatter Plot)과 기간별 활동량(Bar Chart)을 시각화합니다.
    *   **Item Info**: Movie ID를 선택하여 영화의 메타데이터(장르, 감독 등)를 조회합니다.
    *   **Overall Statistics**: Top/Bottom K 유저 및 아이템 통계를 조회하고, 관심 유저를 바로 Sidebar 선택에 추가할 수 있습니다.

---

## 2. 구현상 주요 고려사항 (Implementation Considerations)

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
