import numpy as np
import pandas as pd

# 데이터 로드
static = pd.read_csv("/data/ephemeral/home/jimin/ease22.csv")
sequential = pd.read_csv("/data/ephemeral/home/jimin/bert22.csv")

users = len(static) // 10

# Sequential은 Top 2만 필요하므로 가져옵니다.
seq_items_full = sequential["item"].to_numpy()[: users * 10].reshape(users, 10)
items_2 = seq_items_full[:, :2]

# Static은 중복이 발생할 경우 8번째 이후의 아이템도 끌어다 써야 하므로 전체(10개)를 가져옵니다.
stat_items_full = static["item"].to_numpy()[: users * 10].reshape(users, 10)

final_items = []

# 각 유저별로 루프를 돌며 중복을 제거하고 채워넣습니다.
for seq_row, stat_row in zip(items_2, stat_items_full):
    # 1. Sequential에서 가져온 2개는 무조건 포함
    current_items = list(seq_row)
    seen = set(current_items)  # 중복 체크용 집합

    # 2. Static에서 순서대로 확인하며 이미 들어간 게 아니면 추가
    for item in stat_row:
        if item not in seen:
            current_items.append(item)
            seen.add(item)
        
        # 10개가 채워지면 중단
        if len(current_items) == 10:
            break
            
    final_items.append(current_items)

# 결과를 다시 1차원 배열로 펼칩니다.
items_10 = np.array(final_items).reshape(-1)

# 저장
answers = pd.DataFrame(
    {
        "user": static["user"], # static이 user순으로 정렬되어 있다는 가정
        "item": items_10,
    }
)

answers.to_csv("/data/ephemeral/home/jimin/reasonable_ensemble.csv", index=False)