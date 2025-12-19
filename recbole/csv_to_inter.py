import pandas as pd

# 1. csv 로드
df = pd.read_csv("mylens.inter.csv")

# 2. 컬럼명 RecBole 규칙에 맞게 변경
df = df.rename(columns={
    "user": "user_id",
    "item": "item_id",
    "time": "timestamp"
})

# 3. timestamp 타입 보장
df["timestamp"] = df["timestamp"].astype(float)

# 4. tsv로 저장 (임시)
out_path = "/data/ephemeral/home/Seung/recbole/dataset/mylens.inter"
df.to_csv(out_path, sep="\t", index=False)

# 5. 헤더에 타입 선언 붙이기
with open(out_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

lines[0] = "user_id:token\titem_id:token\ttimestamp:float\n"

with open(out_path, "w", encoding="utf-8") as f:
    f.writelines(lines)

print("RecBole용 mylens.inter 생성 완료!")
