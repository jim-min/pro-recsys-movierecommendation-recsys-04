import os
import json
import argparse
import torch
import pandas as pd
import numpy as np

from data_utils import (
    read_interactions,
    encode_ids,
    build_user_item_matrix,
    train_valid_split_random,
    set_seed
)
from model import FactorizationMachine # 네가 정의한 FM 모델 클래스
from trainer import FMTrainer
from recommend import recommend_topk

def prepare_item_features_from_json(json_path, it2i, num_users, num_items):
    """
    Ml_item2attributes.json 파일을 읽어서 FM용 속성 행렬과 마스크를 생성함.
    모든 속성 인덱스에는 (num_users + num_items)만큼의 Offset을 더해 인덱스 겹침을 방지함.
    """
    if not os.path.exists(json_path):
        print(f"⚠️ {json_path}를 찾을 수 없어 속성 없이 학습을 진행합니다.")
        return torch.zeros((num_items, 1), dtype=torch.long), torch.zeros((num_items, 1)), 0

    with open(json_path, 'r') as f:
        item2attr = json.load(f)

    # 1. 속성의 전체 종류 수 파악
    all_attrs = []
    for attrs in item2attr.values():
        all_attrs.extend(attrs)
    
    num_unique_attrs = max(all_attrs) + 1 if all_attrs else 0
    # 아이템당 최대 속성 개수 파악 (Padding용)
    max_attrs = max(len(v) for v in item2attr.values()) if item2attr else 1

    # 2. 결과 행렬 초기화
    item_attr_mat = np.zeros((num_items, max_attrs), dtype=np.int64)
    item_attr_mask = np.zeros((num_items, max_attrs), dtype=np.float32)

    # 3. 속성 인덱스에 Offset 적용
    # User(0~U-1) + Item(U~U+I-1) + Attr(U+I~...)
    attr_offset = num_users + num_items

    for item_id_str, attrs in item2attr.items():
        item_id = int(item_id_str)
        if item_id in it2i:
            idx = it2i[item_id]
            for i, a in enumerate(attrs):
                item_attr_mat[idx, i] = a + attr_offset
                item_attr_mask[idx, i] = 1.0

    print(f"✅ 속성 데이터 로드 완료: 종류 {num_unique_attrs}개, 최대 속성 수 {max_attrs}개")
    return torch.tensor(item_attr_mat), torch.tensor(item_attr_mask), num_unique_attrs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/data/ephemeral/home/Seung/data/train/")
    parser.add_argument("--output_dir", default="/data/ephemeral/home/Seung/output/FM/")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--steps_per_epoch", type=int, default=500)
    parser.add_argument("--embed_dim", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early_stop_patience", type=int, default=15)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, "best_fm.pt")

    # [1] 데이터 로드 및 인코딩
    df = read_interactions(args.data_dir)
    df_enc, u2i, i2u, it2i, i2it = encode_ids(df)
    num_users, num_items = len(u2i), len(it2i)

    # [2] 검증 데이터 분리 (대회 공식 Recall 측정용)
    train_df, valid_gt = train_valid_split_random(df_enc, num_users, valid_ratio=args.valid_ratio, seed=args.seed)
    train_mat = build_user_item_matrix(train_df, num_users, num_items)

    # [3] JSON에서 아이템 속성 로드 및 Offset 적용
    json_path = os.path.join(args.data_dir, "Ml_item2attributes.json")
    item_attr_mat, item_attr_mask, num_unique_attrs = prepare_item_features_from_json(
        json_path, it2i, num_users, num_items
    )
    
    # 전체 피처 수 계산 (User ID + Item ID + Attribute IDs)
    total_features = num_users + num_items + num_unique_attrs

    # [4] 모델 초기화
    model = FactorizationMachine(total_features, args.embed_dim)
    
    # [5] 트레이너 설정 (Recall 기반 Early Stopping 내장)
    trainer = FMTrainer(
        model=model,
        train_mat=train_mat,
        valid_user_pos=valid_gt,
        num_items=num_items,
        item_attr_mat=item_attr_mat,
        item_attr_mask=item_attr_mask,
        user_offset=0,
        item_offset=num_users,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
        ckpt_path=ckpt_path,
        early_stop_patience=args.early_stop_patience
    )

    # [6] 학습 수행
    print("🚀 FM 학습을 시작합니다...")
    trainer.train(epochs=args.epochs, batch_size=args.batch_size, 
                  steps_per_epoch=args.steps_per_epoch, topk=args.topk)

    # [7] 최적 모델 로드 및 최종 추천 생성 (배치 처리)
    print("🏁 최종 추천 리스트 생성을 시작합니다...")
    best_state = torch.load(ckpt_path)
    model.load_state_dict(best_state)
    
    # recommend_topk는 이제 배치 단위로 작동하여 메모리와 속도를 모두 잡음
    rec = recommend_topk(
        model=model,
        train_mat=train_mat,
        item_attr_mat=item_attr_mat,
        item_attr_mask=item_attr_mask,
        topk=args.topk,
        device=device,
        user_offset=0,
        item_offset=num_users,
        user_batch_size=256 # 메모리 상황에 따라 조절 가능
    )

    # [8] submission.csv 저장
    rows = []
    for u_idx in range(num_users):
        for it_idx in rec[u_idx]:
            rows.append((i2u[u_idx], i2it[int(it_idx)]))
            
    pd.DataFrame(rows, columns=["user", "item"]).to_csv(
        os.path.join(args.output_dir, "submission.csv"), index=False
    )
    print(f"✅ 제출 파일이 {args.output_dir}에 성공적으로 저장되었습니다!")

if __name__ == "__main__":
    main()