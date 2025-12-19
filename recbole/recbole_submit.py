import argparse
import torch
import pandas as pd
import numpy as np

from recbole.config import Config
from recbole.data import create_dataset
from recbole.utils import init_seed, get_model


# ==================================================
# Utils
# ==================================================
def load_checkpoint(path, device):
    # PyTorch 2.6+ 대응
    return torch.load(path, map_location=device, weights_only=False)


def build_global_popular_items(train_df, topk):
    return (
        train_df.groupby("item")
        .size()
        .sort_values(ascending=False)
        .index.tolist()[:topk]
    )


# ==================================================
# Main
# ==================================================
def main(args):
    # ==================================================
    # 0) Load original train.csv (⭐ submission 기준)
    # ==================================================
    train_df = pd.read_csv(args.train_csv)

    # ⭐⭐⭐ 타입 통일 (가장 중요)
    train_df["user"] = train_df["user"].astype(str)
    train_df["item"] = train_df["item"].astype(str)

    all_users = train_df["user"].astype(str).drop_duplicates().tolist()


    # user -> set(items)  (대회 기준 seen-item)
    user2seen = (
        train_df.groupby("user")["item"]
        .apply(set)
        .to_dict()
    )

    # fallback: global popular items
    popular_items = build_global_popular_items(train_df, args.topk)

    # ==================================================
    # 1) RecBole config & dataset (학습/추론 기준)
    # ==================================================
    config = Config(
        model=args.model,
        dataset=args.dataset,
        config_file_list=[args.config],
    )
    init_seed(config["seed"], config["reproducibility"])
    device = config["device"]

    dataset = create_dataset(config)
    R = dataset.inter_matrix(form="csr")
    num_items = R.shape[1]

    # --------------------------
    # user mapping (⭐ 문자열 통일)
    # --------------------------
    uid_map = dataset.field2id_token[dataset.uid_field]  # list / ndarray

    token_to_internal_user = {}
    for internal_id, token in enumerate(uid_map):
        token = str(token)
        if token != "[PAD]":
            token_to_internal_user[token] = internal_id

    # --------------------------
    # item mapping (⭐ 문자열 통일)
    # --------------------------
    iid_map = dataset.field2id_token[dataset.iid_field]

    token_to_internal_item = {}
    internal_to_item = {}
    for internal_id, token in enumerate(iid_map):
        token = str(token)
        if token != "[PAD]":
            token_to_internal_item[token] = internal_id
            internal_to_item[internal_id] = token

    # 🔍 디버깅 출력 (이제 intersection > 0 이어야 정상)
    print("train user dtype sample:", train_df["user"].iloc[0], type(train_df["user"].iloc[0]))
    print("uid_map sample:", uid_map[1], type(uid_map[1]))
    print("all_users:", len(all_users))
    print("recbole_users:", len(token_to_internal_user))
    print("intersection:", len(set(all_users) & set(token_to_internal_user.keys())))

    # ==================================================
    # 2) Load model
    # ==================================================
    model = get_model(config["model"])(config, dataset).to(device)
    ckpt = load_checkpoint(args.checkpoint, device)

    if "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    elif "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        raise KeyError("Checkpoint에 state_dict가 없습니다.")

    model.eval()

    # ==================================================
    # 3) Predict (RecBole 가능 user만)
    # ==================================================
    pred_dict = {}
    topk = args.topk

    with torch.no_grad():
        for user_token, u_internal in token_to_internal_user.items():

            interaction = {
                model.USER_ID: torch.tensor([u_internal], device=device)
            }

            # ⭐ 공식 추론 경로
            if hasattr(model, "full_sort_predict"):
                scores = model.full_sort_predict(interaction)
            else:
                interaction[model.ITEM_ID] = torch.arange(num_items, device=device)
                scores = model.predict(interaction)

            scores = scores.view(-1).cpu().numpy()

            # --------------------------
            # 대회 기준 seen-item 마스킹
            # --------------------------
            seen_tokens = user2seen.get(user_token, set())
            seen_internal_ids = [
                token_to_internal_item[it]
                for it in seen_tokens
                if it in token_to_internal_item
            ]
            scores[seen_internal_ids] = -np.inf

            # PAD item 방어
            pad_item_internal = token_to_internal_item.get("[PAD]")
            if pad_item_internal is not None:
                scores[pad_item_internal] = -np.inf

            # top-k
            top_items = np.argpartition(-scores, topk)[:topk]
            top_items = top_items[np.argsort(-scores[top_items])]

            pred_dict[user_token] = [
                internal_to_item[int(i)] for i in top_items
            ]

    # ==================================================
    # 4) Fill missing users with fallback (⭐ seen 제외)
    # ==================================================
    missing_users = set(all_users) - set(pred_dict.keys())
    print(f"❗ Missing users filled with fallback: {len(missing_users)}")

    for u in missing_users:
        seen = user2seen.get(u, set())
        fill = [it for it in popular_items if it not in seen][:topk]

        # 혹시 10개 안 차면 보강
        if len(fill) < topk:
            fill += [it for it in popular_items if it not in fill][: (topk - len(fill))]

        pred_dict[u] = fill

    # ==================================================
    # 5) Build submission (⭐ 대회 기준)
    # ==================================================
    rows = []
    for u in all_users:
        items = pred_dict[u]
        assert len(items) == topk
        for it in items:
            rows.append([u, it])

    submission = pd.DataFrame(rows, columns=["user", "item"])
    submission.to_csv(args.output, index=False)

    print(f"✅ submission saved to {args.output}")
    print(f"Total users: {len(all_users)}, total rows: {len(rows)}")


# ==================================================
# Entry
# ==================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--output", type=str, default="submission.csv")
    parser.add_argument("--topk", type=int, default=10)

    args = parser.parse_args()
    main(args)
