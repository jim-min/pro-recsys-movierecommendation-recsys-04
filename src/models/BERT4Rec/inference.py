import hydra
from omegaconf import DictConfig

import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from collections import defaultdict

from model import BERT4Rec, load_checkpoint, load_data


@hydra.main(version_base=None, config_path="../../config/BERT4rec", config_name="config")
def main(cfg: DictConfig):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("현재 사용 디바이스:", device)

    ckpt_path = cfg.get('inference', {}).get('checkpoint_path', os.path.join(cfg.train.get('save_dir', 'checkpoints'), 'last.pt'))
    topk = int(cfg.get('inference', {}).get('topk', 10))
    inference_batch_size = int(cfg.get('inference', {}).get('batch_size', 256))
    output_path = cfg.get('inference', {}).get('output_path', 'inference_topk.csv')

    df = load_data(cfg)

    item_ids = df['item'].unique()
    user_ids = df['user'].unique()
    num_item, num_user = len(item_ids), len(user_ids)

    item2idx = pd.Series(data=np.arange(len(item_ids)) + 1, index=item_ids)
    user2idx = pd.Series(data=np.arange(len(user_ids)), index=user_ids)

    df = pd.merge(df, pd.DataFrame({'item': item_ids, 'item_idx': item2idx[item_ids].values}), on='item', how='inner')
    df = pd.merge(df, pd.DataFrame({'user': user_ids, 'user_idx': user2idx[user_ids].values}), on='user', how='inner')
    df.sort_values(['user_idx', 'time'], inplace=True)
    del df['item'], df['user']

    users = defaultdict(list)
    user_train = {}
    for u, i, t in zip(df['user_idx'], df['item_idx'], df['time']):
        users[u].append(i)

    for user in users:
        user_train[user] = users[user][:-1]

    model = BERT4Rec(
        num_user,
        num_item,
        cfg.model.hidden_units,
        cfg.model.num_heads,
        cfg.model.num_layers,
        cfg.model.max_len,
        cfg.model.dropout_rate,
        device,
    )
    model.to(device)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    load_checkpoint(ckpt_path, model, optimizer=None, map_location=device)
    model.eval()

    idx2item = np.empty(num_item + 2, dtype=item_ids.dtype)
    idx2item[:] = 0
    idx2item[1:num_item + 1] = item_ids

    results = []
    mask_token = num_item + 1

    user_indices = np.arange(num_user)
    for start in tqdm(range(0, num_user, inference_batch_size), desc='Inference'):
        end = min(start + inference_batch_size, num_user)
        batch_users = user_indices[start:end]

        seqs = []
        rated_sets = []
        for u in batch_users:
            # Create sequence with mask tokens at specific positions
            seq = user_train[u] + [mask_token]
            
            seq = seq[-cfg.model.max_len:]

            if len(seq) < cfg.model.max_len:
                seq = [0] * (cfg.model.max_len - len(seq)) + seq
            
            seqs.append(seq)
            rated_sets.append(set(user_train[u]))

        seqs = torch.LongTensor(seqs).to(device)

        with torch.no_grad():
            logits = model(seqs)
            
            # 마지막 시점(MASK 위치)의 예측값만 가져오기
            final_logits = logits[:, -1, :] # (batch_size, num_items + 2)

            # 마스킹 처리 (점수를 매우 낮게 설정)
            final_logits[:, 0] = -1e9          # Padding 제외
            final_logits[:, mask_token] = -1e9 # Mask Token 자체 제외
        
            # Mask out already rated items
            for i, rated in enumerate(rated_sets):
                if rated:
                    final_logits[i, list(rated)] = -1e9
            
            _, top_items = torch.topk(final_logits, k=topk, dim=1)
            top_items = top_items.detach().cpu().numpy()

        for row_i, u in enumerate(batch_users):
            user_id = user_ids[u]
            for rank, item_idx in enumerate(top_items[row_i], start=1):
                item_id = idx2item[item_idx]
                results.append((user_id, item_id))

    pred_df = pd.DataFrame(results, columns=['user', 'item'])
    pred_df.to_csv(output_path, index=False)
    print(f'Saved top-{topk} recommendations for {num_user} users to: {output_path}')


if __name__ == "__main__":
    main()
