import hydra
from omegaconf import DictConfig

import torch
import torch.nn as nn
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader
from model import BERT4Rec, SeqDataset, random_neg, save_checkpoint, load_checkpoint, load_data

@hydra.main(version_base=None, config_path="../../config/BERT4rec", config_name="config")
def main(cfg: DictConfig):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("현재 사용 디바이스:", device)

    df = load_data(cfg)

    item_ids = df['item'].unique()
    user_ids = df['user'].unique()
    num_item, num_user = len(item_ids), len(user_ids)
    num_batch = num_user // cfg.train.batch_size

    # user, item indexing
    item2idx = pd.Series(data=np.arange(len(item_ids))+1, index=item_ids) # item re-indexing (1~num_item), num_item+1: mask idx
    user2idx = pd.Series(data=np.arange(len(user_ids)), index=user_ids) # user re-indexing (0~num_user-1)

    # dataframe indexing
    df = pd.merge(df, pd.DataFrame({'item': item_ids, 'item_idx': item2idx[item_ids].values}), on='item', how='inner')
    df = pd.merge(df, pd.DataFrame({'user': user_ids, 'user_idx': user2idx[user_ids].values}), on='user', how='inner')
    df.sort_values(['user_idx', 'time'], inplace=True)
    del df['item'], df['user']

    # train set, valid set 생성
    users = defaultdict(list) # defaultdict은 dictionary의 key가 없을때 default 값을 value로 반환
    user_train = {}
    user_valid = {}
    for u, i, t in zip(df['user_idx'], df['item_idx'], df['time']):
        users[u].append(i)

    for user in users:
        user_train[user] = users[user][:-1]
        user_valid[user] = [users[user][-1]]

    print(f'num users: {num_user}, num items: {num_item}')

    model = BERT4Rec(num_user, num_item, cfg.model.hidden_units, cfg.model.num_heads, cfg.model.num_layers, cfg.model.max_len, cfg.model.dropout_rate, device)
    model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0) # label이 0인 경우 무시
    seq_dataset = SeqDataset(user_train, num_user, num_item, cfg.model.max_len, cfg.train.mask_prob)
    data_loader = DataLoader(seq_dataset, batch_size=cfg.train.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)

    save_dir = cfg.train.get('save_dir', 'checkpoints')
    save_every = cfg.train.get('save_every', 10)
    resume_path = cfg.train.get('resume_path', None)
    os.makedirs(save_dir, exist_ok=True)

    start_epoch = 1
    if resume_path:
        ckpt = load_checkpoint(resume_path, model, optimizer=optimizer, map_location=device)
        start_epoch = int(ckpt.get('epoch', 0)) + 1

    for epoch in range(start_epoch, cfg.train.num_epochs + 1):
        tbar = tqdm(data_loader)
        for step, (log_seqs, labels) in enumerate(tbar):
            if (log_seqs < 0).any() or (log_seqs > (num_item + 1)).any():
                print(f"Invalid token index in batch! Min: {log_seqs.min()}, Max: {log_seqs.max()}")
                continue
            logits = model(log_seqs)

            # size matching
            logits = logits.view(-1, logits.size(-1))
            labels = labels.view(-1).to(device)

            optimizer.zero_grad()
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            tbar.set_description(f'Epoch: {epoch:3d}| Step: {step:3d}| Train loss: {loss:.5f}')

        

        if save_every and (epoch % int(save_every) == 0):
            model.eval()
            NDCG = 0.0  # NDCG@10
            HIT = 0.0  # HIT@10
            HIT_2 = 0.0  # HIT@2

            num_item_sample = 100
            num_user_sample = min(1000, num_user)

            sampled_users = np.random.randint(0, num_user, num_user_sample)
            for u in sampled_users:
                seq = (user_train[u] + [num_item + 1])[-cfg.model.max_len:]
                if len(seq) < cfg.model.max_len:
                    seq = [0] * (cfg.model.max_len - len(seq)) + seq

                rated = set(user_train[u] + user_valid[u])
                item_idx = [user_valid[u][0]] + [random_neg(1, num_item + 1, rated) for _ in range(num_item_sample)]

                with torch.no_grad():
                    predictions = -model(np.array([seq]))
                    predictions = predictions[0][-1][item_idx]
                    rank = predictions.argsort().argsort()[0].item()

                if rank < 10:
                    NDCG += 1 / np.log2(rank + 2)
                    HIT += 1

                    if rank < 2:
                        HIT_2 += 1

            print(f'Valid NDCG@10: {NDCG/num_user_sample}| HIT@10: {HIT/num_user_sample}| HIT@2: {HIT_2/num_user_sample}')
            model.train()
            extra = {
                'num_user': num_user,
                'num_item': num_item,
                'max_len': int(cfg.model.max_len),
            }
            save_checkpoint(os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pt'), model, optimizer, epoch, extra=extra)
            save_checkpoint(os.path.join(save_dir, 'last.pt'), model, optimizer, epoch, extra=extra)

if __name__ == "__main__":
    main()