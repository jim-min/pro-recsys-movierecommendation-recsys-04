import numpy as np
import torch

@torch.no_grad()
def recommend_topk(model, train_mat, item_attr_mat, item_attr_mask, topk, device, user_offset, item_offset, user_batch_size=128):
    model.eval()
    num_users, num_items = train_mat.shape
    all_recs = []

    item_attr_mat = item_attr_mat.to(device) # (I, Attr_len)
    item_attr_mask = item_attr_mask.to(device)

    for u_start in range(0, num_users, user_batch_size):
        u_end = min(u_start + user_batch_size, num_users)
        curr_u_batch = u_end - u_start
        
        # 해당 유저 배치의 모든 아이템에 대한 스코어를 저장할 공간
        user_scores = np.zeros((curr_u_batch, num_items))

        # 메모리 절약을 위해 아이템은 쪼개서 계산
        item_chunk_size = 1024 
        for i_start in range(0, num_items, item_chunk_size):
            i_end = min(i_start + item_chunk_size, num_items)
            curr_i_batch = i_end - i_start

            # 유저와 아이템의 조합 생성 (Broadcast 연산 활용)
            # (curr_u_batch, curr_i_batch) 형태의 조합을 배치로 만듦
            u_indices = torch.arange(u_start, u_end, device=device).view(-1, 1).expand(-1, curr_i_batch) + user_offset
            i_indices = torch.arange(i_start, i_end, device=device).view(1, -1).expand(curr_u_batch, -1) + item_offset
            
            # (Batch, 2 + Max_Attrs) 형태의 feat_idx 구축
            feat_idx = torch.cat([
                u_indices.unsqueeze(-1), 
                i_indices.unsqueeze(-1),
                item_attr_mat[i_start:i_end].unsqueeze(0).expand(curr_u_batch, -1, -1)
            ], dim=-1).reshape(-1, 2 + item_attr_mat.size(1))

            feat_mask = torch.cat([
                torch.ones((curr_u_batch, curr_i_batch, 2), device=device),
                item_attr_mask[i_start:i_end].unsqueeze(0).expand(curr_u_batch, -1, -1)
            ], dim=-1).reshape(-1, 2 + item_attr_mask.size(1))

            scores = model(feat_idx, feat_mask).view(curr_u_batch, curr_i_batch)
            user_scores[:, i_start:i_end] = scores.cpu().numpy()

        # Masking: 이미 본 아이템 제외
        for i, u_idx in enumerate(range(u_start, u_end)):
            seen = train_mat.indices[train_mat.indptr[u_idx] : train_mat.indptr[u_idx + 1]]
            user_scores[i, seen] = -1e9
            
        topk_idx = np.argsort(user_scores, axis=1)[:, -topk:][:, ::-1]
        all_recs.append(topk_idx)

    return np.concatenate(all_recs, axis=0)