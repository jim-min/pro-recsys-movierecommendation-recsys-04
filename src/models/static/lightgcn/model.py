import torch
import torch.nn as nn


class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embed_dim, num_layers):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_layers = num_layers

        self.user_emb = nn.Embedding(num_users, embed_dim)
        self.item_emb = nn.Embedding(num_items, embed_dim)

        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)

    def forward(self, norm_adj):
        all_emb = torch.cat(
            [self.user_emb.weight, self.item_emb.weight], dim=0
        )
        embs = [all_emb]

        x = all_emb
        for _ in range(self.num_layers):
            x = torch.sparse.mm(norm_adj, x)
            embs.append(x)

        out = torch.stack(embs, dim=0).mean(dim=0)
        return out[: self.num_users], out[self.num_users :]
