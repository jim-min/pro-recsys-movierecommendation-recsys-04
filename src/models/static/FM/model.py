import torch
import torch.nn as nn


class FactorizationMachine(nn.Module):
    """
    FM with sparse categorical features.
    Feature index 0 is padding.
    """

    def __init__(self, num_features: int, embed_dim: int):
        super().__init__()
        self.num_features = num_features
        self.embed_dim = embed_dim

        self.linear = nn.Embedding(num_features, 1, padding_idx=0)
        self.v = nn.Embedding(num_features, embed_dim, padding_idx=0)

        nn.init.normal_(self.linear.weight, std=0.01)
        nn.init.normal_(self.v.weight, std=0.01)

    def forward(self, feat_idx, feat_mask):
        """
        feat_idx : (B, F)
        feat_mask: (B, F)
        """
        # linear term
        linear = self.linear(feat_idx).squeeze(-1)
        linear = (linear * feat_mask).sum(dim=1)

        # interaction term
        emb = self.v(feat_idx) * feat_mask.unsqueeze(-1)
        sum_v = emb.sum(dim=1)
        inter = 0.5 * ((sum_v * sum_v) - (emb * emb).sum(dim=1)).sum(dim=1)

        return linear + inter
