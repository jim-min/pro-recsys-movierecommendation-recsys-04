import torch
import torch.nn as nn


class S3RecPretrain(nn.Module):
    """
    S3Rec의 pretraining을 "현실적으로 쓸모있는 핵심(MIP)"만 구현한 버전.
    목적: item embedding + transformer representation을 안정적으로 학습한 뒤,
         item embedding을 BERT4Rec 초기화로 넘겨주기.

    Token:
      PAD = 0
      item tokens = 1..num_items
      MASK = num_items + 1
    """

    def __init__(
        self,
        num_items: int,
        max_len: int = 50,
        hidden_size: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_items = num_items
        self.max_len = max_len

        self.PAD = 0
        self.MASK = num_items + 1
        self.vocab_size = num_items + 2

        # (RecBole 계열 이름과 비슷하게)
        self.item_embedding = nn.Embedding(self.vocab_size, hidden_size, padding_idx=self.PAD)
        self.position_embedding = nn.Embedding(max_len, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(hidden_size)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.trm_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # MIP head
        self.mip_head = nn.Linear(hidden_size, self.vocab_size)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.mip_head.weight)
        nn.init.zeros_(self.mip_head.bias)

    def forward(self, input_ids: torch.Tensor, pad_mask: torch.Tensor):
        """
        input_ids: (batch_size, seq_len)
        pad_mask:  (batch_size, seq_len) True where PAD
        """
        bsz, seq_len = input_ids.size()
        pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, seq_len)

        x = self.item_embedding(input_ids) + self.position_embedding(pos)
        x = self.LayerNorm(self.dropout(x))
        h = self.trm_encoder(x, src_key_padding_mask=pad_mask)
        logits = self.mip_head(h)
        return logits

    @torch.no_grad()
    def export_item_embedding(self, path: str):
        """
        BERT4Rec이 받아먹을 "아이템 임베딩"만 저장.
        - 저장 shape: (num_items, hidden_size)
        - token 1..num_items만 추출 (0=PAD, num_items+1=MASK 제외)
        """
        emb = self.item_embedding.weight[1:self.num_items + 1].detach().cpu()
        torch.save(emb, path)
        return path
