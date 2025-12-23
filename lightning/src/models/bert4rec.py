import logging
import math
import numpy as np
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention with Dropout"""

    def __init__(self, head_dim: int, dropout_rate: float):
        super().__init__()
        self.head_dim = head_dim
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, Q, K, V, mask):
        """
        Args:
            Q, K, V: [batch_size, num_heads, seq_len, head_dim]
            mask: [batch_size, 1, seq_len, seq_len]
        Returns:
            output: [batch_size, num_heads, seq_len, head_dim]
            attn_dist: [batch_size, num_heads, seq_len, seq_len]
        """
        # Scaled dot-product: Q @ K^T / sqrt(d_k)
        attn_score = torch.matmul(Q, K.transpose(2, 3)) / math.sqrt(self.head_dim)

        # Mask 적용 (padding 위치는 -inf로)
        attn_score = attn_score.masked_fill(mask == 0, -1e9)

        # Softmax + Dropout
        attn_dist = self.dropout(F.softmax(attn_score, dim=-1))

        # Attention 적용
        output = torch.matmul(attn_dist, V)

        return output, attn_dist


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention with Residual Connection and Layer Normalization"""

    def __init__(self, num_heads: int, hidden_units: int, dropout_rate: float):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_units = hidden_units

        # hidden_units must be divisible by num_heads
        if hidden_units % num_heads != 0:
            raise ValueError(f"hidden_units ({hidden_units}) must be divisible by num_heads ({num_heads})")

        self.head_dim = hidden_units // num_heads

        # Linear projections for Q, K, V, O
        self.W_Q = nn.Linear(hidden_units, hidden_units, bias=False)
        self.W_K = nn.Linear(hidden_units, hidden_units, bias=False)
        self.W_V = nn.Linear(hidden_units, hidden_units, bias=False)
        self.W_O = nn.Linear(hidden_units, hidden_units, bias=False)

        # Attention module
        self.attention = ScaledDotProductAttention(self.head_dim, dropout_rate)

        # Dropout and LayerNorm
        self.dropout = nn.Dropout(dropout_rate)
        self.layerNorm = nn.LayerNorm(hidden_units, eps=1e-6)

    def forward(self, enc, mask):
        """
        Args:
            enc: [batch_size, seq_len, hidden_units]
            mask: [batch_size, 1, seq_len, seq_len]
        Returns:
            output: [batch_size, seq_len, hidden_units]
            attn_dist: [batch_size, num_heads, seq_len, seq_len]
        """
        residual = enc
        batch_size, seq_len = enc.size(0), enc.size(1)

        # Q, K, V projection and split into multiple heads
        Q = self.W_Q(enc).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.W_K(enc).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.W_V(enc).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose for attention: [batch, num_heads, seq_len, head_dim]
        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)

        # Apply attention
        output, attn_dist = self.attention(Q, K, V, mask)

        # Concatenate heads
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.hidden_units)

        # Output projection + Dropout + Residual + LayerNorm
        output = self.layerNorm(self.dropout(self.W_O(output)) + residual)

        return output, attn_dist


class PositionwiseFeedForward(nn.Module):
    """Position-wise Feed-Forward Network with GELU activation"""

    def __init__(self, hidden_units: int, dropout_rate: float):
        super().__init__()

        # FFN with 4x expansion (BERT4Rec paper)
        self.W_1 = nn.Linear(hidden_units, 4 * hidden_units)
        self.W_2 = nn.Linear(4 * hidden_units, hidden_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.layerNorm = nn.LayerNorm(hidden_units, eps=1e-6)

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, hidden_units]
        Returns:
            output: [batch_size, seq_len, hidden_units]
        """
        residual = x

        # FFN: W_2(GELU(W_1(x)))
        output = self.W_2(F.gelu(self.dropout(self.W_1(x))))

        # Dropout + Residual + LayerNorm
        output = self.layerNorm(self.dropout(output) + residual)

        return output


class BERT4RecBlock(nn.Module):
    """Transformer Block: Multi-Head Attention + Feed-Forward"""

    def __init__(self, num_heads: int, hidden_units: int, dropout_rate: float):
        super().__init__()
        self.attention = MultiHeadAttention(num_heads, hidden_units, dropout_rate)
        self.pointwise_feedforward = PositionwiseFeedForward(hidden_units, dropout_rate)

    def forward(self, input_enc, mask):
        """
        Args:
            input_enc: [batch_size, seq_len, hidden_units]
            mask: [batch_size, 1, seq_len, seq_len]
        Returns:
            output_enc: [batch_size, seq_len, hidden_units]
            attn_dist: [batch_size, num_heads, seq_len, seq_len]
        """
        output_enc, attn_dist = self.attention(input_enc, mask)
        output_enc = self.pointwise_feedforward(output_enc)
        return output_enc, attn_dist


class BERT4Rec(L.LightningModule):
    """
    BERT4Rec: Bidirectional Encoder Representations from Transformers for Sequential Recommendation

    Paper: https://arxiv.org/abs/1904.06690

    Key Features:
    - Bidirectional self-attention (unlike unidirectional models like SASRec)
    - Cloze task training (masked item prediction)
    - Learnable positional embeddings
    - Output embedding weight sharing for efficiency

    Hydra 설정 사용:
        model:
            hidden_units: 64
            num_heads: 4
            num_layers: 3
            max_len: 50
            dropout_rate: 0.3
            mask_prob: 0.15
        training:
            lr: 0.001
            weight_decay: 0.0
    """

    def __init__(
        self,
        num_items: int,
        hidden_units: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        max_len: int = 50,
        dropout_rate: float = 0.3,
        mask_prob: float = 0.15,
        lr: float = 0.001,
        weight_decay: float = 0.0,
        share_embeddings: bool = True,
    ):
        """
        Args:
            num_items: Number of unique items (excluding padding and mask tokens)
            hidden_units: Hidden dimension size
            num_heads: Number of attention heads
            num_layers: Number of transformer blocks
            max_len: Maximum sequence length
            dropout_rate: Dropout probability
            mask_prob: Probability of masking items during training
            lr: Learning rate
            weight_decay: Weight decay for optimizer
            share_embeddings: Whether to share item embeddings with output layer
        """
        super().__init__()
        self.save_hyperparameters()

        self.num_items = num_items
        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_len = max_len
        self.dropout_rate = dropout_rate
        self.mask_prob = mask_prob
        self.lr = lr
        self.weight_decay = weight_decay
        self.share_embeddings = share_embeddings

        # Special tokens: 0=padding, num_items+1=mask
        self.pad_token = 0
        self.mask_token = num_items + 1
        self.num_tokens = num_items + 2

        # Embeddings
        self.item_emb = nn.Embedding(
            self.num_tokens, hidden_units, padding_idx=self.pad_token
        )
        self.pos_emb = nn.Embedding(max_len, hidden_units)

        # Input processing
        self.dropout = nn.Dropout(dropout_rate)
        self.emb_layernorm = nn.LayerNorm(hidden_units, eps=1e-6)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            BERT4RecBlock(num_heads, hidden_units, dropout_rate)
            for _ in range(num_layers)
        ])

        # Output layer
        if share_embeddings:
            # Share weights with item embedding (more efficient)
            self.out = None
        else:
            self.out = nn.Linear(hidden_units, self.num_tokens, bias=False)

        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_token)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    nn.init.zeros_(module.weight[module.padding_idx])
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, log_seqs):
        """
        Args:
            log_seqs: [batch_size, seq_len] - Input sequences (can be on any device)
        Returns:
            logits: [batch_size, seq_len, num_tokens] - Output logits
        """
        # Convert to LongTensor and move to model's device
        if not isinstance(log_seqs, torch.Tensor):
            log_seqs = torch.LongTensor(log_seqs)
        log_seqs = log_seqs.to(self.device)

        batch_size, seq_len = log_seqs.shape

        # Item embeddings
        seqs = self.item_emb(log_seqs)  # [batch, seq_len, hidden]

        # Positional embeddings
        positions = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)
        seqs += self.pos_emb(positions)

        # Dropout + LayerNorm
        seqs = self.emb_layernorm(self.dropout(seqs))

        # Attention mask: [batch, 1, seq_len, seq_len]
        # Bidirectional attention (only mask padding)
        mask = (log_seqs > 0).unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
        mask = mask.expand(-1, -1, seq_len, -1)  # [batch, 1, seq_len, seq_len]

        # Apply transformer blocks
        for block in self.blocks:
            seqs, _ = block(seqs, mask)

        # Output projection
        if self.share_embeddings:
            # Use item embedding weights transposed
            logits = torch.matmul(seqs, self.item_emb.weight.T)  # [batch, seq_len, num_tokens]
        else:
            logits = self.out(seqs)

        return logits

    def mask_sequence(self, seq):
        """
        Apply BERT-style masking to a sequence

        Args:
            seq: [seq_len] - Original sequence
        Returns:
            tokens: [seq_len] - Masked sequence
            labels: [seq_len] - Labels (0 for non-masked, original item for masked)
        """
        tokens = []
        labels = []

        for item in seq:
            prob = np.random.random()

            if prob < self.mask_prob:
                # Masked position
                prob /= self.mask_prob

                if prob < 0.8:
                    # 80%: Replace with [MASK]
                    tokens.append(self.mask_token)
                elif prob < 0.9:
                    # 10%: Replace with random item
                    tokens.append(np.random.randint(1, self.num_items + 1))
                else:
                    # 10%: Keep original
                    tokens.append(item)

                labels.append(item)  # Original item as label
            else:
                # Not masked
                tokens.append(item)
                labels.append(self.pad_token)  # Ignore in loss

        return tokens, labels

    def training_step(self, batch, batch_idx):
        """
        Training step for Lightning

        Args:
            batch: (sequences, labels) from DataLoader
            batch_idx: Batch index
        Returns:
            loss: Training loss
        """
        log_seqs, labels = batch

        # Forward pass
        logits = self(log_seqs)  # [batch, seq_len, num_tokens]

        # Flatten for loss computation
        logits = logits.view(-1, self.num_tokens)  # [batch*seq_len, num_tokens]
        labels = labels.view(-1)  # [batch*seq_len]

        # Compute loss
        loss = self.criterion(logits, labels)

        # Logging
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for Lightning

        Args:
            batch: (sequences, labels, ground_truth_items) from DataLoader
            batch_idx: Batch index
        Returns:
            Dictionary with metrics
        """
        log_seqs, _, target_items = batch

        # Forward pass
        logits = self(log_seqs)  # [batch, seq_len, num_tokens]

        # Get predictions for last position
        scores = logits[:, -1, :]  # [batch, num_tokens]

        # Mask out padding and mask token
        scores[:, self.pad_token] = -1e9
        scores[:, self.mask_token] = -1e9

        # Get top-k predictions
        _, top_items = torch.topk(scores, k=10, dim=1)  # [batch, 10]

        # Compute metrics
        batch_size = target_items.size(0)
        hit_10 = 0
        ndcg_10 = 0

        for i in range(batch_size):
            target = target_items[i].item()
            predictions = top_items[i].cpu().numpy()

            if target in predictions:
                rank = np.where(predictions == target)[0][0]
                hit_10 += 1
                ndcg_10 += 1 / np.log2(rank + 2)

        # Logging
        self.log('val_hit@10', hit_10 / batch_size, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_ndcg@10', ndcg_10 / batch_size, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return {'hit@10': hit_10, 'ndcg@10': ndcg_10, 'batch_size': batch_size}

    def configure_optimizers(self):
        """Configure optimizer for Lightning"""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        return optimizer

    def predict(self, user_sequences, topk=10, exclude_items=None):
        """
        Generate top-k recommendations for given sequences

        Args:
            user_sequences: List of sequences [[item1, item2, ...], ...]
            topk: Number of items to recommend
            exclude_items: List of sets of items to exclude per user
        Returns:
            recommendations: List of lists of top-k item IDs
        """
        self.eval()

        with torch.no_grad():
            # Prepare sequences (add mask token at the end)
            seqs = []
            for seq in user_sequences:
                masked_seq = (list(seq) + [self.mask_token])[-self.max_len:]
                if len(masked_seq) < self.max_len:
                    masked_seq = [self.pad_token] * (self.max_len - len(masked_seq)) + masked_seq
                seqs.append(masked_seq)

            seqs = np.array(seqs, dtype=np.int64)

            # Forward pass
            logits = self(seqs)
            scores = logits[:, -1, :]  # Last position

            # Mask invalid items
            scores[:, self.pad_token] = -1e9
            scores[:, self.mask_token] = -1e9

            # Exclude already seen items
            if exclude_items is not None:
                for i, items in enumerate(exclude_items):
                    if items:
                        scores[i, list(items)] = -1e9

            # Get top-k
            _, top_items = torch.topk(scores, k=topk, dim=1)

            return top_items.cpu().numpy()
