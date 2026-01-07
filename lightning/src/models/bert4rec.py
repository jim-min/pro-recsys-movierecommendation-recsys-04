import logging
import math
import numpy as np
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention with Dropout (using Flash Attention)"""

    def __init__(self, head_dim: int, dropout_rate: float):
        super().__init__()
        self.head_dim = head_dim
        self.dropout_rate = dropout_rate

    def forward(self, Q, K, V, mask):
        """
        Args:
            Q, K, V: [batch_size, num_heads, seq_len, head_dim]
            mask: [batch_size, 1, seq_len, seq_len]
        Returns:
            output: [batch_size, num_heads, seq_len, head_dim]
            attn_dist: [batch_size, num_heads, seq_len, seq_len] or None
        """
        # Convert mask to attention mask format for Flash Attention
        # mask: [batch_size, 1, seq_len, seq_len] -> [batch_size, num_heads, seq_len, seq_len]
        batch_size, num_heads, seq_len, head_dim = Q.shape
        attn_mask = mask.expand(batch_size, num_heads, seq_len, seq_len)

        # Convert boolean mask to float mask for scaled_dot_product_attention
        # True (1) -> 0.0, False (0) -> -inf
        attn_mask = torch.where(attn_mask.bool(), 0.0, float("-inf"))

        # Use Flash Attention (PyTorch 2.0+)
        # This is 2-4x faster than manual implementation
        output = F.scaled_dot_product_attention(
            Q,
            K,
            V,
            attn_mask=attn_mask,
            dropout_p=self.dropout_rate if self.training else 0.0,
            is_causal=False,  # BERT4Rec uses bidirectional attention
        )

        # Note: Flash Attention doesn't return attention weights
        # Return None for attn_dist to maintain API compatibility
        return output, None


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention with Residual Connection and Layer Normalization"""

    def __init__(self, num_heads: int, hidden_units: int, dropout_rate: float):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_units = hidden_units

        # hidden_units must be divisible by num_heads
        if hidden_units % num_heads != 0:
            raise ValueError(
                f"hidden_units ({hidden_units}) must be divisible by num_heads ({num_heads})"
            )

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
            random_mask_prob: 0.15
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
        random_mask_prob: float = 0.2,
        last_item_mask_ratio: float = 0.0,
        lr: float = 0.001,
        weight_decay: float = 0.0,
        share_embeddings: bool = True,
        # Metadata parameters
        num_genres: int = 1,
        num_directors: int = 1,
        num_writers: int = 1,
        title_embedding_dim: int = 0,
        use_genre_emb: bool = True,
        use_director_emb: bool = True,
        use_writer_emb: bool = True,
        use_title_emb: bool = True,
        metadata_fusion: str = "concat",
        metadata_dropout: float = 0.1,
    ):
        """
        Args:
            num_items: Number of unique items (excluding padding and mask tokens)
            hidden_units: Hidden dimension size
            num_heads: Number of attention heads
            num_layers: Number of transformer blocks
            max_len: Maximum sequence length
            dropout_rate: Dropout probability
            random_mask_prob: Probability of masking items during random masking (not used in model, kept for compatibility)
            last_item_mask_ratio: Ratio of samples using last item masking (not used in model, kept for compatibility)
            lr: Learning rate
            weight_decay: Weight decay for optimizer
            share_embeddings: Whether to share item embeddings with output layer
            num_genres: Number of unique genres
            num_directors: Number of unique directors
            num_writers: Number of unique writers
            title_embedding_dim: Dimension of pre-computed title embeddings
            use_genre_emb: Whether to use genre embeddings
            use_director_emb: Whether to use director embeddings
            use_writer_emb: Whether to use writer embeddings
            use_title_emb: Whether to use title embeddings
            metadata_fusion: Fusion strategy ('concat', 'add', 'gate')
            metadata_dropout: Dropout rate for metadata embeddings
        """
        super().__init__()
        self.save_hyperparameters()

        self.num_items = num_items
        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_len = max_len
        self.dropout_rate = dropout_rate
        self.random_mask_prob = random_mask_prob
        self.last_item_mask_ratio = last_item_mask_ratio
        self.lr = lr
        self.weight_decay = weight_decay
        self.share_embeddings = share_embeddings
        self.metadata_fusion = metadata_fusion

        # Special tokens: 0=padding, num_items+1=mask
        self.pad_token = 0
        self.mask_token = num_items + 1
        self.num_tokens = num_items + 2

        # Item ID embedding (base)
        self.item_emb = nn.Embedding(
            self.num_tokens, hidden_units, padding_idx=self.pad_token
        )
        self.pos_emb = nn.Embedding(max_len, hidden_units)

        # Metadata embeddings
        self.use_genre_emb = use_genre_emb and num_genres > 1
        self.use_director_emb = use_director_emb and num_directors > 1
        self.use_writer_emb = use_writer_emb and num_writers > 1
        self.use_title_emb = use_title_emb and title_embedding_dim > 0

        if self.use_genre_emb:
            self.genre_emb = nn.Embedding(num_genres, hidden_units, padding_idx=0)
            log.info(f"Genre embedding enabled: {num_genres} genres -> {hidden_units}D")

        if self.use_director_emb:
            self.director_emb = nn.Embedding(num_directors, hidden_units, padding_idx=0)
            log.info(
                f"Director embedding enabled: {num_directors} directors -> {hidden_units}D"
            )

        if self.use_writer_emb:
            self.writer_emb = nn.Embedding(num_writers, hidden_units, padding_idx=0)
            log.info(
                f"Writer embedding enabled: {num_writers} writers -> {hidden_units}D"
            )

        if self.use_title_emb:
            self.title_projection = nn.Linear(title_embedding_dim, hidden_units)
            log.info(
                f"Title embedding enabled: {title_embedding_dim}D -> {hidden_units}D"
            )

        # Metadata fusion layer
        num_features = 1  # item_emb
        if self.use_genre_emb:
            num_features += 1
        if self.use_director_emb:
            num_features += 1
        if self.use_writer_emb:
            num_features += 1
        if self.use_title_emb:
            num_features += 1

        if metadata_fusion == "concat":
            # Concatenate all embeddings and project back to hidden_units
            self.fusion_layer = nn.Linear(hidden_units * num_features, hidden_units)
            log.info(f"Fusion: concat {num_features} features -> projection")
        elif metadata_fusion == "add":
            # Simple weighted addition (no extra params)
            self.fusion_layer = None
            log.info(f"Fusion: weighted addition of {num_features} features")
        elif metadata_fusion == "gate":
            # Gated fusion (learnable weights per feature)
            self.fusion_gate = nn.Linear(hidden_units * num_features, num_features)
            self.fusion_layer = None
            log.info(f"Fusion: gated mechanism with {num_features} features")
        else:
            raise ValueError(
                f"Unknown metadata_fusion: {metadata_fusion}. Choose 'concat', 'add', or 'gate'"
            )

        self.metadata_dropout = nn.Dropout(metadata_dropout)

        # Input processing
        self.dropout = nn.Dropout(dropout_rate)
        self.emb_layernorm = nn.LayerNorm(hidden_units, eps=1e-6)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                BERT4RecBlock(num_heads, hidden_units, dropout_rate)
                for _ in range(num_layers)
            ]
        )

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

    def forward(self, log_seqs, metadata=None, return_gate_values=False):
        """
        Args:
            log_seqs: [batch_size, seq_len] - Input sequences (can be on any device)
            metadata: Dict with keys:
                - 'genres': [batch_size, seq_len, max_genres] - Genre indices (padded)
                - 'directors': [batch_size, seq_len] - Director indices
                - 'writers': [batch_size, seq_len, max_writers] - Writer indices (padded)
                - 'title_embs': [batch_size, seq_len, title_emb_dim] - Pre-computed title embeddings
            return_gate_values: If True, return gate values along with logits (only for gate fusion)
        Returns:
            logits: [batch_size, seq_len, num_tokens] - Output logits
            gate_values: [batch_size, seq_len, num_features] - Gate values (if return_gate_values=True and using gate fusion)
        """
        # Convert to LongTensor and move to model's device
        if not isinstance(log_seqs, torch.Tensor):
            log_seqs = torch.LongTensor(log_seqs)
        log_seqs = log_seqs.to(self.device)

        batch_size, seq_len = log_seqs.shape

        # Item ID embeddings
        item_embs = self.item_emb(log_seqs)  # [batch, seq_len, hidden]

        # Collect embeddings to fuse
        embeddings_to_fuse = [item_embs]

        # Genre embeddings (average pooling over multiple genres)
        if self.use_genre_emb and metadata is not None and "genres" in metadata:
            genre_indices = metadata["genres"].to(
                self.device
            )  # [batch, seq_len, max_genres]
            genre_embs = self.genre_emb(
                genre_indices
            )  # [batch, seq_len, max_genres, hidden]
            # Average pooling (ignore padding=0)
            genre_mask = (
                (genre_indices != 0).float().unsqueeze(-1)
            )  # [batch, seq_len, max_genres, 1]
            genre_embs_avg = (genre_embs * genre_mask).sum(dim=2) / (
                genre_mask.sum(dim=2) + 1e-9
            )
            embeddings_to_fuse.append(genre_embs_avg)

        # Director embeddings
        if self.use_director_emb and metadata is not None and "directors" in metadata:
            director_indices = metadata["directors"].to(self.device)  # [batch, seq_len]
            director_embs = self.director_emb(
                director_indices
            )  # [batch, seq_len, hidden]
            embeddings_to_fuse.append(director_embs)

        # Writer embeddings (average pooling)
        if self.use_writer_emb and metadata is not None and "writers" in metadata:
            writer_indices = metadata["writers"].to(
                self.device
            )  # [batch, seq_len, max_writers]
            writer_embs = self.writer_emb(
                writer_indices
            )  # [batch, seq_len, max_writers, hidden]
            writer_mask = (writer_indices != 0).float().unsqueeze(-1)
            writer_embs_avg = (writer_embs * writer_mask).sum(dim=2) / (
                writer_mask.sum(dim=2) + 1e-9
            )
            embeddings_to_fuse.append(writer_embs_avg)

        # Title embeddings (pre-computed, just project)
        if self.use_title_emb and metadata is not None and "title_embs" in metadata:
            title_embs_raw = metadata["title_embs"].to(
                self.device
            )  # [batch, seq_len, title_dim]
            title_embs = self.title_projection(
                title_embs_raw
            )  # [batch, seq_len, hidden]
            embeddings_to_fuse.append(title_embs)

        # Fusion
        gate_values = None
        if self.metadata_fusion == "concat":
            fused_embs = torch.cat(
                embeddings_to_fuse, dim=-1
            )  # [batch, seq_len, hidden*N]
            seqs = self.fusion_layer(fused_embs)  # [batch, seq_len, hidden]
        elif self.metadata_fusion == "add":
            seqs = sum(embeddings_to_fuse) / len(embeddings_to_fuse)
        elif self.metadata_fusion == "gate":
            stacked = torch.cat(
                embeddings_to_fuse, dim=-1
            )  # [batch, seq_len, hidden*N]
            gates = torch.softmax(
                self.fusion_gate(stacked), dim=-1
            )  # [batch, seq_len, N]
            if return_gate_values:
                gate_values = gates  # [batch, seq_len, N]
            gates = gates.unsqueeze(-1)  # [batch, seq_len, N, 1]
            stacked_embs = torch.stack(
                embeddings_to_fuse, dim=2
            )  # [batch, seq_len, N, hidden]
            seqs = (gates * stacked_embs).sum(dim=2)  # [batch, seq_len, hidden]

        seqs = self.metadata_dropout(seqs)

        # Positional embeddings
        positions = (
            torch.arange(seq_len, device=self.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
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
            logits = torch.matmul(
                seqs, self.item_emb.weight.T
            )  # [batch, seq_len, num_tokens]
        else:
            logits = self.out(seqs)

        if return_gate_values and gate_values is not None:
            return logits, gate_values
        return logits

    def training_step(self, batch, batch_idx):
        """
        Training step for Lightning

        Args:
            batch: (sequences, labels, metadata) from DataLoader
            batch_idx: Batch index
        Returns:
            loss: Training loss
        """
        # Unpack batch (with metadata)
        if len(batch) == 3:
            log_seqs, labels, metadata = batch
        else:
            # Backward compatibility (without metadata)
            log_seqs, labels = batch
            metadata = None

        # Forward pass
        logits = self(log_seqs, metadata)  # [batch, seq_len, num_tokens]

        # Flatten for loss computation
        logits = logits.view(-1, self.num_tokens)  # [batch*seq_len, num_tokens]
        labels = labels.view(-1)  # [batch*seq_len]

        # Compute loss
        loss = self.criterion(logits, labels)

        # Logging
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for Lightning

        Args:
            batch: (sequences, labels, metadata, ground_truth_items) from DataLoader
            batch_idx: Batch index
        Returns:
            Dictionary with metrics
        """
        # Unpack batch (with metadata)
        if len(batch) == 4:
            log_seqs, _, metadata, target_items = batch
        else:
            # Backward compatibility (without metadata)
            log_seqs, _, target_items = batch
            metadata = None

        # Forward pass (with gate values if using gate fusion)
        if self.metadata_fusion == "gate":
            logits, gate_values = self(log_seqs, metadata, return_gate_values=True)
        else:
            logits = self(log_seqs, metadata)
            gate_values = None

        # Get predictions for last position
        scores = logits[:, -1, :]  # [batch, num_tokens]

        # Mask out padding and mask token
        # Float16 compatible: -1e4 instead of -1e9
        scores[:, self.pad_token] = -1e4
        scores[:, self.mask_token] = -1e4

        # Get top-k predictions
        _, top_items = torch.topk(scores, k=10, dim=1)  # [batch, 10]

        # Compute metrics using GPU batch processing (10-50x faster than Python loop)
        batch_size = target_items.size(0)

        # Reshape target for broadcasting: [batch, 1]
        target_items_expanded = target_items.view(-1, 1)

        # Check if target is in top-k: [batch, 10] -> [batch]
        hits = (top_items == target_items_expanded).any(dim=1)
        val_hit_10 = hits.sum().item()

        # Calculate NDCG for hits only
        # Find positions where predictions match targets
        matches = top_items == target_items_expanded  # [batch, 10]
        ranks = matches.float().argmax(dim=1)  # [batch] - position of match (0-9)

        # Calculate NDCG only for hits
        ndcg_values = torch.where(
            hits,
            1.0 / torch.log2(ranks.float() + 2),  # +2 because rank starts at 0
            torch.zeros_like(ranks.float()),
        )
        val_ndcg_10 = ndcg_values.sum().item()

        # Logging
        self.log(
            "val_hit@10",
            val_hit_10 / batch_size,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "val_ndcg@10",
            val_ndcg_10 / batch_size,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        # Log gate values if using gate fusion
        if gate_values is not None:
            # Average gate values across batch and sequence length
            # gate_values: [batch, seq_len, num_features]
            avg_gates = gate_values.mean(dim=[0, 1])  # [num_features]

            # Feature names based on enabled features
            feature_names = ["item"]
            if self.use_genre_emb:
                feature_names.append("genre")
            if self.use_director_emb:
                feature_names.append("director")
            if self.use_writer_emb:
                feature_names.append("writer")
            if self.use_title_emb:
                feature_names.append("title")

            # Log each feature's importance
            for i, name in enumerate(feature_names):
                self.log(
                    f"val_gate/{name}",
                    avg_gates[i].item(),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                )

            # Store final gate values (will be printed after training)
            self._final_gate_values = (feature_names, avg_gates.detach().cpu())

        return {
            "val_hit@10": val_hit_10,
            "val_ndcg@10": val_ndcg_10,
            "batch_size": batch_size,
        }

    def configure_optimizers(self):
        """Configure optimizer for Lightning"""
        # We train the model using Adam [24] with learning
        # rate of 1e-4, β1 = 0.9, β2 = 0.999, ℓ2 weight decay of 0.01, and
        # linear decay of the learning rate.
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=self.lr / 10.0,
        )
        return [optimizer], [scheduler]

    def _prepare_batch_metadata(self, seqs, item_metadata):
        """
        Prepare metadata dict for a batch of sequences

        Args:
            seqs: np.array or torch.Tensor [batch_size, seq_len] - Item sequences
            item_metadata: Dict with metadata mappings from DataModule
                {
                    'genres': Dict[item_idx, List[genre_idx]],
                    'directors': Dict[item_idx, director_idx],
                    'writers': Dict[item_idx, List[writer_idx]],
                    'title_embs': Dict[item_idx, np.array]
                }
        Returns:
            metadata: Dict with batched metadata tensors
        """
        if item_metadata is None:
            return None

        # Convert to numpy if needed
        if isinstance(seqs, torch.Tensor):
            seqs = seqs.cpu().numpy()

        batch_size, seq_len = seqs.shape
        metadata = {}
        max_genres = 5
        max_writers = 5

        # Genres (multi-hot, padded)
        if "genres" in item_metadata:
            item_genres = item_metadata["genres"]
            genre_batch = []
            for seq in seqs:
                seq_genres = []
                for item_idx in seq:
                    if item_idx in item_genres:
                        genres = item_genres[item_idx][:max_genres]
                        genres = genres + [0] * (max_genres - len(genres))
                    else:
                        genres = [0] * max_genres
                    seq_genres.append(genres)
                genre_batch.append(seq_genres)
            metadata["genres"] = torch.LongTensor(genre_batch).to(
                self.device
            )  # [batch, seq_len, max_genres]

        # Directors (single value per item)
        if "directors" in item_metadata:
            item_directors = item_metadata["directors"]
            director_batch = []
            for seq in seqs:
                seq_directors = [
                    item_directors.get(int(item_idx), 0) for item_idx in seq
                ]
                director_batch.append(seq_directors)
            metadata["directors"] = torch.LongTensor(director_batch).to(
                self.device
            )  # [batch, seq_len]

        # Writers (multi-hot, padded)
        if "writers" in item_metadata:
            item_writers = item_metadata["writers"]
            writer_batch = []
            for seq in seqs:
                seq_writers = []
                for item_idx in seq:
                    if item_idx in item_writers:
                        writers = item_writers[item_idx][:max_writers]
                        writers = writers + [0] * (max_writers - len(writers))
                    else:
                        writers = [0] * max_writers
                    seq_writers.append(writers)
                writer_batch.append(seq_writers)
            metadata["writers"] = torch.LongTensor(writer_batch).to(
                self.device
            )  # [batch, seq_len, max_writers]

        # Title embeddings (pre-computed)
        if "title_embs" in item_metadata:
            item_title_embs = item_metadata["title_embs"]
            # Get embedding dimension from first available embedding
            title_dim = None
            for emb in item_title_embs.values():
                title_dim = len(emb)
                break

            if title_dim is not None:
                title_batch = []
                for seq in seqs:
                    seq_titles = []
                    for item_idx in seq:
                        if item_idx in item_title_embs:
                            seq_titles.append(item_title_embs[item_idx])
                        else:
                            seq_titles.append(np.zeros(title_dim))
                    title_batch.append(seq_titles)
                metadata["title_embs"] = torch.FloatTensor(np.array(title_batch)).to(
                    self.device
                )  # [batch, seq_len, title_dim]

        return metadata

    def predict(self, user_sequences, topk=10, exclude_items=None, item_metadata=None):
        """
        Generate top-k recommendations for given sequences

        Args:
            user_sequences: List of sequences [[item1, item2, ...], ...]
            topk: Number of items to recommend
            exclude_items: List of sets of items to exclude per user
            item_metadata: Dict with metadata mappings (genres, directors, writers, title_embs)
        Returns:
            recommendations: List of lists of top-k item IDs
        """
        self.eval()

        with torch.no_grad():
            # Prepare sequences (add mask token at the end)
            seqs = []
            for seq in user_sequences:
                masked_seq = (list(seq) + [self.mask_token])[-self.max_len :]
                if len(masked_seq) < self.max_len:
                    masked_seq = [self.pad_token] * (
                        self.max_len - len(masked_seq)
                    ) + masked_seq
                seqs.append(masked_seq)

            seqs_array = np.array(seqs, dtype=np.int64)

            # Prepare metadata for batch (if provided)
            metadata = None
            if item_metadata is not None:
                metadata = self._prepare_batch_metadata(seqs_array, item_metadata)

            # Forward pass WITH metadata
            logits = self(seqs_array, metadata)
            scores = logits[:, -1, :]  # Last position

            # Mask invalid items
            # Float16 compatible: -1e4 instead of -1e9
            scores[:, self.pad_token] = -1e4
            scores[:, self.mask_token] = -1e4

            # Exclude already seen items
            if exclude_items is not None:
                for i, items in enumerate(exclude_items):
                    if items:
                        scores[i, list(items)] = -1e4

            # Get top-k (cap topk to number of available items)
            max_k = scores.size(1)  # Total number of items (including special tokens)
            actual_k = min(topk, max_k)
            _, top_items = torch.topk(scores, k=actual_k, dim=1)

            return top_items.cpu().numpy()
