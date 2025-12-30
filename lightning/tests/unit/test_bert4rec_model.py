"""Unit tests for BERT4Rec model"""

import pytest
import torch
from src.models.bert4rec import BERT4Rec


@pytest.mark.unit
class TestBERT4RecModelInitialization:
    """Test BERT4Rec model initialization"""

    def test_model_creates_successfully(self, sample_config):
        """Test that model can be created with valid config"""
        model = BERT4Rec(**sample_config)
        assert model is not None

    def test_model_has_correct_attributes(self, bert4rec_model, sample_config):
        """Test model stores configuration correctly"""
        assert bert4rec_model.num_items == sample_config["num_items"]
        assert bert4rec_model.hidden_units == sample_config["hidden_units"]
        assert bert4rec_model.num_heads == sample_config["num_heads"]
        assert bert4rec_model.num_layers == sample_config["num_layers"]
        assert bert4rec_model.random_mask_prob == sample_config["random_mask_prob"]
        assert bert4rec_model.last_item_mask_ratio == sample_config["last_item_mask_ratio"]

    def test_special_tokens_initialized(self, bert4rec_model):
        """Test special tokens (pad, mask) are set correctly"""
        assert bert4rec_model.pad_token == 0
        assert bert4rec_model.mask_token == bert4rec_model.num_items + 1
        assert bert4rec_model.num_tokens == bert4rec_model.num_items + 2

    def test_metadata_embeddings_created(self, bert4rec_model):
        """Test metadata embedding layers are created"""
        assert bert4rec_model.use_genre_emb
        assert bert4rec_model.use_director_emb
        assert bert4rec_model.use_writer_emb
        assert hasattr(bert4rec_model, "genre_emb")
        assert hasattr(bert4rec_model, "director_emb")
        assert hasattr(bert4rec_model, "writer_emb")


    def test_last_item_mask_ratio_initialized(self, sample_config):
        """Test last_item_mask_ratio is properly initialized"""
        # Test with different ratios
        for ratio in [0.0, 0.1, 0.5, 1.0]:
            config = {**sample_config, "last_item_mask_ratio": ratio}
            model = BERT4Rec(**config)
            assert model.last_item_mask_ratio == ratio

    def test_random_mask_prob_initialized(self, sample_config):
        """Test random_mask_prob is properly initialized"""
        # Test with different probabilities
        for prob in [0.0, 0.15, 0.2, 0.5]:
            config = {**sample_config, "random_mask_prob": prob}
            model = BERT4Rec(**config)
            assert model.random_mask_prob == prob


@pytest.mark.unit
class TestBERT4RecForwardPass:
    """Test BERT4Rec forward pass"""

    def test_forward_pass_without_metadata(
        self, bert4rec_model_no_metadata, sample_batch
    ):
        """Test forward pass returns correct shape without metadata"""
        sequences, _ = sample_batch
        batch_size, seq_len = sequences.shape

        logits = bert4rec_model_no_metadata(sequences)

        expected_shape = (batch_size, seq_len, bert4rec_model_no_metadata.num_tokens)
        assert logits.shape == expected_shape

    def test_forward_pass_with_metadata(
        self, bert4rec_model, sample_batch, sample_metadata
    ):
        """Test forward pass returns correct shape with metadata"""
        sequences, _ = sample_batch
        batch_size, seq_len = sequences.shape

        logits = bert4rec_model(sequences, sample_metadata)

        expected_shape = (batch_size, seq_len, bert4rec_model.num_tokens)
        assert logits.shape == expected_shape

    def test_forward_output_is_tensor(self, bert4rec_model_no_metadata, sample_batch):
        """Test forward pass returns a tensor"""
        sequences, _ = sample_batch
        logits = bert4rec_model_no_metadata(sequences)

        assert isinstance(logits, torch.Tensor)

    def test_forward_output_dtype(self, bert4rec_model_no_metadata, sample_batch):
        """Test forward pass returns float tensor"""
        sequences, _ = sample_batch
        logits = bert4rec_model_no_metadata(sequences)

        assert logits.dtype == torch.float32

    def test_forward_no_nan_or_inf(self, bert4rec_model_no_metadata, sample_batch):
        """Test forward pass doesn't produce NaN or Inf"""
        sequences, _ = sample_batch
        logits = bert4rec_model_no_metadata(sequences)

        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()

    @pytest.mark.parametrize("batch_size,seq_len", [(1, 5), (8, 10), (16, 7)])
    def test_different_batch_sizes(
        self, bert4rec_model_no_metadata, batch_size, seq_len
    ):
        """Test model handles different batch sizes (seq_len must be <= max_len)"""
        sequences = torch.randint(1, 51, (batch_size, seq_len))
        logits = bert4rec_model_no_metadata(sequences)

        assert logits.shape[0] == batch_size
        assert logits.shape[1] == seq_len


@pytest.mark.unit
class TestBERT4RecMetadataFusion:
    """Test metadata fusion strategies"""

    def test_concat_fusion(self, sample_config):
        """Test concat fusion strategy"""
        config = {**sample_config, "metadata_fusion": "concat"}
        model = BERT4Rec(**config)

        assert model.metadata_fusion == "concat"
        assert model.fusion_layer is not None

    def test_add_fusion(self, sample_config):
        """Test add fusion strategy"""
        config = {**sample_config, "metadata_fusion": "add"}
        model = BERT4Rec(**config)

        assert model.metadata_fusion == "add"
        assert model.fusion_layer is None

    def test_gate_fusion(self, sample_config):
        """Test gate fusion strategy"""
        config = {**sample_config, "metadata_fusion": "gate"}
        model = BERT4Rec(**config)

        assert model.metadata_fusion == "gate"
        assert hasattr(model, "fusion_gate")

    def test_invalid_fusion_raises_error(self, sample_config):
        """Test invalid fusion strategy raises ValueError"""
        config = {**sample_config, "metadata_fusion": "invalid"}

        with pytest.raises(ValueError):
            BERT4Rec(**config)


@pytest.mark.unit
class TestBERT4RecTrainingStep:
    """Test training step functionality"""

    def test_training_step_without_metadata(
        self, bert4rec_model_no_metadata, sample_batch
    ):
        """Test training step works without metadata"""
        loss = bert4rec_model_no_metadata.training_step(sample_batch, batch_idx=0)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
        assert loss.item() > 0

    def test_training_step_with_metadata(
        self, bert4rec_model, sample_batch, sample_metadata
    ):
        """Test training step works with metadata"""
        sequences, labels = sample_batch
        batch = (sequences, labels, sample_metadata)

        loss = bert4rec_model.training_step(batch, batch_idx=0)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_training_step_backward_compatible(self, bert4rec_model_no_metadata):
        """Test training step is backward compatible (2-element batch)"""
        sequences = torch.randint(1, 51, (4, 10))
        labels = torch.randint(0, 51, (4, 10))

        # Old format (without metadata)
        loss = bert4rec_model_no_metadata.training_step(
            (sequences, labels), batch_idx=0
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0


@pytest.mark.unit
class TestBERT4RecValidationStep:
    """Test validation step functionality"""

    def test_validation_step_without_metadata(self, bert4rec_model_no_metadata):
        """Test validation step without metadata"""
        sequences = torch.randint(1, 51, (4, 10))
        labels = torch.zeros((4, 10), dtype=torch.long)
        targets = torch.randint(1, 51, (4,))

        batch = (sequences, labels, targets)
        result = bert4rec_model_no_metadata.validation_step(batch, batch_idx=0)

        assert "val_hit@10" in result
        assert "val_ndcg@10" in result

    def test_validation_step_with_metadata(self, bert4rec_model, sample_metadata):
        """Test validation step with metadata"""
        sequences = torch.randint(1, 51, (4, 10))
        labels = torch.zeros((4, 10), dtype=torch.long)
        targets = torch.randint(1, 51, (4,))

        batch = (sequences, labels, sample_metadata, targets)
        result = bert4rec_model.validation_step(batch, batch_idx=0)

        assert "val_hit@10" in result
        assert "val_ndcg@10" in result


@pytest.mark.unit
class TestBERT4RecGateValues:
    """Test gate value return and logging functionality"""

    def test_gate_fusion_returns_values(self, sample_config, sample_metadata):
        """Test that gate fusion returns gate values when requested"""
        config = {**sample_config, "metadata_fusion": "gate"}
        model = BERT4Rec(**config)

        sequences = torch.randint(1, 51, (4, 10))
        logits, gate_values = model(sequences, sample_metadata, return_gate_values=True)

        # Check logits shape
        assert logits.shape == (4, 10, model.num_tokens)

        # Check gate values shape: [batch, seq_len, num_features]
        assert gate_values is not None
        assert gate_values.shape[0] == 4  # batch size
        assert gate_values.shape[1] == 10  # seq_len
        assert gate_values.shape[2] > 1  # at least 2 features (item + metadata)

    def test_gate_values_sum_to_one(self, sample_config, sample_metadata):
        """Test that gate values are properly normalized (softmax)"""
        config = {**sample_config, "metadata_fusion": "gate"}
        model = BERT4Rec(**config)

        sequences = torch.randint(1, 51, (4, 10))
        _, gate_values = model(sequences, sample_metadata, return_gate_values=True)

        # Gate values should sum to 1 along feature dimension (due to softmax)
        gate_sums = gate_values.sum(dim=-1)
        assert torch.allclose(gate_sums, torch.ones_like(gate_sums), atol=1e-5)

    def test_gate_values_are_positive(self, sample_config, sample_metadata):
        """Test that all gate values are positive (softmax output)"""
        config = {**sample_config, "metadata_fusion": "gate"}
        model = BERT4Rec(**config)

        sequences = torch.randint(1, 51, (4, 10))
        _, gate_values = model(sequences, sample_metadata, return_gate_values=True)

        # All gate values should be positive (softmax output)
        assert (gate_values >= 0).all()
        assert (gate_values <= 1).all()

    def test_non_gate_fusion_no_gate_values(self, sample_config, sample_metadata):
        """Test that non-gate fusion methods don't return gate values"""
        for fusion_type in ["concat", "add"]:
            config = {**sample_config, "metadata_fusion": fusion_type}
            model = BERT4Rec(**config)

            sequences = torch.randint(1, 51, (4, 10))
            result = model(sequences, sample_metadata, return_gate_values=True)

            # Should only return logits, not tuple
            assert isinstance(result, torch.Tensor)
            assert result.shape == (4, 10, model.num_tokens)

    def test_gate_values_without_flag(self, sample_config, sample_metadata):
        """Test that gate values are not returned when flag is False"""
        config = {**sample_config, "metadata_fusion": "gate"}
        model = BERT4Rec(**config)

        sequences = torch.randint(1, 51, (4, 10))
        result = model(sequences, sample_metadata, return_gate_values=False)

        # Should only return logits, not tuple
        assert isinstance(result, torch.Tensor)
        assert result.shape == (4, 10, model.num_tokens)

    def test_validation_step_logs_gate_values(self, sample_config, sample_metadata):
        """Test that validation step logs gate values for gate fusion"""
        config = {**sample_config, "metadata_fusion": "gate"}
        model = BERT4Rec(**config)

        sequences = torch.randint(1, 51, (4, 10))
        labels = torch.zeros((4, 10), dtype=torch.long)
        targets = torch.randint(1, 51, (4,))

        batch = (sequences, labels, sample_metadata, targets)

        # Run validation step
        result = model.validation_step(batch, batch_idx=0)

        # Check that basic metrics are present
        assert "val_hit@10" in result
        assert "val_ndcg@10" in result

        # Note: Gate values are logged via self.log(), which requires a trainer
        # In unit tests without trainer, we just verify no errors occur

    def test_gate_num_features_matches_enabled_embeddings(self):
        """Test that gate values have correct number of features"""
        # Create model with specific embeddings enabled
        config = {
            "num_items": 100,
            "hidden_units": 64,
            "num_heads": 2,
            "num_layers": 2,
            "max_len": 10,
            "num_genres": 10,
            "num_directors": 20,
            "num_writers": 15,
            "title_embedding_dim": 768,
            "use_genre_emb": True,
            "use_director_emb": True,
            "use_writer_emb": False,  # Disabled
            "use_title_emb": True,
            "metadata_fusion": "gate",
        }
        model = BERT4Rec(**config)

        # Create matching metadata
        batch_size = 2
        seq_len = 5
        metadata = {
            "genres": torch.randint(0, 10, (batch_size, seq_len, 3)),
            "directors": torch.randint(0, 20, (batch_size, seq_len)),
            "title_embs": torch.randn(batch_size, seq_len, 768),
        }

        sequences = torch.randint(1, 51, (batch_size, seq_len))
        _, gate_values = model(sequences, metadata, return_gate_values=True)

        # Should have: item + genre + director + title = 4 features (writer disabled)
        expected_num_features = 4
        assert gate_values.shape[2] == expected_num_features


@pytest.mark.unit
class TestBERT4RecSpecialCases:
    """Test special cases and edge conditions"""

    def test_all_padding_sequence(self, bert4rec_model_no_metadata):
        """Test model handles all-padding sequences"""
        sequences = torch.zeros((2, 5), dtype=torch.long)
        logits = bert4rec_model_no_metadata(sequences)

        assert logits.shape == (2, 5, bert4rec_model_no_metadata.num_tokens)
        assert not torch.isnan(logits).any()

    def test_all_mask_tokens(self, bert4rec_model_no_metadata):
        """Test model handles all-mask sequences"""
        sequences = torch.full(
            (2, 5), bert4rec_model_no_metadata.mask_token, dtype=torch.long
        )
        logits = bert4rec_model_no_metadata(sequences)

        assert logits.shape == (2, 5, bert4rec_model_no_metadata.num_tokens)
        assert not torch.isnan(logits).any()

    def test_single_item_sequence(self, bert4rec_model_no_metadata):
        """Test model handles single-item sequences"""
        sequences = torch.randint(1, 51, (2, 1))
        logits = bert4rec_model_no_metadata(sequences)

        assert logits.shape == (2, 1, bert4rec_model_no_metadata.num_tokens)

    def test_max_length_sequence(self, bert4rec_model_no_metadata):
        """Test model handles max-length sequences"""
        sequences = torch.randint(1, 51, (2, bert4rec_model_no_metadata.max_len))
        logits = bert4rec_model_no_metadata(sequences)

        assert logits.shape == (
            2,
            bert4rec_model_no_metadata.max_len,
            bert4rec_model_no_metadata.num_tokens,
        )


@pytest.mark.unit
class TestBERT4RecDeviceHandling:
    """Test device handling (CPU/GPU)"""

    def test_model_on_cpu(self, bert4rec_model_no_metadata):
        """Test model works on CPU"""
        bert4rec_model_no_metadata = bert4rec_model_no_metadata.cpu()
        sequences = torch.randint(1, 51, (2, 5))

        logits = bert4rec_model_no_metadata(sequences)

        assert logits.device.type == "cpu"

    @pytest.mark.gpu
    def test_model_on_gpu(self, bert4rec_model_no_metadata):
        """Test model works on GPU (if available)"""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")

        bert4rec_model_no_metadata = bert4rec_model_no_metadata.cuda()
        sequences = torch.randint(1, 51, (2, 5)).cuda()

        logits = bert4rec_model_no_metadata(sequences)

        assert logits.device.type == "cuda"
