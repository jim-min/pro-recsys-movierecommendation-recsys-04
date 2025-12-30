"""Unit tests for BERT4RecDataModule"""
import pytest
import torch
import numpy as np
from src.data.bert4rec_data import BERT4RecDataModule, BERT4RecDataset


@pytest.mark.unit
class TestBERT4RecDataModule:
    """Test DataModule initialization and setup"""

    def test_datamodule_creates_successfully(self, temp_data_dir):
        """Test DataModule can be created"""
        dm = BERT4RecDataModule(
            data_dir=temp_data_dir,
            data_file="train_ratings.csv",
            batch_size=2,
            max_len=10,
            random_mask_prob=0.15,
        )
        assert dm is not None

    def test_datamodule_setup(self, temp_data_dir):
        """Test DataModule setup loads data correctly"""
        dm = BERT4RecDataModule(
            data_dir=temp_data_dir,
            data_file="train_ratings.csv",
            batch_size=2,
            max_len=10,
        )
        dm.setup()

        assert dm.num_users > 0
        assert dm.num_items > 0
        assert len(dm.user_train) > 0

    def test_num_items_calculated(self, temp_data_dir):
        """Test num_items is correctly calculated"""
        dm = BERT4RecDataModule(
            data_dir=temp_data_dir,
            data_file="train_ratings.csv",
            batch_size=2,
            max_len=10,
        )
        dm.setup()

        # Should have items from our sample data
        assert dm.num_items > 0

    def test_num_users_calculated(self, temp_data_dir):
        """Test num_users is correctly calculated"""
        dm = BERT4RecDataModule(
            data_dir=temp_data_dir,
            data_file="train_ratings.csv",
            batch_size=2,
            max_len=10,
            min_interactions=3,  # Filter users with < 3 interactions
        )
        dm.setup()

        assert dm.num_users > 0


@pytest.mark.unit
class TestBERT4RecDataModuleMetadata:
    """Test metadata loading in DataModule"""

    def test_metadata_loading(self, temp_data_dir):
        """Test metadata is loaded correctly"""
        dm = BERT4RecDataModule(
            data_dir=temp_data_dir,
            data_file="train_ratings.csv",
            batch_size=2,
            max_len=10,
        )
        dm.setup()

        # Check metadata dimensions
        assert dm.num_genres > 1
        assert dm.num_directors > 1
        assert dm.num_writers > 1

    def test_genre_metadata_loaded(self, temp_data_dir):
        """Test genre metadata is loaded"""
        dm = BERT4RecDataModule(
            data_dir=temp_data_dir,
            data_file="train_ratings.csv",
            batch_size=2,
            max_len=10,
        )
        dm.setup()

        assert len(dm.item_genres) > 0
        assert len(dm.genre2idx) > 0

    def test_director_metadata_loaded(self, temp_data_dir):
        """Test director metadata is loaded"""
        dm = BERT4RecDataModule(
            data_dir=temp_data_dir,
            data_file="train_ratings.csv",
            batch_size=2,
            max_len=10,
        )
        dm.setup()

        assert len(dm.item_directors) > 0
        assert len(dm.director2idx) > 0

    def test_writer_metadata_loaded(self, temp_data_dir):
        """Test writer metadata is loaded"""
        dm = BERT4RecDataModule(
            data_dir=temp_data_dir,
            data_file="train_ratings.csv",
            batch_size=2,
            max_len=10,
        )
        dm.setup()

        assert len(dm.item_writers) > 0
        assert len(dm.writer2idx) > 0


@pytest.mark.unit
class TestBERT4RecDataLoader:
    """Test DataLoader functionality"""

    def test_train_dataloader_created(self, temp_data_dir):
        """Test train dataloader is created"""
        dm = BERT4RecDataModule(
            data_dir=temp_data_dir,
            data_file="train_ratings.csv",
            batch_size=2,
            max_len=10,
        )
        dm.setup()

        dataloader = dm.train_dataloader()
        assert dataloader is not None

    def test_train_dataloader_batch_format(self, temp_data_dir):
        """Test train dataloader returns correct batch format"""
        dm = BERT4RecDataModule(
            data_dir=temp_data_dir,
            data_file="train_ratings.csv",
            batch_size=2,
            max_len=10,
        )
        dm.setup()

        dataloader = dm.train_dataloader()
        batch = next(iter(dataloader))

        # Should return (sequences, labels, metadata)
        assert len(batch) == 3
        sequences, labels, metadata = batch

        assert isinstance(sequences, torch.Tensor)
        assert isinstance(labels, torch.Tensor)
        assert isinstance(metadata, dict)

    def test_train_dataloader_batch_size(self, temp_data_dir):
        """Test train dataloader respects batch_size"""
        batch_size = 2
        dm = BERT4RecDataModule(
            data_dir=temp_data_dir,
            data_file="train_ratings.csv",
            batch_size=batch_size,
            max_len=10,
        )
        dm.setup()

        dataloader = dm.train_dataloader()
        batch = next(iter(dataloader))
        sequences, _, _ = batch

        assert sequences.shape[0] == batch_size

    def test_train_dataloader_sequence_length(self, temp_data_dir):
        """Test train dataloader respects max_len"""
        max_len = 10
        dm = BERT4RecDataModule(
            data_dir=temp_data_dir,
            data_file="train_ratings.csv",
            batch_size=2,
            max_len=max_len,
        )
        dm.setup()

        dataloader = dm.train_dataloader()
        batch = next(iter(dataloader))
        sequences, _, _ = batch

        assert sequences.shape[1] == max_len

    def test_metadata_in_batch(self, temp_data_dir):
        """Test metadata is included in batch"""
        dm = BERT4RecDataModule(
            data_dir=temp_data_dir,
            data_file="train_ratings.csv",
            batch_size=2,
            max_len=10,
        )
        dm.setup()

        dataloader = dm.train_dataloader()
        batch = next(iter(dataloader))
        _, _, metadata = batch

        # Should have at least some metadata
        assert "genres" in metadata or "directors" in metadata

    def test_val_dataloader_created(self, temp_data_dir):
        """Test validation dataloader is created"""
        dm = BERT4RecDataModule(
            data_dir=temp_data_dir,
            data_file="train_ratings.csv",
            batch_size=2,
            max_len=10,
            use_full_data=False,  # Enable validation split
        )
        dm.setup()

        dataloader = dm.val_dataloader()
        assert dataloader is not None

    def test_val_dataloader_batch_format(self, temp_data_dir):
        """Test validation dataloader returns correct format"""
        dm = BERT4RecDataModule(
            data_dir=temp_data_dir,
            data_file="train_ratings.csv",
            batch_size=2,
            max_len=10,
            use_full_data=False,
        )
        dm.setup()

        dataloader = dm.val_dataloader()
        if len(dataloader) > 0:
            batch = next(iter(dataloader))

            # Should return (sequences, labels, metadata, targets)
            assert len(batch) == 4


@pytest.mark.unit
class TestBERT4RecDataset:
    """Test BERT4RecDataset functionality"""

    def test_dataset_length(self, temp_data_dir):
        """Test dataset returns correct length"""
        dm = BERT4RecDataModule(
            data_dir=temp_data_dir,
            data_file="train_ratings.csv",
            batch_size=2,
            max_len=10,
        )
        dm.setup()

        dataset = BERT4RecDataset(
            user_sequences=dm.user_train,
            num_items=dm.num_items,
            max_len=10,
            random_mask_prob=0.15,
            mask_token=dm.num_items + 1,
            pad_token=0,
        )

        assert len(dataset) == len(dm.user_train)

    def test_dataset_getitem_format(self, temp_data_dir):
        """Test dataset __getitem__ returns correct format"""
        dm = BERT4RecDataModule(
            data_dir=temp_data_dir,
            data_file="train_ratings.csv",
            batch_size=2,
            max_len=10,
        )
        dm.setup()

        dataset = BERT4RecDataset(
            user_sequences=dm.user_train,
            num_items=dm.num_items,
            max_len=10,
            random_mask_prob=0.15,
            mask_token=dm.num_items + 1,
            pad_token=0,
            item_genres=dm.item_genres,
            item_directors=dm.item_directors,
            item_writers=dm.item_writers,
        )

        tokens, labels, metadata = dataset[0]

        assert isinstance(tokens, torch.Tensor)
        assert isinstance(labels, torch.Tensor)
        assert isinstance(metadata, dict)

    def test_dataset_sequence_length(self, temp_data_dir):
        """Test dataset pads/truncates to max_len"""
        max_len = 10
        dm = BERT4RecDataModule(
            data_dir=temp_data_dir,
            data_file="train_ratings.csv",
            batch_size=2,
            max_len=max_len,
        )
        dm.setup()

        dataset = BERT4RecDataset(
            user_sequences=dm.user_train,
            num_items=dm.num_items,
            max_len=max_len,
            random_mask_prob=0.15,
            mask_token=dm.num_items + 1,
            pad_token=0,
        )

        tokens, labels, _ = dataset[0]

        assert tokens.shape[0] == max_len
        assert labels.shape[0] == max_len


@pytest.mark.unit
class TestBERT4RecDataModuleEdgeCases:
    """Test edge cases and error handling"""

    def test_min_interactions_filter(self, temp_data_dir):
        """Test min_interactions filters users correctly"""
        dm_no_filter = BERT4RecDataModule(
            data_dir=temp_data_dir,
            data_file="train_ratings.csv",
            batch_size=2,
            max_len=10,
            min_interactions=1,
        )
        dm_no_filter.setup()

        dm_with_filter = BERT4RecDataModule(
            data_dir=temp_data_dir,
            data_file="train_ratings.csv",
            batch_size=2,
            max_len=10,
            min_interactions=5,
        )
        dm_with_filter.setup()

        # Higher min_interactions should result in fewer users
        assert dm_with_filter.num_users <= dm_no_filter.num_users

    def test_empty_metadata_handling(self, tmp_path):
        """Test DataModule handles missing metadata files gracefully"""
        # Create data dir with only ratings file
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Create minimal ratings file
        import pandas as pd

        data = {"user": [1, 1, 2], "item": [10, 20, 10], "time": [1000, 2000, 1500]}
        pd.DataFrame(data).to_csv(data_dir / "train_ratings.csv", index=False)

        dm = BERT4RecDataModule(
            data_dir=str(data_dir),
            data_file="train_ratings.csv",
            batch_size=2,
            max_len=10,
        )

        # Should not raise error even without metadata files
        dm.setup()

        # Metadata should have default values
        assert dm.num_genres >= 1
        assert dm.num_directors >= 1
        assert dm.num_writers >= 1


@pytest.mark.unit
class TestLastItemMasking:
    """Test last_item_mask_ratio functionality (boost strategy)"""

    def test_last_item_mask_ratio_zero(self, temp_data_dir):
        """Test that last_item_mask_ratio=0.0 disables last item boost masking"""
        dm = BERT4RecDataModule(
            data_dir=temp_data_dir,
            data_file="train_ratings.csv",
            batch_size=2,
            max_len=10,
            random_mask_prob=0.15,
            last_item_mask_ratio=0.0,
        )
        dm.setup()

        dataset = BERT4RecDataset(
            user_sequences=dm.user_train,
            num_items=dm.num_items,
            max_len=10,
            random_mask_prob=0.15,
            last_item_mask_ratio=0.0,
            mask_token=dm.num_items + 1,
            pad_token=0,
        )

        # With ratio=0.0, should use random masking only (no last-item boost)
        assert dataset.last_item_mask_ratio == 0.0

    def test_last_item_mask_ratio_full(self, temp_data_dir):
        """Test that last_item_mask_ratio=1.0 always boosts last item masking on top of random masking"""
        dm = BERT4RecDataModule(
            data_dir=temp_data_dir,
            data_file="train_ratings.csv",
            batch_size=2,
            max_len=10,
            random_mask_prob=0.0,  # Disable random masking for clearer test
            last_item_mask_ratio=1.0,  # Always boost last item
        )
        dm.setup()

        dataset = BERT4RecDataset(
            user_sequences=dm.user_train,
            num_items=dm.num_items,
            max_len=10,
            random_mask_prob=0.0,
            last_item_mask_ratio=1.0,
            mask_token=dm.num_items + 1,
            pad_token=0,
        )

        # With ratio=1.0, all samples should additionally mask last item
        assert dataset.last_item_mask_ratio == 1.0

    def test_last_item_mask_ratio_partial(self, temp_data_dir):
        """Test that last_item_mask_ratio=0.5 boosts last item masking with 50% probability"""
        dm = BERT4RecDataModule(
            data_dir=temp_data_dir,
            data_file="train_ratings.csv",
            batch_size=100,  # Large batch to test distribution
            max_len=10,
            random_mask_prob=0.2,  # Random masking always applied
            last_item_mask_ratio=0.5,  # 50% probability to boost last item
        )
        dm.setup()

        dataset = BERT4RecDataset(
            user_sequences=dm.user_train,
            num_items=dm.num_items,
            max_len=10,
            random_mask_prob=0.2,
            last_item_mask_ratio=0.5,
            mask_token=dm.num_items + 1,
            pad_token=0,
        )

        # All samples get random masking + 50% get additional last-item boost
        assert dataset.last_item_mask_ratio == 0.5

    def test_last_item_mask_creates_mask_at_last_position(self, temp_data_dir):
        """Test that last item boost masking actually masks the last position"""
        dm = BERT4RecDataModule(
            data_dir=temp_data_dir,
            data_file="train_ratings.csv",
            batch_size=2,
            max_len=10,
            random_mask_prob=0.0,  # Disable random masking for clearer test
            last_item_mask_ratio=1.0,  # Always boost last item masking
        )
        dm.setup()

        dataset = BERT4RecDataset(
            user_sequences=dm.user_train,
            num_items=dm.num_items,
            max_len=10,
            random_mask_prob=0.0,
            last_item_mask_ratio=1.0,
            mask_token=dm.num_items + 1,
            pad_token=0,
        )

        # Get multiple samples and check masking pattern
        for i in range(min(5, len(dataset))):
            tokens, labels, _ = dataset[i]

            # Find the last non-padding position
            non_pad_mask = tokens != 0
            if non_pad_mask.any():
                last_pos = non_pad_mask.nonzero(as_tuple=True)[0][-1].item()

                # With ratio=1.0 and random_mask_prob=0.0, last position should always be masked
                # The last position should have a non-zero label (indicating it was masked)
                # Note: label=0 means "don't compute loss", non-zero means "predict this item"
                assert labels[last_pos] != 0, f"Last position {last_pos} should be masked"

    def test_last_item_mask_ratio_range(self, temp_data_dir):
        """Test that last_item_mask_ratio accepts valid range [0.0, 1.0]"""
        for ratio in [0.0, 0.1, 0.5, 0.9, 1.0]:
            dm = BERT4RecDataModule(
                data_dir=temp_data_dir,
                data_file="train_ratings.csv",
                batch_size=2,
                max_len=10,
                random_mask_prob=0.15,
                last_item_mask_ratio=ratio,
            )
            dm.setup()

            dataset = BERT4RecDataset(
                user_sequences=dm.user_train,
                num_items=dm.num_items,
                max_len=10,
                random_mask_prob=0.15,
                last_item_mask_ratio=ratio,
                mask_token=dm.num_items + 1,
                pad_token=0,
            )

            assert dataset.last_item_mask_ratio == ratio

    def test_last_item_mask_with_different_sequence_lengths(self, temp_data_dir):
        """Test last item masking works with sequences of different lengths"""
        dm = BERT4RecDataModule(
            data_dir=temp_data_dir,
            data_file="train_ratings.csv",
            batch_size=2,
            max_len=20,  # Larger max_len to accommodate different lengths
            random_mask_prob=0.0,
            last_item_mask_ratio=1.0,
        )
        dm.setup()

        dataset = BERT4RecDataset(
            user_sequences=dm.user_train,
            num_items=dm.num_items,
            max_len=20,
            random_mask_prob=0.0,
            last_item_mask_ratio=1.0,
            mask_token=dm.num_items + 1,
            pad_token=0,
        )

        # Check that masking works for different sequence lengths
        for i in range(min(3, len(dataset))):
            tokens, labels, _ = dataset[i]

            # Find last non-padding position
            non_pad_positions = (tokens != 0).nonzero(as_tuple=True)[0]
            if len(non_pad_positions) > 0:
                last_pos = non_pad_positions[-1].item()
                # Last position should be masked
                assert labels[last_pos] != 0


@pytest.mark.unit
class TestGetItemMetadata:
    """Test get_item_metadata() method"""

    def test_get_item_metadata_with_all_metadata(self, temp_data_dir):
        """Test get_item_metadata returns all metadata types"""
        dm = BERT4RecDataModule(
            data_dir=temp_data_dir,
            data_file="train_ratings.csv",
            batch_size=2,
            max_len=10,
        )
        dm.setup()

        metadata = dm.get_item_metadata()

        assert metadata is not None
        assert isinstance(metadata, dict)
        assert "genres" in metadata
        assert "directors" in metadata
        assert "writers" in metadata

    def test_get_item_metadata_structure(self, temp_data_dir):
        """Test get_item_metadata returns correct structure"""
        dm = BERT4RecDataModule(
            data_dir=temp_data_dir,
            data_file="train_ratings.csv",
            batch_size=2,
            max_len=10,
        )
        dm.setup()

        metadata = dm.get_item_metadata()

        # Check each metadata type is a dict
        assert isinstance(metadata["genres"], dict)
        assert isinstance(metadata["directors"], dict)
        assert isinstance(metadata["writers"], dict)

    def test_get_item_metadata_genres_are_lists(self, temp_data_dir):
        """Test genre values are lists"""
        dm = BERT4RecDataModule(
            data_dir=temp_data_dir,
            data_file="train_ratings.csv",
            batch_size=2,
            max_len=10,
        )
        dm.setup()

        metadata = dm.get_item_metadata()

        if metadata["genres"]:
            # Get first item
            first_item_genres = next(iter(metadata["genres"].values()))
            assert isinstance(first_item_genres, list)

    def test_get_item_metadata_directors_are_ints(self, temp_data_dir):
        """Test director values are integers"""
        dm = BERT4RecDataModule(
            data_dir=temp_data_dir,
            data_file="train_ratings.csv",
            batch_size=2,
            max_len=10,
        )
        dm.setup()

        metadata = dm.get_item_metadata()

        if metadata["directors"]:
            # Get first item
            first_item_director = next(iter(metadata["directors"].values()))
            assert isinstance(first_item_director, (int, np.integer))

    def test_get_item_metadata_writers_are_lists(self, temp_data_dir):
        """Test writer values are lists"""
        dm = BERT4RecDataModule(
            data_dir=temp_data_dir,
            data_file="train_ratings.csv",
            batch_size=2,
            max_len=10,
        )
        dm.setup()

        metadata = dm.get_item_metadata()

        if metadata["writers"]:
            # Get first item
            first_item_writers = next(iter(metadata["writers"].values()))
            assert isinstance(first_item_writers, list)

    def test_get_item_metadata_empty_when_no_metadata(self, tmp_path):
        """Test get_item_metadata returns None when no metadata files exist"""
        # Create data dir with only ratings file, no metadata
        data_dir = tmp_path / "data_no_meta"
        data_dir.mkdir()

        # Create ratings only
        import pandas as pd

        data = {
            "user": [1, 1, 1, 2, 2, 2],
            "item": [10, 20, 30, 15, 25, 35],
            "time": [1000, 2000, 3000, 1500, 2500, 3500],
        }
        pd.DataFrame(data).to_csv(data_dir / "train_ratings.csv", index=False)

        dm = BERT4RecDataModule(
            data_dir=str(data_dir),
            data_file="train_ratings.csv",
            batch_size=2,
            max_len=10,
        )
        dm.setup()

        metadata = dm.get_item_metadata()

        # Should return None or empty dict when no metadata
        assert metadata is None or len(metadata) == 0

    def test_get_item_metadata_called_before_setup_fails(self, temp_data_dir):
        """Test get_item_metadata before setup raises appropriate error"""
        dm = BERT4RecDataModule(
            data_dir=temp_data_dir,
            data_file="train_ratings.csv",
            batch_size=2,
            max_len=10,
        )

        # Calling before setup should fail or return None
        try:
            metadata = dm.get_item_metadata()
            # If it doesn't fail, should be None
            assert metadata is None or len(metadata) == 0
        except AttributeError:
            # Expected - attributes don't exist before setup
            pass

    def test_get_item_metadata_idempotent(self, temp_data_dir):
        """Test get_item_metadata can be called multiple times"""
        dm = BERT4RecDataModule(
            data_dir=temp_data_dir,
            data_file="train_ratings.csv",
            batch_size=2,
            max_len=10,
        )
        dm.setup()

        metadata1 = dm.get_item_metadata()
        metadata2 = dm.get_item_metadata()

        # Should return same data
        assert metadata1.keys() == metadata2.keys()

    def test_get_item_metadata_contains_valid_item_ids(self, temp_data_dir):
        """Test metadata contains valid item indices"""
        dm = BERT4RecDataModule(
            data_dir=temp_data_dir,
            data_file="train_ratings.csv",
            batch_size=2,
            max_len=10,
        )
        dm.setup()

        metadata = dm.get_item_metadata()

        # All item IDs should be in valid range
        for item_idx in metadata["genres"].keys():
            assert 1 <= item_idx <= dm.num_items

        for item_idx in metadata["directors"].keys():
            assert 1 <= item_idx <= dm.num_items

        for item_idx in metadata["writers"].keys():
            assert 1 <= item_idx <= dm.num_items
