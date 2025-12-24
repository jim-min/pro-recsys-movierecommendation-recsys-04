import os
import logging
import numpy as np
import pandas as pd
import lightning as L
import torch
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader

log = logging.getLogger(__name__)


class BERT4RecDataset(Dataset):
    """
    Dataset for BERT4Rec training

    Applies BERT-style masking to sequences during training
    """

    def __init__(self, user_sequences, num_items, max_len, mask_prob, mask_token, pad_token):
        """
        Args:
            user_sequences: Dict[user_id, List[item_id]] - User interaction sequences
            num_items: Number of unique items
            max_len: Maximum sequence length
            mask_prob: Probability of masking items
            mask_token: Token ID for [MASK]
            pad_token: Token ID for padding
        """
        self.user_sequences = user_sequences
        self.num_items = num_items
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.mask_token = mask_token
        self.pad_token = pad_token

        # Get list of users for indexing
        self.users = list(user_sequences.keys())

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        """
        Returns:
            tokens: [max_len] - Masked sequence
            labels: [max_len] - Labels for loss (0 for non-masked positions)
        """
        user = self.users[idx]
        seq = self.user_sequences[user]

        # Apply masking
        tokens, labels = self._mask_sequence(seq)

        # Truncate or pad
        tokens = tokens[-self.max_len:]
        labels = labels[-self.max_len:]

        # Pad if necessary
        pad_len = self.max_len - len(tokens)
        if pad_len > 0:
            tokens = [self.pad_token] * pad_len + tokens
            labels = [self.pad_token] * pad_len + labels

        return torch.LongTensor(tokens), torch.LongTensor(labels)

    def _mask_sequence(self, seq):
        """
        Apply BERT-style masking to sequence

        Args:
            seq: List[int] - Original sequence
        Returns:
            tokens: List[int] - Masked sequence
            labels: List[int] - Labels (0 for non-masked, original item for masked)
        """
        tokens = []
        labels = []

        for item in seq:
            prob = np.random.random()

            if prob < self.mask_prob:
                # This position will be masked
                prob /= self.mask_prob

                if prob < 0.8:
                    # 80%: Replace with [MASK]
                    tokens.append(self.mask_token)
                elif prob < 0.9:
                    # 10%: Replace with random item (excluding pad)
                    tokens.append(np.random.randint(1, self.num_items + 1))
                else:
                    # 10%: Keep original
                    tokens.append(item)

                labels.append(item)  # Original item as label
            else:
                # Not masked
                tokens.append(item)
                labels.append(self.pad_token)  # Ignore in loss (label = 0)

        return tokens, labels


class BERT4RecValidationDataset(Dataset):
    """
    Dataset for BERT4Rec validation

    Each sample includes:
    - Full sequence with [MASK] at the end
    - Ground truth item (next item to predict)
    """

    def __init__(self, user_sequences, user_targets, num_items, max_len, mask_token, pad_token):
        """
        Args:
            user_sequences: Dict[user_id, List[item_id]] - User training sequences
            user_targets: Dict[user_id, item_id] - Ground truth items to predict
            num_items: Number of unique items
            max_len: Maximum sequence length
            mask_token: Token ID for [MASK]
            pad_token: Token ID for padding
        """
        self.user_sequences = user_sequences
        self.user_targets = user_targets
        self.num_items = num_items
        self.max_len = max_len
        self.mask_token = mask_token
        self.pad_token = pad_token

        self.users = list(user_sequences.keys())

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        """
        Returns:
            tokens: [max_len] - Sequence with [MASK] at the end
            labels: [max_len] - Dummy labels (not used in validation)
            target: int - Ground truth item
        """
        user = self.users[idx]
        seq = self.user_sequences[user]
        target = self.user_targets[user]

        # Add [MASK] token at the end
        tokens = (list(seq) + [self.mask_token])[-self.max_len:]

        # Pad if necessary
        pad_len = self.max_len - len(tokens)
        if pad_len > 0:
            tokens = [self.pad_token] * pad_len + tokens

        # Dummy labels (all zeros, not used in validation)
        labels = [self.pad_token] * self.max_len

        return torch.LongTensor(tokens), torch.LongTensor(labels), torch.LongTensor([target])


class BERT4RecDataModule(L.LightningDataModule):
    """
    LightningDataModule for BERT4Rec

    Hydra 설정 사용:
        data:
            data_dir: "~/data/train/"
            data_file: "train_ratings.csv"
            batch_size: 128
            max_len: 50
            mask_prob: 0.15
            min_interactions: 3
            seed: 42
            num_workers: 4

    Data Format:
        CSV with columns: user, item, time (optional)
    """

    def __init__(
        self,
        data_dir: str,
        data_file: str = "train_ratings.csv",
        batch_size: int = 128,
        max_len: int = 50,
        mask_prob: float = 0.15,
        min_interactions: int = 3,
        seed: int = 42,
        num_workers: int = 4,
    ):
        super().__init__()
        self.data_dir = os.path.expanduser(data_dir)
        self.data_file = data_file
        self.batch_size = batch_size
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.min_interactions = min_interactions
        self.seed = seed
        self.num_workers = num_workers

        # Special tokens
        self.pad_token = 0
        self.mask_token = None  # Will be set to num_items + 1

        # Data will be loaded in setup()
        self.num_users = None
        self.num_items = None
        self.user_train = None
        self.user_valid = None
        self.item2idx = None
        self.user2idx = None
        self.idx2item = None
        self.idx2user = None

    def prepare_data(self):
        """
        Download or prepare data (called only on 1 GPU/TPU)
        For BERT4Rec, we just check if data file exists
        """
        data_path = os.path.join(self.data_dir, self.data_file)
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")

    def setup(self, stage=None):
        """
        Load and split data (called on every GPU/TPU)

        Args:
            stage: 'fit', 'validate', 'test', or 'predict'
        """
        # Load data
        data_path = os.path.join(self.data_dir, self.data_file)
        log.info(f"Loading data from {data_path}")

        df = pd.read_csv(data_path)
        log.info(f"Data loaded. Shape: {df.shape}")

        # Check required columns
        if 'user' not in df.columns or 'item' not in df.columns:
            raise ValueError("Data must have 'user' and 'item' columns")

        # Get unique items and users
        item_ids = df['item'].unique()
        user_ids = df['user'].unique()

        # Create mappings (item: 1~num_items, user: 0~num_users-1)
        self.item2idx = pd.Series(data=np.arange(len(item_ids)) + 1, index=item_ids)
        self.user2idx = pd.Series(data=np.arange(len(user_ids)), index=user_ids)

        # Reverse mappings
        self.idx2item = pd.Series(index=self.item2idx.values, data=self.item2idx.index)
        self.idx2user = pd.Series(index=self.user2idx.values, data=self.user2idx.index)

        self.num_items = len(item_ids)
        self.num_users = len(user_ids)
        self.mask_token = self.num_items + 1

        log.info(f"num_users: {self.num_users}, num_items: {self.num_items}")

        # Re-index dataframe
        df['item_idx'] = df['item'].map(self.item2idx)
        df['user_idx'] = df['user'].map(self.user2idx)

        # Sort by user and time (if available)
        if 'time' in df.columns:
            df = df.sort_values(['user_idx', 'time'])
        else:
            df = df.sort_values(['user_idx'])

        # Group by user
        user_sequences = defaultdict(list)
        for user_idx, item_idx in zip(df['user_idx'], df['item_idx']):
            user_sequences[user_idx].append(item_idx)

        # Filter users with minimum interactions
        user_sequences = {
            u: seq for u, seq in user_sequences.items()
            if len(seq) >= self.min_interactions
        }

        log.info(f"Users after filtering (min_interactions={self.min_interactions}): {len(user_sequences)}")

        # Split: last item for validation, rest for training
        self.user_train = {}
        self.user_valid = {}

        for user, seq in user_sequences.items():
            self.user_train[user] = seq[:-1]
            self.user_valid[user] = seq[-1]  # Last item

        log.info(f"Train sequences: {len(self.user_train)}")
        log.info(f"Valid sequences: {len(self.user_valid)}")

        # Update num_users to filtered count
        self.num_users = len(self.user_train)

    def train_dataloader(self):
        """Create training dataloader"""
        dataset = BERT4RecDataset(
            user_sequences=self.user_train,
            num_items=self.num_items,
            max_len=self.max_len,
            mask_prob=self.mask_prob,
            mask_token=self.mask_token,
            pad_token=self.pad_token,
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        """Create validation dataloader"""
        dataset = BERT4RecValidationDataset(
            user_sequences=self.user_train,
            user_targets=self.user_valid,
            num_items=self.num_items,
            max_len=self.max_len,
            mask_token=self.mask_token,
            pad_token=self.pad_token,
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def get_user_sequence(self, user_id):
        """
        Get training sequence for a specific user (by original ID)

        Args:
            user_id: Original user ID
        Returns:
            List of item indices
        """
        user_idx = self.user2idx.get(user_id)
        if user_idx is None:
            return None
        return self.user_train.get(user_idx)

    def get_all_sequences(self):
        """
        Get all training sequences

        Returns:
            Dict[user_idx, List[item_idx]]
        """
        return self.user_train
