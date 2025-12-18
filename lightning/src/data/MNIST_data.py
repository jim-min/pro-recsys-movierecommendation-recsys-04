import lightning as L
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import logging


class MNISTDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "~/data/", batch_size=32, target_labels=None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.target_labels = target_labels
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    def _filter_by_labels(self, dataset):
        """특정 레이블만 필터링"""
        indices = [
            i for i, (_, label) in enumerate(dataset) if label in self.target_labels
        ]
        return torch.utils.data.Subset(dataset, indices)

    def setup(self, stage: str = "fit"):
        mnist_test_full = MNIST(
            self.data_dir, train=False, download=True, transform=self.transform
        )
        mnist_full = MNIST(
            self.data_dir, train=True, download=True, transform=self.transform
        )

        # 특정 레이블만 필터링
        logging.info(f"필터링: {self.target_labels}")
        mnist_test_filtered = self._filter_by_labels(mnist_test_full)
        mnist_train_filtered = self._filter_by_labels(mnist_full)

        # train/val split (약 90/10 비율)
        train_size = int(0.9 * len(mnist_train_filtered))
        val_size = len(mnist_train_filtered) - train_size
        self.mnist_train, self.mnist_val = torch.utils.data.random_split(
            mnist_train_filtered, [train_size, val_size]
        )

        self.mnist_test = mnist_test_filtered
        self.mnist_predict = mnist_test_filtered

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train, shuffle=True, batch_size=self.batch_size, num_workers=7
        )

    def val_dataloader(self):
        return DataLoader(
            self.mnist_val, shuffle=False, batch_size=self.batch_size, num_workers=7
        )

    def test_dataloader(self):
        return DataLoader(
            self.mnist_test, shuffle=False, batch_size=self.batch_size, num_workers=7
        )

    def predict_dataloader(self):
        return DataLoader(
            self.mnist_predict, shuffle=False, batch_size=self.batch_size, num_workers=7
        )
