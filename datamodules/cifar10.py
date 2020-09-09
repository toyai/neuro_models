"""LightningDataModule for CIFAR10."""

import os
from typing import Optional, Union

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchvision import transforms as T
from torchvision.datasets import CIFAR10


class CIFAR10DataModule(LightningDataModule):
    """DataModule for CIFAR10."""

    def __init__(
        self,
        train_transforms_conf: T.Compose,
        test_transforms_conf: T.Compose,
        train_dataset_size: Union[float, int],
        train_dataloader_conf: Optional[DictConfig] = None,
        val_dataloader_conf: Optional[DictConfig] = None,
    ):
        super().__init__()
        self.train_transforms_conf = train_transforms_conf
        self.test_transforms_conf = test_transforms_conf
        self.train_dataset_size = train_dataset_size
        self.train_dataloader_conf = train_dataloader_conf or OmegaConf.create()
        self.val_dataloader_conf = val_dataloader_conf or OmegaConf.create()

    def setup(self, stage: Optional[str] = None):
        train = CIFAR10(
            os.getcwd(),
            train=True,
            download=True,
            transform=self.train_transforms_conf,
        )
        test = CIFAR10(
            os.getcwd(),
            train=False,
            download=True,
            transform=self.test_transforms_conf,
        )
        split_sizes = [
            round(len(train) * self.train_dataset_size),
            round(len(train) * (1 - self.train_dataset_size)),
        ]
        train, val = random_split(train, split_sizes)
        self.train_dataset = train
        self.val_dataset = val
        self.test_dataset = test

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self.train_dataloader_conf)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self.val_dataloader_conf)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, **self.val_dataloader_conf)
