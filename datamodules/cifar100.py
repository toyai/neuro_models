"""LightningDataModule for CIFAR100."""

import os
from typing import Optional

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchvision import transforms as T
from torchvision.datasets import CIFAR100


class CIFAR100DataModule(LightningDataModule):
    """DataModule for CIFAR100."""

    def __init__(
        self,
        train_transforms_conf: T.Compose = None,
        test_transforms_conf: T.Compose = None,
        train_dataloader_conf: Optional[DictConfig] = None,
        val_dataloader_conf: Optional[DictConfig] = None,
    ):
        super().__init__()
        self.train_transforms_conf = train_transforms_conf
        self.test_transforms_conf = test_transforms_conf
        self.train_dataloader_conf = train_dataloader_conf or OmegaConf.create()
        self.val_dataloader_conf = val_dataloader_conf or OmegaConf.create()

    def setup(self, stage: Optional[str] = None):
        train = CIFAR100(
            os.getcwd(),
            train=True,
            download=True,
            transform=self.train_transforms_conf,
        )
        test = CIFAR100(
            os.getcwd(),
            train=False,
            download=True,
            transform=self.test_transforms_conf,
        )
        train, val = random_split(train, [45000, 5000])
        self.train_dataset = train
        self.val_dataset = val
        self.test_dataset = test

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self.train_dataloader_conf)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self.val_dataloader_conf)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, **self.val_dataloader_conf)
