from typing import Any, Callable, List, Optional

from akiset import AKIDataset

import torch
from torch.utils.data import DataLoader

from torchvision.transforms import v2 as transform_lib
import pytorch_lightning as pl

from pytorch_lightning import LightningDataModule

import logging
log = logging.getLogger(__name__)

from utils import weather_condition2numeric

class WeatherClassificationDataModule(LightningDataModule):
    def __init__(
        self,
        datasets: List[str] = ["all"],
        batch_size: int = 128,
        image_size: int = 1024,
        num_workers: int = 2,
        itersize: int = 1000,
        mean: Optional[tuple] = (0.0, 0.0, 0.0),
        std: Optional[tuple] = (1.0, 1.0, 1.0),
        ignore_index: Optional[int] = 255,
        dbtype = "psycopg@ants",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            batch_size: number of examples per training/eval step
            image_size: image resolution for training/eval
        """
        super().__init__()
        self.scenario = "dayrainclear"
        self.datasets = datasets

        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = int(num_workers)  # can also be a string
        self.itersize = itersize
        self.mean = torch.as_tensor(mean)
        self.std = torch.as_tensor(std)

        self._ignore = ignore_index
        self.dbtype = dbtype

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    @property
    def classes(self) -> List[str]:
        """Return: the names of valid classes"""
        return ["clear/sunny", "rain"]

    @property
    def num_classes(self) -> int:
        """Return: number of classes"""
        return 2

    @property
    def ignore_index(self) -> Optional[int]:
        return self._ignoreing

    def setup(self, stage=None):
        log.info(f"Running setup function")
        data = {"camera": ["image"], "weather": ["weather"]}

        self.train_ds = AKIDataset(
            data,
            splits=["train"],
            scenario=self.scenario,
            datasets=self.datasets,
            itersize=self.itersize,
            dbtype=self.dbtype,
            shuffle=True
        )

        self.val_ds = AKIDataset(
            data,
            splits=["validation"],
            scenario=self.scenario,
            datasets=self.datasets,
            itersize=self.itersize,
            dbtype=self.dbtype
        )

        self.test_ds = AKIDataset(
            data,
            splits=["validation"],
            scenario=self.scenario,
            datasets=self.datasets,
            itersize=self.itersize,
            dbtype=self.dbtype,
            limit=10_000
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._prepare_batch
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self._prepare_batch
        )

    def test_dataloader(self) -> DataLoader:
        """Same as *val* set, because test annotations are not public"""
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self._prepare_batch
        )

    def _prepare_batch(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        input_batch = torch.stack([self._preprocess()(elem[0]) for elem in batch], 0)
        label_batch = torch.tensor([weather_condition2numeric(elem[1]) for elem in batch], dtype=torch.float32)

        return input_batch, label_batch

    def _preprocess(self) -> Callable:
        return transform_lib.Compose([
            transform_lib.Normalize(mean=self.mean, std=self.std),
            transform_lib.RandomCrop(size=(886, 1600))
        ])
