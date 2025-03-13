from typing import Any, Callable, List, Optional

from akiset import AKIDataset

import torch
from torch.utils.data import DataLoader

from torchvision.transforms import v2 as transform_lib

from lightning.pytorch import LightningDataModule

import logging
log = logging.getLogger("my_logger")

from utils.data_preparation import weather_condition2numeric_v2
from data_modules.data_module_utils import runs_per_epoch, elems_in_dataloader, get_label_distribution


class WeatherClassificationDataModule(LightningDataModule):
    def __init__(
        self,
        datasets: List[str] = ["all"],
        batch_size: int = 32,
        num_workers: int = 1,
        order_by: str = None,
        shuffle = False,
        dbtype = "psycopg@ants"
    ) -> None:
        """
        Args:
            batch_size: number of examples per training/eval step
        """
        super().__init__()
        self.datasets = datasets
        self.order_by = order_by
        self.shuffle = shuffle

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.dbtype = dbtype

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

        self.train_limit = 8520
        self.val_limit = 2120
        self.test_limit = 1290


        self.sunny_mean = torch.tensor([0.3291, 0.3426, 0.3622])
        self.sunny_std = torch.tensor([0.1929, 0.2077, 0.2460])
        self.rainy_mean = torch.tensor([0.3306, 0.3650, 0.4050])
        self.rainy_std = torch.tensor([0.2084, 0.2113, 0.2848])
        self.normalize_sunny = transform_lib.Normalize(mean=self.sunny_mean, std=self.sunny_std)
        self.normalize_rainy = transform_lib.Normalize(mean=self.rainy_mean, std=self.rainy_std)

        log.info(f"Datasets: {self.datasets}")
        log.info(f"Batch size: {self.batch_size}")
        log.info(f"Workers: {self.num_workers}")
        log.info(f"Order by: {self.order_by}")
        log.info(f"Shuffle: {self.shuffle}")

    @property
    def classes(self) -> List[str]:
        """Return: the names of valid classes"""
        return ["sunny", "rain"]

    @property
    def num_classes(self) -> int:
        """Return: number of classes"""
        return len(self.classes)


    def setup(self, stage=None):
        log.info(f"Running setup function")
        data = {"camera": ["image"], "weather": ["weather"]}

        self.train_ds = AKIDataset(
            data,
            splits=["training"],
            datasets=self.datasets,
            orderby=self.order_by,
            limit=self.train_limit,
            dbtype=self.dbtype,
            shuffle=self.shuffle
        )

        self.val_ds = AKIDataset(
            data,
            splits=["validation"],
            datasets=self.datasets,
            dbtype=self.dbtype,
            orderby=self.order_by,
            limit=self.val_limit,
            shuffle=self.shuffle
        )

        self.test_ds = AKIDataset(
            data,
            splits=["testing"],
            datasets=self.datasets,
            dbtype=self.dbtype,
            orderby=self.order_by,
            limit=self.test_limit,
            shuffle=self.shuffle
        )

        train_ds_amt_sunny_imgs, train_ds_amt_rainy_imgs = get_label_distribution(self.train_ds, 1)
        val_ds_amt_sunny_imgs, val_ds_amt_rainy_imgs = get_label_distribution(self.val_ds, 1)
        test_ds_amt_sunny_imgs, test_ds_amt_rainy_imgs = get_label_distribution(self.test_ds, 1)

        log.info(f"Train dataloader contains {train_ds_amt_sunny_imgs} sunny images, {train_ds_amt_rainy_imgs} rainy images, {elems_in_dataloader(self.train_ds.count, self.train_limit)} total elements and yields {runs_per_epoch(self.train_ds.count, self.batch_size, self.train_limit)} batches per epoch.")
        log.info(f"Validation dataloader contains {val_ds_amt_sunny_imgs} sunny images, {val_ds_amt_rainy_imgs} rainy images, {elems_in_dataloader(self.val_ds.count, self.val_limit)} total elements and yields {runs_per_epoch(self.val_ds.count, self.batch_size, self.val_limit)} batches per epoch.")
        log.info(f"Test dataloader contains {test_ds_amt_sunny_imgs} sunny images, {test_ds_amt_rainy_imgs} rainy images, {elems_in_dataloader(self.test_ds.count, self.test_limit)} total elements and yields {runs_per_epoch(self.test_ds.count, self.batch_size, self.test_limit)} batches per epoch.")

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
        return DataLoader(
            self.test_ds,    
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self._prepare_batch
        )

    def _prepare_batch(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        label_batch = torch.tensor([weather_condition2numeric_v2(elem[1]) for elem in batch], dtype=torch.long)

        img_batch = torch.stack([self._preprocess(label_batch[i])(elem[0]) for i, elem in enumerate(batch)], 0)

        return img_batch, label_batch

    def _preprocess(self, label: int) -> Callable:
        if label == 0: # Sunny image
            return transform_lib.Compose([
                self.normalize_sunny,
                transform_lib.RandomCrop(size=(886, 1600))
            ])
        else: # Rainy image
            return transform_lib.Compose([
                self.normalize_rainy,
                transform_lib.RandomCrop(size=(886, 1600))
            ])
