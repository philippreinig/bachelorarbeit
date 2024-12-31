from typing import Any, Callable, List, Optional

from akiset import AKIDataset

import torch
from torch.utils.data import DataLoader

from torchvision.transforms import v2 as transform_lib

from lightning.pytorch import LightningDataModule

from utils import FilterVoidLabels

import logging
log = logging.getLogger(__name__)

from rich import inspect

class AkisetDataModule(LightningDataModule):
    def __init__(
        self,
        scenario: str,
        datasets: List[str],
        batch_size: int = 32,
        image_size: int = 1024,
        num_workers: int = 10,
        itersize: int = 1000,
        mean: Optional[tuple] = (0.0, 0.0, 0.0),
        std: Optional[tuple] = (1.0, 1.0, 1.0),
        classes: Optional[List[str]] = None,
        void: Optional[List[str]] = None,
        ignore_index: Optional[int] = 255,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            batch_size: number of examples per training/eval step
            image_size: image resolution for training/eval
        """
        super().__init__()
        self.scenario = scenario
        self.datasets = datasets

        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = int(num_workers)  # can also be a string
        self.itersize = itersize
        self.mean = torch.as_tensor(mean)
        self.std = torch.as_tensor(std)

        self._valid = [name for name in classes if name not in void]
        self._ignore = ignore_index

        valid_idx = [classes.index(c) for c in self._valid]
        void_idx = [classes.index(c) for c in void]
        self.filter_void_labels = FilterVoidLabels(valid_idx, void_idx, ignore_index)

    @property
    def classes(self) -> List[str]:
        """Return: the names of valid classes in AKI-Set"""
        return self._valid

    @property
    def num_classes(self) -> int:
        """Return: number of AKI classes"""
        return len(self.classes)

    @property
    def ignore_index(self) -> Optional[int]:
        return self._ignore

    def setup(self, stage=None):
        data = {"camera": ["image"], "camera_segmentation": ["camera_segmentation"]}

        self.train_ds = AKIDataset(
            data,
            splits=["training"],
            scenario=self.scenario,
            datasets=self.datasets,
            itersize=self.itersize,
            dbtype="psycopg@ants",
            shuffle=True
        )

        self.val_ds = AKIDataset(
            data,
            splits=["validation"],
            scenario=self.scenario,
            datasets=self.datasets,
            itersize=self.itersize,
            dbtype="psycopg@ants"
        )

        self.test_ds = AKIDataset(
            data,
            splits=["validation"],
            scenario=self.scenario,
            datasets=self.datasets,
            itersize=self.itersize,
            dbtype="psycopg@ants"
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._preprare_train_batch()
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        """Same as *val* set, because test annotations arent public"""
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self._prepare_val_or_test_batch
        )

    def _preprare_train_batch(self, batch) -> Callable:
        input_batch = torch.stack([elem[0] for elem in batch], 0)
        inspect(input_batch)
        label_batch = torch.stack([elem[1] for elem in batch], 0)
        inspect(label_batch)
        return input_batch, label_batch

        """
        return transform_lib.Compose(
            [
                # Images Arrive as tv_tensors.Image at full resolution with dtype=float32 and values in range [0, 1]
                # Labels Arrive as tv_tensors.Mask at full resolution with dtype=int64 and shape [H, W]
                transform_lib.RandomHorizontalFlip(),
                # TODO: Resizing for easier batching ...
                transform_lib.Resize(size=(720, 1280)),
                # transform_lib.RandomCrop(size=self.image_size),
                # transform_lib.RandomPhotometricDistort(),
                # transform_lib.RandomGrayscale(p=0.05),
                # transform_lib.RandomInvert(p=0.005),
                # transform_lib.ColorJitter(),
                transform_lib.Normalize(mean=self.mean, std=self.std),
                # Label-Only Transforms
                self.filter_void_labels,
            ]
        )
        """

    def _prepare_val_or_test_batch(self, list) -> Callable:
        return transform_lib.Compose(
            [
                # Images Arrive as tv_tensors.Image at full resolution with dtype=float32 and values in range [0, 1]
                # Labels Arrive as tv_tensors.Mask at full resolution with dtype=int64 and shape [H, W]
                transform_lib.Normalize(mean=self.mean, std=self.std),
                # TODO: Resizing for easier batching ...
                transform_lib.Resize(size=(720, 1280)),
                # Label-Only Transforms
                self.filter_void_labels,
            ]
        )
