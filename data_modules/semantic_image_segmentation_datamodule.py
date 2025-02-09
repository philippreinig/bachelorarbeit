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

class SemanticImageSegmentationDataModule(LightningDataModule):
    def __init__(
        self,
        scenario: str = "all",
        datasets: List[str] = ["all"],
        batch_size: int = 32,
        image_size: int = 1024,
        num_workers: int = 10,
        itersize: int = 1000,
        mean: Optional[tuple] = (0.0, 0.0, 0.0),
        std: Optional[tuple] = (1.0, 1.0, 1.0),
        classes: Optional[List[str]] = None,
        void: Optional[List[str]] = None,
        ignore_index: Optional[int] = 255,
        dbtype: str = "psycopg@ants",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        self.dbtype = dbtype

        self.scenario = scenario
        self.datasets = datasets

        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = num_workers
        self.itersize = itersize
        self.mean = torch.as_tensor(mean)
        self.std = torch.as_tensor(std)

        self._valid_classes = [name for name in classes if name not in void]
        self._ignore_index = ignore_index

        valid_idx = [classes.index(c) for c in self._valid_classes]
        void_idx = [classes.index(c) for c in void]
        self.filter_void_labels = FilterVoidLabels(valid_idx, void_idx, ignore_index)

    @property
    def classes(self) -> List[str]:
        """Return: the names of valid classes in AKI-Set"""
        return self._valid_classes

    @property
    def num_classes(self) -> int:
        """Return: number of AKI classes"""
        return len(self.classes)

    @property
    def ignore_index(self) -> Optional[int]:
        return self._ignore_index

    def setup(self, stage=None):
        data = {"camera": ["image"], "camera_segmentation": ["camera_segmentation"]}

        self.train_ds = AKIDataset(
            data,
            splits=["training"],
            scenario=self.scenario,
            datasets=self.datasets,
            itersize=self.itersize,
            dbtype=self.dbtype,
            #transforms=self._transforms(),
            shuffle=True
        )

        self.val_ds = AKIDataset(
            data,
            splits=["validation"],
            scenario=self.scenario,
            datasets=self.datasets,
            itersize=self.itersize,
            dbtype=self.dbtype,
            #transforms=self._transforms()
        )

        self.test_ds = AKIDataset(
            data,
            splits=["validation"],
            scenario=self.scenario,
            datasets=self.datasets,
            itersize=self.itersize,
            dbtype=self.dbtype,
            #transforms=self._transforms()
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            #collate_fn=self._prepare_batch
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            #collate_fn=self._prepare_batch
        )

    def test_dataloader(self) -> DataLoader:
        """Same as *val* set, because test annotations are not public"""
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            #collate_fn=self._prepare_batch
        )

    def _prepare_batch(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        input_batch = torch.stack([elem[0] for elem in batch], 0)
        label_batch = torch.stack([elem[1] for elem in batch], 0)
        return input_batch, label_batch


    def _transforms(self) -> Callable:
        return transform_lib.Compose(
            [
                # Images Arrive as tv_tensors.Image at full resolution with dtype=float32 and values in range [0, 1]
                # Labels Arrive as tv_tensors.Mask at full resolution with dtype=int64 and shape [H, W]
                transform_lib.Normalize(mean=self.mean, std=self.std),
                transform_lib.Resize(size=(886, 1600)),
                # Label-Only Transforms
                self.filter_void_labels,
            ]
        )
