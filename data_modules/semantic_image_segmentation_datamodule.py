import logging
import torch

from typing import Any, Callable, List, Optional
from torch.utils.data import DataLoader
from lightning.pytorch import LightningDataModule
from data_modules.data_module_utils import FilterVoidLabels, runs_per_epoch

from akiset import AKIDataset

from torchvision.transforms import v2 as transform_lib

log = logging.getLogger("rich")


class SemanticImageSegmentationDataModule(LightningDataModule):
    def __init__(
        self,
        scenario: str = "all",
        datasets: List[str] = ["all"],
        batch_size: int = 32,
        image_size: int = 1024,
        num_workers: int = 10,
        itersize: int = 1000,
        order_by: str = None,
        limit: int = None,
        mean: Optional[tuple] = (0.0, 0.0, 0.0),
        std: Optional[tuple] = (1.0, 1.0, 1.0),
        classes: Optional[List[str]] = None,
        void: Optional[List[str]] = None,
        ignore_index: Optional[int] = 255,
        dbtype: str = "psycopg@ants"
    ) -> None:
        super().__init__()

        self.dbtype = dbtype

        self.scenario = scenario
        self.datasets = datasets
        self.order_by = order_by
        self.limit = limit

        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = num_workers
        self.itersize = itersize
        self.mean = torch.as_tensor(mean)
        self.std = torch.as_tensor(std)

        self._valid_classes = [name for name in classes if name not in void]
        self._ignore_index = ignore_index

        self.valid_idx = [classes.index(c) for c in self._valid_classes]
        self.void_idx = [classes.index(c) for c in void]

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
            orderby=self.order_by,
            limit=self.limit,
            dbtype=self.dbtype,
            transforms=self._transforms(),
            shuffle=True
        )

        self.val_ds = AKIDataset(
            data,
            splits=["validation"],
            scenario=self.scenario,
            datasets=self.datasets,
            itersize=self.itersize,
            dbtype=self.dbtype,
            transforms=self._transforms()
        )

        self.test_ds = AKIDataset(
            data,
            splits=["testing"],
            scenario=self.scenario,
            datasets=self.datasets,
            itersize=self.itersize,
            dbtype=self.dbtype,
            transforms=self._transforms()
        )

        log.info(f"Train dataloader contains {self.train_ds.count} elements. It yields {runs_per_epoch(self.train_ds.count, self.batch_size)} runs per epoch (batch size is {self.batch_size})")
        log.info(f"Validation dataloader contains {self.val_ds.count} elements. It yields {runs_per_epoch(self.val_ds.count, self.batch_size)} runs per epoch (batch size is {self.batch_size})")
        log.info(f"Test dataloader contains {self.test_ds.count} elements. It yields {runs_per_epoch(self.test_ds.count, self.batch_size)} runs per epoch (batch size is {self.batch_size})")


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
                transform_lib.RandomCrop(size=(886, 1600)),
                # Label-Only Transforms
                FilterVoidLabels(self.valid_idx, self.void_idx, self.ignore_index)
            ]
        )
