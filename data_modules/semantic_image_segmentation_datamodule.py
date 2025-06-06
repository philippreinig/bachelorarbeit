import logging
import torch

from typing import Any, Callable, List, Optional
from torch.utils.data import DataLoader
from lightning.pytorch import LightningDataModule
from data_modules.data_module_utils import FilterVoidLabels, runs_per_epoch, elems_in_dataset, get_label_distribution

from akiset import AKIDataset

from torchvision.transforms import v2 as transform_lib

log = logging.getLogger("rich")


class SemanticImageSegmentationDataModule(LightningDataModule):
    def __init__(
        self,
        scenario: str = "all",
        datasets: List[str] = ["all"],
        batch_size: int = 32,
        num_workers: int = 16,
        itersize: int = 1000,
        order_by: str = None,
        mean: Optional[tuple] = (0.0, 0.0, 0.0),
        std: Optional[tuple] = (1.0, 1.0, 1.0),
        classes: Optional[List[str]] = None,
        void: Optional[List[str]] = None,
        ignore_index: Optional[int] = 255,
        shuffle: bool = False,
        train_limit: int = 10_000_000,
        val_limit: int = 10_000_000,
        test_limit: int = 10_000_000,
        log_distributions: bool = False,
        dbtype: str = "psycopg@ants"
    ) -> None:
        super().__init__()

        self.dbtype = dbtype

        self.scenario = scenario
        self.datasets = datasets
        self.order_by = order_by
        self.train_limit = train_limit
        self.val_limit = val_limit
        self.test_limit = test_limit

        self.shuffle = shuffle

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.itersize = itersize
        self.mean = torch.as_tensor(mean)
        self.std = torch.as_tensor(std)

        self.log_distributions = log_distributions

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
        log.info(f"Running setup function")
        data = {"camera": ["image"], "camera_segmentation": ["camera_segmentation"], "weather": ["weather"]}

        self.train_ds = AKIDataset(
            data,
            splits=["training"],
            scenario=self.scenario,
            datasets=self.datasets,
            itersize=self.itersize,
            orderby=self.order_by,
            limit=self.train_limit,
            dbtype=self.dbtype,
            transforms=self._transforms(),
            shuffle=self.shuffle
        )

        self.val_ds = AKIDataset(
            data,
            splits=["validation"],
            scenario=self.scenario,
            datasets=self.datasets,
            itersize=self.itersize,
            limit=self.val_limit,
            orderby=self.order_by,
            dbtype=self.dbtype,
            transforms=self._transforms(),
            shuffle=self.shuffle
        )

        self.test_ds = AKIDataset(
            data,
            splits=["testing"],
            scenario=self.scenario,
            datasets=self.datasets,
            itersize=self.itersize,
            limit=self.test_limit,
            orderby=self.order_by,
            dbtype=self.dbtype,
            transforms=self._transforms(),
            shuffle = self.shuffle
        )

        if self.log_distributions:
            train_ds_amt_sunny_imgs, train_ds_amt_rainy_imgs = get_label_distribution(self.train_ds, 2)
            val_ds_amt_sunny_imgs, val_ds_amt_rainy_imgs = get_label_distribution(self.val_ds, 2)
            test_ds_amt_sunny_imgs, test_ds_amt_rainy_imgs = get_label_distribution(self.test_ds, 2)
            log.info(f"Train dataloader contains {train_ds_amt_sunny_imgs} point clouds in sunny situations, {train_ds_amt_rainy_imgs} point clouds in rainy situations, {elems_in_dataset(self.train_ds.count, self.train_limit)} total elements and yields {runs_per_epoch(self.train_ds.count, self.batch_size, self.train_limit)} batches per epoch.")
            log.info(f"Validation dataloader contains {val_ds_amt_sunny_imgs} point clouds in sunny situations, {val_ds_amt_rainy_imgs} point clouds in rainy situations, {elems_in_dataset(self.val_ds.count, self.val_limit)} total elements and yields {runs_per_epoch(self.val_ds.count, self.batch_size, self.val_limit)} batches per epoch.")
            log.info(f"Test dataloader contains {test_ds_amt_sunny_imgs} point clouds in sunny situations, {test_ds_amt_rainy_imgs} point clouds in rainy situations, {elems_in_dataset(self.test_ds.count, self.test_limit)} total elements and yields {runs_per_epoch(self.test_ds.count, self.batch_size, self.test_limit)} batches per epoch.")
        else:
            log.info(f"Train dataloader contains {elems_in_dataset(self.train_ds.count, self.train_limit)} elements and yields {runs_per_epoch(self.train_ds.count, self.batch_size, self.train_limit)} batches per epoch.")
            log.info(f"Validation dataloader contains {elems_in_dataset(self.val_ds.count, self.val_limit)} elements and yields {runs_per_epoch(self.val_ds.count, self.batch_size, self.val_limit)} batches per epoch.")
            log.info(f"Test dataloader contains {elems_in_dataset(self.test_ds.count, self.test_limit)} elements and yields {runs_per_epoch(self.test_ds.count, self.batch_size, self.test_limit)} batches per epoch.")
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._prepare_batch
        )

    def val_dataloader(self) -> DataLoader:
        log.info(f"Creating validation loader with batch size: {self.batch_size}")
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
