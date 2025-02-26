import logging
import torch

from typing import Any, Callable, List, Optional
from torch.utils.data import DataLoader
from lightning.pytorch import LightningDataModule
from data_modules.data_module_utils import FilterVoidLabels, runs_per_epoch

from akiset import AKIDataset

from torchvision.transforms import v2 as transform_lib

log = logging.getLogger("rich")


class SemanticLidarSegmentationDataModule(LightningDataModule):
    def __init__(
        self,
        scenario: str = "all",
        datasets: List[str] = ["all"],
        batch_size: int = 32,
        num_workers: int = 10,
        itersize: int = 1000,
        order_by: str = None,
        limit: int = None,
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
        self.num_workers = num_workers
        self.itersize = itersize

        self._valid_classes = [name for name in classes if name not in void]
        self._ignore_index = ignore_index

        self.valid_idx = [classes.index(c) for c in self._valid_classes]
        self.void_idx = [classes.index(c) for c in void]

        log.info(f"Valid indxs: {self.valid_idx}")
        log.info(f"Void indxs: {self.void_idx}")
        log.info(f"Ignore index: {self.ignore_index}")

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
        data = {"lidar": ["points"], "lidar_segmentation": ["lidar_segmentation"]}

        self.train_ds = AKIDataset(
            data,
            splits=["training"],
            scenario=self.scenario,
            datasets=self.datasets,
            itersize=self.itersize,
            orderby=self.order_by,
            limit=self.limit,
            dbtype=self.dbtype,
            shuffle=True
        )

        self.val_ds = AKIDataset(
            data,
            splits=["validation"],
            scenario=self.scenario,
            datasets=self.datasets,
            itersize=self.itersize,
            dbtype=self.dbtype,
        )

        self.test_ds = AKIDataset(
            data,
            splits=["testing"],
            scenario=self.scenario,
            datasets=self.datasets,
            itersize=self.itersize,
            dbtype=self.dbtype,
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

        def pad_batch(pcs: list[torch.Tensor], lbls: list[torch.Tensor], pad_value=0) -> torch.Tensor:
            max_points = max(pc.shape[0] for pc in pcs)
            padded_point_clouds = []
            padded_labels = []

            for pc, lbl in zip(pcs, lbls):
                if pc.shape[0] < max_points:
                    pc_padded = torch.cat([pc,
                                           torch.full((max_points - pc.shape[0], pc.shape[1]), pad_value)], dim=0)
                    lbl_padded = torch.cat([lbl,
                                            torch.full([max_points - lbl.shape[0]], self.ignore_index)], dim=0)
                    padded_point_clouds.append(pc_padded)
                    padded_labels.append(lbl_padded)
                else:
                    padded_point_clouds.append(pc)
                    padded_labels.append(lbl)                    

            return torch.stack(padded_point_clouds), torch.stack(padded_labels)
        
        point_clouds = [elem[0] for elem in batch]
        labels = [elem[1] for elem in batch]
        
        point_clouds_padded, labels_padded = pad_batch(point_clouds, labels)

        for void_lbl in self.void_idx:
            labels_padded[labels_padded == void_lbl] = self.ignore_index       

        return point_clouds_padded, labels_padded

