import logging
import torch
import os

import lightning as L

from typing import List, Optional
from torch.utils.data import DataLoader
from data_modules.data_module_utils import runs_per_epoch, get_label_distribution, elems_in_dataset

from akiset import AKIDataset

log = logging.getLogger("rich")


class SemanticLidarSegmentationDataModule(L.LightningDataModule):
    def __init__(
        self,
        scenario: str = "all",
        datasets: List[str] = ["all"],
        batch_size: Optional[int] = 32,
        num_workers: int = 16,
        itersize: int = 1000,
        order_by: str = None,
        train_limit: int = None,
        val_limit: int = None,
        test_limit: int = None,
        downsampled_pointcloud_size: Optional[int] = None,
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
        self.train_limit = train_limit
        self.val_limit = val_limit
        self.test_limit = test_limit

        self.downsampled_pointcloud_size = downsampled_pointcloud_size

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
        log.info(f"Downsampled pointcloud size: {self.downsampled_pointcloud_size}")
        log.info(f"Scenario: {self.scenario}")
        log.info(f"Batch size: {self.batch_size}")
        log.info(f"Num workers: {self.num_workers}")
        log.info(f"Oder by: {self.order_by}")
        log.info(f"Train limit: {self.train_limit}")
        log.info(f"Val limit: {self.val_limit}")
        log.info(f"Datasets: {self.datasets}")

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
        data = {"lidar": ["points"],
                "lidar_segmentation": ["lidar_segmentation"],
                "weather": ["weather"]
                }

        if self.downsampled_pointcloud_size:
            data = {"lidar":  [f"points_downsampled_{self.downsampled_pointcloud_size % 1000}k"],
                    "lidar_segmentation": [f"lidar_segmentation_downsampled_{self.downsampled_pointcloud_size % 1000}k"],
                    "weather": ["weather"]
                    }


        self.train_ds = AKIDataset(
            data,
            splits=["training"],
            scenario=self.scenario,
            datasets=self.datasets,
            itersize=self.itersize,
            orderby=self.order_by,
            limit=self.train_limit,
            dbtype=self.dbtype,
            shuffle=True
        )

        self.val_ds = AKIDataset(
            data,
            splits=["validation"],
            scenario=self.scenario,
            datasets=self.datasets,
            limit=self.val_limit,
            itersize=self.itersize,
            dbtype=self.dbtype,
        )

        self.test_ds = AKIDataset(
            data,
            splits=["testing"],
            scenario=self.scenario,
            datasets=self.datasets,
            itersize=self.itersize,
            limit=self.test_limit,
            dbtype=self.dbtype,
        )

        #train_ds_amt_sunny_imgs, train_ds_amt_rainy_imgs = get_label_distribution(self.train_ds, 2)
        #log.info(f"Train distribution obtained")
        #val_ds_amt_sunny_imgs, val_ds_amt_rainy_imgs = get_label_distribution(self.val_ds, 2)
        #log.info(f"Validation distribution obtained")
        #test_ds_amt_sunny_imgs, test_ds_amt_rainy_imgs = get_label_distribution(self.test_ds, 2)
        #log.info(f"Test distribution obtained")

        #log.info(f"Train dataloader contains {train_ds_amt_sunny_imgs} point clouds in sunny situations, {train_ds_amt_rainy_imgs} point clouds in rainy situations, {elems_in_dataloader(self.train_ds.count, self.train_limit)} total elements and yields {runs_per_epoch(self.train_ds.count, self.batch_size, self.train_limit)} batches per epoch.")
        #log.info(f"Validation dataloader contains {val_ds_amt_sunny_imgs} point clouds in sunny situations, {val_ds_amt_rainy_imgs} point clouds in rainy situations, {elems_in_dataloader(self.val_ds.count, self.val_limit)} total elements and yields {runs_per_epoch(self.val_ds.count, self.batch_size, self.val_limit)} batches per epoch.")
        #log.info(f"Test dataloader contains {test_ds_amt_sunny_imgs} point clouds in sunny situations, {test_ds_amt_rainy_imgs} point clouds in rainy situations, {elems_in_dataloader(self.test_ds.count, self.test_limit)} total elements and yields {runs_per_epoch(self.test_ds.count, self.batch_size, self.test_limit)} batches per epoch.")

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
        
        def pad_batch(pcs: list[torch.Tensor], lbls: list[torch.Tensor], pad_value=0) -> torch.Tensor:
            max_points = max(pc.shape[0] for pc in pcs)
            padded_point_clouds = []
            padded_labels = []

            for pc, lbl in zip(pcs, lbls):
                # Normalize to unit cube [0, 1]
                min_coords = pc.min(dim=0).values
                max_coords = pc.max(dim=0).values
                log.info(f"Before normalization — Min: {min_coords.tolist()}, Max: {max_coords.tolist()}")

                pc = (pc - min_coords) / (max_coords - min_coords).clamp(min=1e-6)

                # Log min-max after normalization
                min_coords_after = pc.min(dim=0).values
                max_coords_after = pc.max(dim=0).values
                log.info(f"After normalization — Min: {min_coords_after.tolist()}, Max: {max_coords_after.tolist()}")



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

