import logging
import torch
import os

import lightning as L

from typing import List, Optional
from torch.utils.data import DataLoader
from data_modules.data_module_utils import runs_per_epoch, get_label_distribution, elems_in_dataset

from akiset import AKIDataset

log = logging.getLogger("rich")


class SemanticLidarSegmentationDownsampledCloudsDataModule(L.LightningDataModule):
    def __init__(
        self,
        scenario: str = "all",
        datasets: List[str] = ["all"],
        batch_size: int = 3,
        train_batch_size: Optional[int] = None,
        val_batch_size: Optional[int] = None,
        num_workers: int = 16,
        order_by: str = None,
        train_limit: int = None,
        val_limit: int = None,
        test_limit: int = None,
        downsampled_pointcloud_size: Optional[int] = None,
        only_use_downsampled_clouds_for_training = False,
        classes: Optional[List[str]] = None,
        void: Optional[List[str]] = None,
        ignore_index: Optional[int] = 255,
        log_distributions=False,
        dbtype: str = "psycopg@ants"
    ) -> None:
        super().__init__()

        self.dbtype = dbtype

        if train_batch_size and val_batch_size:
            log.info(f"Both train and val batch size provided using them: {train_batch_size}, {val_batch_size}")
            self.train_batch_size = train_batch_size
            self.val_batch_size = val_batch_size
        else:
            log.info(f"Not both train ({train_batch_size}) and val batch size ({val_batch_size}) provided using general batch size for both of them: {batch_size}")
            self.train_batch_size = batch_size
            self.val_batch_size = batch_size

        self.scenario = scenario
        self.datasets = datasets
        self.order_by = order_by
        self.train_limit = train_limit
        self.val_limit = val_limit
        self.test_limit = test_limit

        self.downsampled_pointcloud_size = downsampled_pointcloud_size
        self.only_use_downsampled_clouds_for_training = only_use_downsampled_clouds_for_training

        self.batch_size = batch_size
        self.num_workers = num_workers

        self._valid_classes = [name for name in classes if name not in void]
        self._ignore_index = ignore_index

        self.valid_idx = [classes.index(c) for c in self._valid_classes]
        self.void_idx = [classes.index(c) for c in void]

        self.log_distributions = log_distributions

        log.info(f"Valid indxs: {self.valid_idx}")
        log.info(f"Void indxs: {self.void_idx}")
        log.info(f"Ignore index: {self.ignore_index}")
        log.info(f"Downsampled pointcloud size: {self.downsampled_pointcloud_size}")
        log.info(f"Scenario: {self.scenario}")
        log.info(f"Batch size: {self.batch_size}")
        log.info(f"Train batch size: {self.train_batch_size}")
        log.info(f"Val batch size: {self.val_batch_size}")
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
        data = {"lidar":                ["points"],
                "lidar_segmentation":   ["lidar_segmentation"],
                "weather":              ["weather"]
                }
        
        data_downsampled = {"lidar":                [f"points_downsampled_{self.downsampled_pointcloud_size // 1000}k"],
                            "lidar_segmentation":   [f"lidar_segmentation_downsampled_{self.downsampled_pointcloud_size // 1000}k"],
                            "weather":              ["weather"]
                            }


        self.train_ds = AKIDataset(
            data_downsampled if self.downsampled_pointcloud_size else data,
            splits=["training"],
            scenario=self.scenario,
            datasets=self.datasets,
            orderby=self.order_by,
            limit=self.train_limit,
            dbtype=self.dbtype,
            shuffle=True
        )

        self.val_ds = AKIDataset(
            data_downsampled if self.downsampled_pointcloud_size and not self.only_use_downsampled_clouds_for_training else data,
            splits=["validation"],
            orderby=self.order_by,
            scenario=self.scenario,
            datasets=self.datasets,
            limit=self.val_limit,
            dbtype=self.dbtype
        )

        self.test_ds = AKIDataset(
            data_downsampled if self.downsampled_pointcloud_size and not self.only_use_downsampled_clouds_for_training else data,
            splits=["testing"],
            orderby=self.order_by,
            scenario=self.scenario,
            datasets=self.datasets,
            limit=self.test_limit,
            dbtype=self.dbtype,
        )

        if self.log_distributions:
            train_ds_amt_sunny_imgs, train_ds_amt_rainy_imgs = get_label_distribution(self.train_ds, 2)
            val_ds_amt_sunny_imgs, val_ds_amt_rainy_imgs = get_label_distribution(self.val_ds, 2)
            test_ds_amt_sunny_imgs, test_ds_amt_rainy_imgs = get_label_distribution(self.test_ds, 2)
            log.info(f"Train dataloader contains {train_ds_amt_sunny_imgs} point clouds in sunny situations, {train_ds_amt_rainy_imgs} point clouds in rainy situations, {elems_in_dataset(self.train_ds.count, self.train_limit)} total elements and yields {runs_per_epoch(self.train_ds.count, self.batch_size, self.train_limit)} batches per epoch.")
            log.info(f"Validation dataloader contains {val_ds_amt_sunny_imgs} point clouds in sunny situations, {val_ds_amt_rainy_imgs} point clouds in rainy situations, {elems_in_dataset(self.val_ds.count, self.val_limit)} total elements and yields {runs_per_epoch(self.val_ds.count, self.batch_size, self.val_limit)} batches per epoch.")
            log.info(f"Test dataloader contains {test_ds_amt_sunny_imgs} point clouds in sunny situations, {test_ds_amt_rainy_imgs} point clouds in rainy situations, {elems_in_dataset(self.test_ds.count, self.test_limit)} total elements and yields {runs_per_epoch(self.test_ds.count, self.batch_size, self.test_limit)} batches per epoch.")
        else:
            log.info(f"Train dataloader contains {elems_in_dataset(self.train_ds.count, self.train_limit)} elements and yields {runs_per_epoch(self.train_ds.count, self.train_batch_size, self.train_limit)} batches per epoch.")
            log.info(f"Validation dataloader contains {elems_in_dataset(self.val_ds.count, self.val_limit)} elements and yields {runs_per_epoch(self.val_ds.count, self.val_batch_size, self.val_limit)} batches per epoch.")
            log.info(f"Test dataloader contains {elems_in_dataset(self.test_ds.count, self.test_limit)} elements and yields {runs_per_epoch(self.test_ds.count, self.val_batch_size, self.test_limit)} batches per epoch.")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._prepare_batch
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            collate_fn=self._prepare_batch
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            collate_fn=self._prepare_batch
        )

    def _prepare_batch(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        def pad_batch(pcs: list[torch.Tensor], lbls: list[torch.Tensor], pad_value=0) -> torch.Tensor:
            max_points = max(pc.shape[0] for pc in pcs)
            padded_point_clouds = []
            padded_labels = []

            for pc, lbl in zip(pcs, lbls):
                #N Normalize to unit cube
                min_coords = pc.min(dim=0).values
                max_coords = pc.max(dim=0).values
                pc = (pc - min_coords) / (max_coords - min_coords).clamp(min=1e-6)

                # Pad point cloud and labels to largest element of the batch
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

