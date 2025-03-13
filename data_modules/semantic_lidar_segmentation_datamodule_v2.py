import logging
import torch
import json

import matplotlib.pyplot as plt
import lightning as L
import numpy as np

from typing import List, Optional
from torch.utils.data import DataLoader
from data_modules.data_module_utils import runs_per_epoch, randomly_crop
from torchvision.transforms.functional import to_pil_image

from akiset import AKIDataset
from akiset.utils.transform import project_vehicle_to_image_waymo

log = logging.getLogger("rich")


class SemanticLidarSegmentationDataModule(L.LightningDataModule):
    def __init__(
        self,
        scenario: str = "all",
        datasets: List[str] = ["all"],
        batch_size: int = 3,
        num_workers: int = 1,
        order_by: str = None,
        train_limit: int = None,
        val_limit: int = None,
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

        self.downsampled_pointcloud_size = downsampled_pointcloud_size

        self.batch_size = batch_size
        self.num_workers = num_workers

        self._valid_classes = [name for name in classes if name not in void]
        self._ignore_index = ignore_index

        self.valid_idx = [classes.index(c) for c in self._valid_classes]
        self.void_idx = [classes.index(c) for c in void]

        self.crop_size = (886, 1600)

        self.prepared_elems_counter = 0

        log.info(f"Valid indxs: {self.valid_idx}")
        log.info(f"Void indxs: {self.void_idx}")
        log.info(f"Ignore index: {self.ignore_index}")
        log.info(f"Downsampled pointcloud size: {self.downsampled_pointcloud_size}")
        log.info(f"Scenario: {self.scenario}")
        log.info(f"Batch size: {self.batch_size}")
        log.info(f"Num workers: {self.num_workers}")
        log.info(f"Order by: {self.order_by}")
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
        data = {
            "camera": ["image", "camera_id", "camera_parameters", "camera_vehicle_pose"],
            "camera_segmentation": ["camera_segmentation"],
            "lidar": ["points", "lidar_id", "lidar_parameters", "lidar_vehicle_pose"],
            "lidar_segmentation": ["lidar_segmentation"]
        }

        self.train_ds = AKIDataset(
            data,
            splits=["training"],
            scenario=self.scenario,
            datasets=self.datasets,
            orderby=self.order_by,
            limit=self.train_limit,
            dbtype=self.dbtype,
            shuffle=False
        )

        self.val_ds = AKIDataset(
            data,
            splits=["validation"],
            scenario=self.scenario,
            datasets=self.datasets,
            limit=self.val_limit,
            dbtype=self.dbtype,
            shuffle=True
        )

        self.test_ds = AKIDataset(
            data,
            splits=["testing"],
            scenario=self.scenario,
            datasets=self.datasets,
            dbtype=self.dbtype,
        )

        log.info(f"Train dataloader contains {self.train_ds.count} elements. It yields {runs_per_epoch(self.train_ds.count, self.batch_size, self.train_limit)} runs per epoch (batch size is {self.batch_size})")
        log.info(f"Validation dataloader contains {self.val_ds.count} elements. It yields {runs_per_epoch(self.val_ds.count, self.batch_size, self.val_limit)} runs per epoch (batch size is {self.batch_size})")
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
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self._prepare_batch
        )

    def _prepare_batch(self, batch) -> tuple[torch.Tensor, torch.Tensor]:

        def pad_batch(pcs_batch: list[torch.Tensor], lbls_batch: list[torch.Tensor], pad_value=0) -> torch.Tensor:
            for pc, lbls in zip(pcs_batch, lbls_batch):
                assert pc.shape[0] == lbls.shape[0], f"Amount of points in point cloud doesn't match amount its labeels: {pc.shape[0]} vs. {lbls.shape[0]}"
            
            max_points = max(pc.shape[0] for pc in pcs_batch)
            padded_point_clouds = []
            padded_labels = []

            for pc, lbls in zip(pcs_batch, lbls_batch):
                if pc.shape[0] < max_points:
                    pc_padded = torch.cat([pc,
                                           torch.full((max_points - pc.shape[0], pc.shape[1]), pad_value)], dim=0)
                    lbls_padded = torch.cat([lbls,
                                            torch.full([max_points - lbls.shape[0]], self.ignore_index)], dim=0)
                    padded_point_clouds.append(pc_padded)
                    padded_labels.append(lbls_padded)
                else:
                    padded_point_clouds.append(pc)
                    padded_labels.append(lbls)                    

            return torch.stack(padded_point_clouds), torch.stack(padded_labels)

        point_clouds_to_pad = []
        labels_to_pad = []

        for elem in batch:
            image = elem[0]
            camera_id = elem[1]
            camera_params = json.loads(elem[2])
            camera_vehicle_pose = json.loads(elem[3])
            segmentation_mask = elem[4]
            point_cloud = elem[5].numpy()
            lidar_id = elem[6]
            lidar_params = json.loads(elem[7])
            lidar_vehicle_pose = json.loads(elem[8])
            lidar_segmentation = elem[9]

            if lidar_id == "waymo_1":
                #if camera_id == lidar_id:
                image_heigth = image.shape[1]
                image_width = image.shape[2]
                points_projected = project_vehicle_to_image_waymo(lidar_vehicle_pose, camera_params, image_width, image_heigth, point_cloud)
                image_cropped, segmentation_mask_cropped, i, j, h, w = randomly_crop(image, segmentation_mask, self.crop_size)

                image_pil = to_pil_image(image)
                image_pil_cropped = to_pil_image(image_cropped)
                points_inside_crop_mask = points_projected[:, 2].astype(bool)
                points_inside_crop_mask = np.logical_and(points_inside_crop_mask, points_projected[:, 0] >= j)
                points_inside_crop_mask = np.logical_and(points_inside_crop_mask, points_projected[:, 0] < j+w)
                points_inside_crop_mask = np.logical_and(points_inside_crop_mask, points_projected[:, 1] >= i)
                points_inside_crop_mask = np.logical_and(points_inside_crop_mask, points_projected[:, 1] < i+h)

                log.info(f"i: {i}, j: {j}, h: {h}, w: {w}, i+h: {i+h}, j+w: {j+w}")

                # Create a combined tensor where each element consists of 5 values: x,y,z coordinates and u,v pixel coordinates
                points = torch.hstack((torch.Tensor(point_cloud), torch.Tensor(points_projected[:,:2]).to(torch.int)))


                # Only get all points inside the cropped area by applying the mask
                points_inside_crop = torch.Tensor(points)[torch.BoolTensor(points_inside_crop_mask)]
                
                log.info(f"u_min: {points_inside_crop[:, 3].min()}, u_max: {points_inside_crop[:, 3].max()}, v_min: {points_inside_crop[:, 4].min()}, v_max: {points_inside_crop[:, 4].max()}")


                labels_inside_crop = torch.Tensor([segmentation_mask[int(point[4].item()), int(point[3].item())] for point in points_inside_crop])
                labels_np = np.array(labels_inside_crop)
                
                point_clouds_to_pad.append(points_inside_crop)
                labels_to_pad.append(labels_inside_crop)

                # Visualization: full image with all points
                fig, axs = plt.subplots(1, 2, figsize=(12, 6))
                axs[0].imshow(image_pil)
                axs[0].scatter(points_projected[:, 0], points_projected[:, 1], c='blue', s=1, alpha=0.3)
                axs[0].set_title("Full Image with All Projected Points")
                axs[0].axis("off")
                
                # Visualization: full image with crop region and points inside crop
                axs[1].imshow(image_pil)
                rect = plt.Rectangle((j, i), w, h, linewidth=2, edgecolor='red', facecolor='none')
                axs[1].add_patch(rect)
                axs[1].scatter(points_inside_crop[:, 3], points_inside_crop[:, 4], c=labels_np, s=1, alpha=0.7)
                axs[1].set_title("Cropped Region with Points")
                axs[1].axis("off")

                plt.tight_layout()
                plt.savefig(f"imgs/pcls_imgs_projections_dm/projection_comparison_{self.prepared_elems_counter}.png")
                plt.close()

                self.prepared_elems_counter += 1
            """
            fig, ax = plt.subplots(1, 1)
            #ax.imshow(image_pil)
            x_vals = points_inside_crop[:, 0]
            y_vals = points_inside_crop[:, 1]
            ax.scatter(x_vals, y_vals, c=labels_np, s=2, alpha=0.5)
            ax.axis("off")
            plt.savefig("projection.png")"
            """
                
            #else:
            #    log.info(f"Camera and lidar data incompatible: {camera_id} != {lidar_id}")
        
        point_clouds_padded, labels_padded = pad_batch(point_clouds_to_pad, labels_to_pad)

        # Mask void_lbls with value ignore_index
        for void_lbl in self.void_idx:
            labels_padded[labels_padded == void_lbl] = self.ignore_index       

        log.info(f"Result of prepare_batch: {point_clouds_padded.shape}, {labels_padded.shape}")

        return point_clouds_padded, labels_padded

