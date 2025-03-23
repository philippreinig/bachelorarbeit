import logging
import torch
import json

import matplotlib.pyplot as plt
import lightning as L
import numpy as np
import uuid

from typing import List, Optional
from torch.utils.data import DataLoader
from data_modules.data_module_utils import runs_per_epoch, randomly_crop, weather_condition2numeric, elems_in_dataset
from torchvision.transforms.functional import to_pil_image

from akiset import AKIDataset
from akiset.utils.transform import project_vehicle_to_image_waymo

log = logging.getLogger("rich")


class UnifiedDataModule(L.LightningDataModule):
    def __init__(
        self,
        scenario: str = "all",
        datasets: List[str] = ["all"],
        batch_size: int = 32,
        num_workers: int = 1,
        order_by: str = None,
        train_limit: int = None,
        val_limit: int = None,
        test_limit: int = None,
        downsampled_pointcloud_size: Optional[int] = None,
        shuffle: bool = False,
        crop_size: tuple[int, int] = (886, 1600),
        grid_cells: tuple[int, int] = (1,1),
        classes: Optional[List[str]] = None,
        void: Optional[List[str]] = None,
        ignore_index: Optional[int] = 255,
        dbtype: str = "psycopg@ants"
    ) -> None:
        super().__init__()

        if downsampled_pointcloud_size not in [None, 4000, 8000, 16000]:
            raise ValueError(f"Invalid downsampled point cloud size. Must be one of: 4000, 8000, 16000, but is: {downsampled_pointcloud_size}")

        if crop_size[0] % grid_cells[0] != 0 or crop_size[1] % grid_cells[1] != 0:
            raise ValueError(f"Crop size must be divisible by grid dimensions, which is not the case for {crop_size} and {grid_cells}")

        self.dbtype = dbtype

        self.scenario = scenario
        self.datasets = datasets
        self.order_by = order_by
        self.train_limit = train_limit
        self.val_limit = val_limit
        self.test_limit = test_limit
        self.shuffle = shuffle
        self.downsampled_pointcloud_size = downsampled_pointcloud_size

        self.batch_size = batch_size
        self.num_workers = num_workers

        self._valid_classes = [name for name in classes if name not in void]
        self._ignore_index = ignore_index

        self.valid_idx = [classes.index(c) for c in self._valid_classes]
        self.void_idx = [classes.index(c) for c in void]

        self.crop_size = crop_size
        self.grid_cells = grid_cells

        self.prepared_elems = 0

        log.info(f"Valid indxs: {self.valid_idx}")
        log.info(f"Void indxs: {self.void_idx}")
        log.info(f"Ignore index: {self.ignore_index}")
        log.info(f"Downsampled pointcloud size: {self.downsampled_pointcloud_size}")
        log.info(f"Scenario: {self.scenario}")
        log.info(f"Batch size: {self.batch_size}")
        log.info(f"Num workers: {self.num_workers}")
        log.info(f"Shuffle: {self.shuffle}")
        log.info(f"Order by: {self.order_by}")
        log.info(f"Train limit: {self.train_limit}")
        log.info(f"Val limit: {self.val_limit}")
        log.info(f"Datasets: {self.datasets}")
        log.info(f"Crop size: {self.crop_size}")
        log.info(f"Grid cells: {self.grid_cells}")

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
        points_col_name = f"points{f'_downsampled_{self.downsampled_pointcloud_size // 1000}k' if self.downsampled_pointcloud_size else ''}"
        data = {
            "camera": ["image", "camera_id", "camera_parameters", "camera_vehicle_pose"],
            "camera_segmentation": ["camera_segmentation"],
            "lidar": [points_col_name, "lidar_id", "lidar_parameters", "lidar_vehicle_pose"],
            "weather": ["weather"]
        }

        log.info(f"Columns for datasets are: {data}")

        self.train_ds = AKIDataset(
            data,
            splits=["training"],
            scenario=self.scenario,
            datasets=self.datasets,
            orderby=self.order_by,
            limit=self.train_limit,
            dbtype=self.dbtype,
            shuffle=self.shuffle
        )

        self.val_ds = AKIDataset(
            data,
            splits=["validation"],
            scenario=self.scenario,
            datasets=self.datasets,
            limit=self.val_limit,
            dbtype=self.dbtype,
            shuffle=self.shuffle
        )

        self.test_ds = AKIDataset(
            data,
            splits=["testing"],
            scenario=self.scenario,
            datasets=self.datasets,
            dbtype=self.dbtype,
            limit=self.test_limit,
            shuffle=self.shuffle
        )

        log.info(f"Train dataloader contains {elems_in_dataset(self.train_ds.count, self.train_limit)} elements. It yields {runs_per_epoch(self.train_ds.count, self.batch_size, self.train_limit)} runs per epoch (batch size is {self.batch_size})")
        log.info(f"Validation dataloader contains {elems_in_dataset(self.val_ds.count, self.val_limit)} elements. It yields {runs_per_epoch(self.val_ds.count, self.batch_size, self.val_limit)} runs per epoch (batch size is {self.batch_size})")
        log.info(f"Test dataloader contains {elems_in_dataset(self.test_ds.count, self.test_limit)} elements. It yields {runs_per_epoch(self.test_ds.count, self.batch_size, self.test_limit)} runs per epoch (batch size is {self.batch_size})")

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

        def pad_pc_batch(pcs_batch: list[torch.Tensor], lbls_batch: list[torch.Tensor], pad_value=0) -> torch.Tensor:
            for pc, lbls in zip(pcs_batch, lbls_batch):
                assert pc.shape[0] == lbls.shape[0], f"Amount of points in point cloud doesn't match amount its labeels: {pc.shape[0]} vs. {lbls.shape[0]}"
            
            max_points = max(pc.shape[0] for pc in pcs_batch) if len(pcs_batch) > 0 else 0
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

            return (torch.stack(padded_point_clouds), torch.stack(padded_labels)) if len(padded_point_clouds) else (torch.empty(0,3), torch.empty(0,3))

        images = []
        image_seg_masks = []
        point_clouds_to_pad = []
        pc_labels_to_pad = []
        point_pixel_projections = []
        weather_conditions = []
        fusable_pixels_masks = []

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
            weather_condition = elem[9]

            if lidar_id == "waymo_1":
                image_heigth = image.shape[1]
                image_width = image.shape[2]
                points_projected = project_vehicle_to_image_waymo(lidar_vehicle_pose, camera_params, image_width, image_heigth, point_cloud)
                image_cropped, segmentation_mask_cropped, i, j, h, w = randomly_crop(image, segmentation_mask, self.crop_size)

                
                points_inside_crop_mask = points_projected[:, 2].astype(bool)
                points_inside_crop_mask = np.logical_and(points_inside_crop_mask, points_projected[:, 0] >= j)
                points_inside_crop_mask = np.logical_and(points_inside_crop_mask, points_projected[:, 0] < j+w)
                points_inside_crop_mask = np.logical_and(points_inside_crop_mask, points_projected[:, 1] >= i)
                points_inside_crop_mask = np.logical_and(points_inside_crop_mask, points_projected[:, 1] < i+h)

                #log.info(f"i: {i}, j: {j}, h: {h}, w: {w}, i+h: {i+h}, j+w: {j+w}")

                # Create a combined tensor where each element consists of 5 values: x,y,z coordinates and u,v pixel coordinates
                points = torch.hstack((torch.tensor(point_cloud), torch.tensor(points_projected[:,:2]).to(torch.int)))

                # Only get all points inside the cropped area by applying the mask
                points_inside_crop = points[torch.tensor(points_inside_crop_mask, dtype=torch.bool)]

                # Shift projected pixel indices of points to the cropped region
                points_shifted_to_crop = points_inside_crop.clone()
                points_shifted_to_crop[:, 3:5] -= torch.tensor([j, i])
                
                labels_inside_crop = torch.tensor([segmentation_mask[int(point[4].item()), int(point[3].item())] for point in points_inside_crop])

                cell_height = self.crop_size[0] // self.grid_cells[0]
                cell_width = self.crop_size[1] // self.grid_cells[1]

                for grid_row in range(self.grid_cells[0]):
                    for grid_col in range(self.grid_cells[1]):
                        u_start, u_end = grid_row * cell_height, (grid_row + 1) * cell_height
                        v_start, v_end = grid_col * cell_width, (grid_col + 1) * cell_width

                        img_cell = image_cropped[:, u_start:u_end, v_start:v_end]
                        img_seg_mask_cell = segmentation_mask_cropped[u_start:u_end, v_start:v_end]
                        
                        points_inside_cell_mask = torch.ones(points_shifted_to_crop.shape[0], dtype=torch.bool)
                        points_inside_cell_mask = torch.logical_and(points_inside_cell_mask, points_shifted_to_crop[:, 3] >= v_start)
                        points_inside_cell_mask = torch.logical_and(points_inside_cell_mask, points_shifted_to_crop[:, 3] < v_end)
                        points_inside_cell_mask = torch.logical_and(points_inside_cell_mask, points_shifted_to_crop[:, 4] >= u_start)
                        points_inside_cell_mask = torch.logical_and(points_inside_cell_mask, points_shifted_to_crop[:, 4] < u_end)
                                           
                        points_in_cell = points_shifted_to_crop[points_inside_cell_mask]
                        points_in_and_shifted_to_cell = points_in_cell.clone()
                        points_in_and_shifted_to_cell[:, 3:5] -= torch.tensor([v_start, u_start])

                        points_in_cell_labels = labels_inside_crop[points_inside_cell_mask]

                        fusable_pixels_cell_mask = torch.zeros((cell_height, cell_width), dtype=torch.bool)
                        u_coords_points_in_cell = points_in_and_shifted_to_cell[:, 3].int()
                        v_coords_points_in_cell = points_in_and_shifted_to_cell[:, 4].int()
                        fusable_pixels_cell_mask[v_coords_points_in_cell, u_coords_points_in_cell] = True

                        if fusable_pixels_cell_mask.sum() >= 1:
                            point_clouds_to_pad.append(points_in_and_shifted_to_cell[:, :3])
                            pc_labels_to_pad.append(points_in_cell_labels)
                            point_pixel_projections.append(points_in_and_shifted_to_cell)
                            images.append(img_cell)
                            image_seg_masks.append(img_seg_mask_cell)
                            weather_conditions.append(torch.tensor(weather_condition2numeric(weather_condition), dtype=torch.int))
                            fusable_pixels_masks.append(fusable_pixels_cell_mask)

                        else:
                            log.info(f"Cell discarded because it doesn't contain any fusable pixels")

                # Visualization: full image with all points
                fig, axs = plt.subplots(1, 2, figsize=(12, 6))
                image_pil = to_pil_image(image)
                axs[0].imshow(image_pil)
                axs[0].scatter(points_projected[:, 0], points_projected[:, 1], c='blue', s=1, alpha=0.3)
                axs[0].set_title("Full Image with All Projected Points")
                axs[0].axis("off")
                
                # Visualization: full image with crop region and points inside crop
                axs[1].imshow(image_pil)
                rect = plt.Rectangle((j, i), w, h, linewidth=2, edgecolor='red', facecolor='none')
                axs[1].add_patch(rect)
                axs[1].scatter(points_inside_crop[:, 3], points_inside_crop[:, 4], c=np.array(labels_inside_crop), s=1, alpha=0.7)
                axs[1].set_title("Cropped Region with Points")
                axs[1].axis("off")

                plt.tight_layout()
                plt.savefig(f"imgs/pcls_imgs_projections_dm/projection_comparison_{uuid.uuid4()}.png")
                plt.close()

            #else:
                #log.warning(f"Element not projectable because lidar {lidar_id} != waymo_1")

        point_clouds_padded, pc_labels_padded = pad_pc_batch(point_clouds_to_pad, pc_labels_to_pad)

        # Mask void_lbls with value ignore_index
        for void_lbl in self.void_idx:
            pc_labels_padded[pc_labels_padded == void_lbl] = self.ignore_index       

        images_tensor = torch.stack(images)
        image_seg_masks_tensor = torch.stack(image_seg_masks)
        point_clouds_padded_tensor = point_clouds_padded
        pc_labels_tensor = pc_labels_padded
        weather_conditions_tensor = torch.stack(weather_conditions)
        fusable_pixels_tensor = torch.stack(fusable_pixels_masks)
        
        self.prepared_elems += len(images)
        log.info(f"Batch contains {len(images)} elements, total elements: {self.prepared_elems}")

        return images_tensor, image_seg_masks_tensor, point_clouds_padded_tensor, pc_labels_tensor, weather_conditions_tensor, fusable_pixels_tensor, point_pixel_projections

