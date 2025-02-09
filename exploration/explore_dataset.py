import torch
import uuid
import math
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torchvision as tv
import torch.nn.functional as F

from torch.utils.data import DataLoader
from rich.progress import track
from akiset import AKIDataset
from data_modules.semantic_image_segmentation_datamodule import SemanticImageSegmentationDataModule
#from utils import prepare_batch, add_segmentation_mask_to_img, divide_batch_of_tensors

log = logging.getLogger("rich")

def overlay_mask_and_export_img(img, label, colors, export_dir, contains_masked_pixels = False):
    os.makedirs(export_dir, exist_ok=True)

    if contains_masked_pixels:
            label[label == 255] = len(colors)
            colors.append((0, 0, 0))
    
    # Convert mask to one-hot format for drawing segmentation masks
    # Permuting because one hot creates [H,W,C] format, but [C,H,W] format is required
    # .bool() because tv.utils.draw_segmentation_masks expects bool tensors
    mask_one_hot = F.one_hot(label, num_classes=len(colors)).permute(2, 0, 1).bool()

    # Convert mask to RGB
    mask_rgb = torch.zeros(3, label.shape[0], label.shape[1], dtype=torch.uint8)  # Create empty RGB mask
    for label_id, color in enumerate(colors):
        mask_rgb[:, label == label_id] = torch.tensor(color, dtype=torch.uint8).view(3, 1)  # Apply color


    # Draw segmentation mask
    img_with_mask = tv.utils.draw_segmentation_masks(img, mask_one_hot, colors=colors, alpha=0.5)

    # Plot images
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img.permute(1, 2, 0))  # Convert from C, H, W to H, W, C
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(mask_rgb.permute(1,2,0))
    axes[1].set_title("Segmentation Mask")
    axes[1].axis("off")

    axes[2].imshow(img_with_mask.permute(1, 2, 0))  # Convert from C, H, W to H, W, C
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(export_dir, f"{uuid.uuid4()}.png"))
    plt.close(fig)

def explore_aki_dataset(aki_dataset: AKIDataset, colors: list[tuple], max_imgs_to_export = 10, export_dir: str = "imgs/explore_aki_dataset/"):
    """Exports images from the aki_dataset with the semantic segmentation mask overlayed

    Args:
        aki_dataset (AKIDataset): The AkiDataset to export images from.
        colors (list[tuple]): A list of RGB colors (3 tuples in range [0, 255]), length has to be equal to the
        amount of classes of the dataset.
        export_dir (str, optional): Directory to export images to. Defaults to "imgs/explore_aki_dataset/".
    """
    log.info(f"Dataset contains {aki_dataset.count} images")
    
    os.makedirs(export_dir, exist_ok=True)
    
    imgs_exported = 0

    for element in aki_dataset:
        if imgs_exported >= max_imgs_to_export:
            return
        
        img = element[0]
        label = element[1]

        overlay_mask_and_export_img(img, label, colors, export_dir=export_dir)

        imgs_exported += 1


def explore_sis_dm_train_dataloader(dm: SemanticImageSegmentationDataModule, colors: list[tuple], max_imgs_to_export: int = 10,
                                    export_dir = "imgs/explore_sis_dm_train_dataloader/"): 
    """
    dm = SemanticImageSegmentationDataModule(scenario=scenario,
                                            datasets=datasets,
                                            classes=classes,
                                           void=void_classes)
    dm.setup()
    """
    imgs_exported = 0
    for batch in dm.train_dataloader():
        
        imgs, labels = batch
        
        for img, label in zip(imgs, labels):
            if imgs_exported >= max_imgs_to_export:
                log.info(f"Exported {imgs_exported} of {max_imgs_to_export} -> Aborting")
                return
            overlay_mask_and_export_img(img, label, colors, export_dir, contains_masked_pixels=True)
            imgs_exported += 1


"""
def explore_imgs_in_different_weather_conditions():

    imgs_per_slide = 4 # This should be a square number for the grid ordering to play out nicely
    total_imgs = 40 # This should be a multiple of imgs_per_slide
    n = int(math.sqrt(imgs_per_slide))

    data_dict = {"camera": ["image"], "weather": ["weather"]}

    ds_day_light_rain = AKIDataset(data=data_dict,
                        datasets=["all"],
                        scenario="daylightrain")
    ds_day_moderate_rain = AKIDataset(data=data_dict,
                        datasets=["all"],
                        scenario="daymoderaterain")
    ds_day_heavy_rain = AKIDataset(data=data_dict,
                        datasets=["all"],
                        scenario="dayheavyrain")

    dl_day_light_rain = DataLoader(ds_day_light_rain, batch_size=imgs_per_slide)
    dl_day_moderate_rain = DataLoader(ds_day_moderate_rain, batch_size=imgs_per_slide)
    dl_day_heavy_rain = DataLoader(ds_day_heavy_rain, batch_size=imgs_per_slide)

    folder_names = {dl_day_light_rain: "light_rain/",
                    dl_day_moderate_rain: "moderate_rain/",
                    dl_day_heavy_rain: "heavy_rain/"}

    for dl in [dl_day_light_rain, dl_day_moderate_rain, dl_day_heavy_rain]:
        imgs_exported = 0
        for batch in dl:
            if imgs_exported >= total_imgs:
                break

            imgs = batch[0]
            labels = batch[1]

            fig, axs = plt.subplots(n, n)

            for i in range(0, imgs_per_slide):
                img = imgs[i]
                ax = axs[i // n, i % n]
                label = labels[i]
                ax.axis("off")
                #log.info(f"img dimensions: {img.shape}")
                ax.imshow(img.permute(1,2,0))
                ax.text(0,0, label)
                #ax.imshow(add_segmentation_mask_to_img(img, label).permute(1,2,0))
            plt.tight_layout(h_pad=0.3, w_pad=0.3)
            plt.savefig(f"imgs/{folder_names[dl]}/{str(uuid.uuid4())}", dpi=300)
            plt.close(fig)

            imgs_exported += imgs_per_slide

def export_divided_imgs():
    aki_ds = AKIDataset(data={"camera": ["image"], "camera_segmentation": ["camera_segmentation"]})

    log.info(f"Number of elements in dataset: {aki_ds.count}")

    batch_size = 5

    data_loader = DataLoader(aki_ds, batch_size=batch_size, collate_fn=prepare_batch)

    amount_of_batches = aki_ds.count // batch_size if aki_ds.count % batch_size == 0 else aki_ds.count // batch_size + 1

    log.info(f"Amount of batches in data loader: {amount_of_batches}")

    rows = 5
    cols = 4

    for batch in track(data_loader, total=amount_of_batches):
        inputs, labels = batch

        divided_imgs = divide_batch_of_tensors(inputs, rows, cols)

        for x in range(batch_size):
            img = inputs[x]
            divided_img = divided_imgs[x]

            fig, axs = plt.subplots(rows, cols)
            for i in range(0, rows * cols):

                ax = axs[i // cols, i % cols]
                ax.axis("off")
                ax.imshow(divided_img[i].permute(1,2,0))
                #ax.imshow(add_segmentation_mask_to_img(img, label).permute(1,2,0))
            plt.tight_layout(h_pad=0.3, w_pad=0.3)
            plt.savefig(f"imgs/{str(uuid.uuid4())}")
            plt.close(fig)

            log.info(f"{x * batch_size} images divided and exported to file")
"""