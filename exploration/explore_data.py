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
from data_modules.weather_classification_datamodule import WeatherClassificationDataModule
from data_modules.semantic_lidar_segmentation_datamodule_v2 import SemanticLidarSegmentationDataModule
from utils.data_preparation import prepare_batch, divide_batch_of_tensors
from utils.misc import add_segmentation_mask_to_img
from utils.aki_labels import get_aki_label_names

log = logging.getLogger("rich")
plt.ion()

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
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].imshow(img.permute(1, 2, 0))  # Convert from C, H, W to H, W, C
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(mask_rgb.permute(1,2,0))
    axes[1].set_title("Segmentation Mask")
    axes[1].axis("off")

    #axes[2].imshow(img_with_mask.permute(1, 2, 0))  # Convert from C, H, W to H, W, C
    #axes[2].set_title("Overlay")
    #axes[2].axis("off")

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


def explore_weather_classification_train_dataloader(dm: WeatherClassificationDataModule):
    rain_labels = 0
    sunny_labels = 0
    total_elements = 0
    for batch in dm.train_dataloader():
        imgs, labels = batch
        amt_labels = labels.shape[0]
        #log.info(f"imgs shape: {imgs.shape}, labels shape: {labels.shape}")
        #log.info(f"amt labels is: {amt_labels}")
        log.info(f"Rain labels {torch.count_nonzero(labels)}, sunny labels: {amt_labels - torch.count_nonzero(labels)}")
        rain_labels += torch.count_nonzero(labels)
        sunny_labels += (amt_labels - torch.count_nonzero(labels))
        #log.info(f"rain labels: {rain_labels}, sunny labels: {sunny_labels}")
        total_elements += amt_labels

    log.info(f"Total amount of rain labels in train data loader: {rain_labels}")
    log.info(f"Total amount of sunny labels in train data loader: {sunny_labels}")
    log.info(f"Total amount of elements in train data loader: {total_elements}")

def explore_aki_ds_for_weather_classification():
    aki_ds = AKIDataset(data= {"camera": ["image"], "weather": ["weather"]},
                        splits=["training"],
                        datasets=["waymo"],
                        scenario="all",
                        orderby="weather",
                        limit=10)
    
    rain_labels = 0
    sunny_labels = 0
    total_elements = 0

    for elem in aki_ds:
        img, lbl = elem

        rain_labels += 1 if lbl == "rain" else 0
        sunny_labels += 1 if lbl == "sunny" else 0
        total_elements += 1

    log.info(f"Total amount of rain labels in train data loader: {rain_labels}")
    log.info(f"Total amount of sunny labels in train data loader: {sunny_labels}")
    log.info(f"Total amount of elements in train data loader: {total_elements}")

def explore_waymo_rain_images():
    imgs_per_slide = 36 # This should be a square number for the grid ordering to play out nicely
    total_imgs = 720 # This should be a multiple of imgs_per_slide
    n = int(math.sqrt(imgs_per_slide))

    wcdm = WeatherClassificationDataModule(order_by="weather",
                                           limit=total_imgs,
                                           datasets=["waymo"],
                                           batch_size=imgs_per_slide)
    wcdm.setup()

    imgs_exported = 0
    for batch in wcdm.train_dataloader():
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
        plt.savefig(f"imgs/waymo_rain/{str(uuid.uuid4())}", dpi=300)
        plt.close(fig)

        imgs_exported += imgs_per_slide

def waymo_sunny_vs_rainy_images():
    total_imgs = 200 #

    aki_ds_rainy = AKIDataset(data={"camera": ["image"], "weather": ["weather"]},
                                datasets=["waymo"],
                                limit = total_imgs/2,
                                shuffle=True,
                                scenario="rain")
    aki_ds__sunny = AKIDataset(data={"camera": ["image"], "weather": ["weather"]},
                                datasets=["waymo"],
                                scenario="sun",
                                shuffle=True,
                                limit = total_imgs/2)
    

    for elem in zip(aki_ds_rainy, aki_ds__sunny):
        print("elem loaded")

        fig, axs = plt.subplots(2)

        for i in range(2):
            img = elem[i][0]
            lbl = elem[i][1]
            ax = axs[i]
            ax.axis("off")
            ax.imshow(img.permute(1,2,0))
            ax.text(0,0, lbl)
        plt.tight_layout()
        plt.savefig(f"imgs/waymo_sunny_vs_rainy/{str(uuid.uuid4())}", dpi=300)
        plt.close(fig)

def calc_waymo_rainy_vs_sunny_image_stats():
    wcdm = WeatherClassificationDataModule(order_by="weather",
                                           limit=10_000,
                                           datasets=["waymo"],
                                           batch_size=32)
    wcdm.setup()

    # Initialize counters and accumulators
    sunny_mean = torch.zeros(3)
    sunny_M2 = torch.zeros(3)
    sunny_pixels = 0

    rainy_mean = torch.zeros(3)
    rainy_M2 = torch.zeros(3)
    rainy_pixels = 0

    for batch in track(wcdm.train_dataloader(), total=313):
        images, labels = batch  # images: (B, C, H, W), labels: (B)

        for img, lbl in zip(images, labels):
            img = img.view(3, -1)  # Flatten spatial dimensions
            batch_mean = img.mean(dim=1)
            batch_var = img.var(dim=1, unbiased=False)
            batch_pixels = img.shape[1]

            if lbl == 0:  # Sunny images
                delta = batch_mean - sunny_mean
                sunny_mean += delta * (batch_pixels / (sunny_pixels + batch_pixels))
                sunny_M2 += batch_var * batch_pixels + delta**2 * (sunny_pixels * batch_pixels) / (sunny_pixels + batch_pixels)
                sunny_pixels += batch_pixels

            else:  # Rainy images
                delta = batch_mean - rainy_mean
                rainy_mean += delta * (batch_pixels / (rainy_pixels + batch_pixels))
                rainy_M2 += batch_var * batch_pixels + delta**2 * (rainy_pixels * batch_pixels) / (rainy_pixels + batch_pixels)
                rainy_pixels += batch_pixels

    sunny_std = torch.sqrt(sunny_M2 / sunny_pixels)
    rainy_std = torch.sqrt(rainy_M2 / rainy_pixels)

    print(f"Sunny images mean per channel: {sunny_mean}")
    print(f"Sunny images standard deviation per channel: {sunny_std}")
    print(f"Rainy images mean per channel: {rainy_mean}")
    print(f"Rainy images standard deviation per channel: {rainy_std}")

    return sunny_mean, sunny_std, rainy_mean, rainy_std

def explore_point_cloud_img_projection_datamodule():
    scenario = "all"
    datasets = ["waymo"]
    order_by = "weather"
    batch_size = 1
    downsampled_pointcloud_size = None
    classes = get_aki_label_names()
    void_classes = ["void", "static"]
    train_limit = 30
    val_limit = 50

    # Create data module
    proj_dm = SemanticLidarSegmentationDataModule(scenario=scenario,
                                                     order_by=order_by,
                                                     batch_size=batch_size,
                                                     downsampled_pointcloud_size=downsampled_pointcloud_size,
                                                     datasets=datasets,
                                                     classes=classes,
                                                     void=void_classes,
                                                     train_limit=train_limit,
                                                     val_limit=val_limit)

    proj_dm.setup()

    for elem in proj_dm.train_dataloader():
        points, labels = elem

    


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