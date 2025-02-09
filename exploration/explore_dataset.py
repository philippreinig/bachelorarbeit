# Imports
import torch
import matplotlib.pyplot as plt
from random import randint
import numpy as np
from torch.utils.data import DataLoader
import uuid
import math
import logging
from rich.logging import RichHandler
from rich import inspect
from rich.progress import track
from torchvision.utils import draw_segmentation_masks
from akiset import AKIDataset
from utils import prepare_batch, add_segmentation_mask_to_img, divide_batch_of_tensors

logging.basicConfig(level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()])

log = logging.getLogger("rich")


########################################################################################################################

def export_imgs_in_different_conditions():

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

if __name__ == "__main__":
    log.info(f"Exploration script started")
    export_imgs_in_different_conditions()
    #export_divided_imgs()
