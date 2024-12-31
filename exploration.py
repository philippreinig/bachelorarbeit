# Imports
import torch
import matplotlib.pyplot as plt
from random import randint
import numpy as np
from torch.utils.data import DataLoader
import uuid
import logging
from rich.logging import RichHandler
from rich import inspect
from rich.progress import track
from torchvision.utils import draw_segmentation_masks
from akiset_datamodule import AKIDataset
from utils import prepare_batch, add_segmentation_mask_to_img, divide_batch_of_tensors

logging.basicConfig(level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()])

log = logging.getLogger("rich")


########################################################################################################################
IMGS_FOLDER = "imgs/"


def explore_dataset():
    aki_ds = AKIDataset(data={"camera": ["image"], "camera_segmentation": ["camera_segmentation"]},
                          datasets=['nuimages'],
                          scenario='rain',
                          offset=0,
                          dbtype="psycopg@ants")

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
            plt.savefig(f"{IMGS_FOLDER + str(uuid.uuid4())}")
            plt.close(fig)

            log.info(f"{x * batch_size} images divided and exported to file")

if __name__ == "__main__":
    log.info(f"Exploration script started")
    explore_dataset()
