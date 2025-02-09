import torch
import logging

import pytorch_lightning as pl
import torchvision as tv

from utils.aki_labels import get_aki_label_colors

log = logging.getLogger("rich")

def divide_batch_of_tensors(t: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """
    Divide tensor into a grid of n rows and m columns.

    raise ValueError if width of tensor is not divisible by n or height of tensor is not divisible by m.
    """
    if len(t.shape) != 4:
        raise ValueError(f"Input tensor must have 4 dimensions (b x l x w x h), but shape is:  {t.shape}")

    b, l, h, w = t.shape

    if l != 3:
        raise ValueError(f"Expected 3 layers, but got: {l}")

    if w % cols != 0:
        raise ValueError(f"Width of image {w} is not divisible by {cols}")

    if h % rows != 0:
        raise ValueError(f"Height of image {h} is not divisible by {rows}")

    cell_width = w // cols
    cell_height = h // rows

    new_tensor = torch.zeros((b, rows * cols, l, cell_height, cell_width))

    for i in range(rows):
        for j in range(cols):
            new_tensor[:, i * cols + j, :, :, :] = t[:, :, i * cell_height:(i + 1) * cell_height, j * cell_width:(j + 1) * cell_width]

    return new_tensor

def prepare_batch(batch):
    input_batch = torch.stack([elem[0] for elem in batch], 0)
    label_batch = torch.stack([elem[1] for elem in batch], 0)
    return input_batch, label_batch

def add_segmentation_mask_to_img(img: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    label[label == 255] = 0
    label_transformed = torch.nn.functional.one_hot(label, num_classes=27).permute(2, 0, 1).bool()
    img_with_segmentation_mask = tv.utils.draw_segmentation_masks(img, label_transformed, colors=get_aki_label_colors(), alpha=0.5)
    return img_with_segmentation_mask

      
def unpack_feature_pyramid(feature_pyramid):
    try:
        # This works for resnets, efficient-nets
        [_, quarter, eights, _, out] = feature_pyramid
    except ValueError:
        # This works for convnexts
        [quarter, eights, _, out] = feature_pyramid
    return quarter, out

def weather_condition2numeric(weather_condition: str) -> list:
    mapping = {
        "Clear Sky": [1, 0],
        "Heavy Rain": [0, 1],
        "Dense Drizzle": [0,1],
        "Light Drizzle": [0, 1],
        "Light Rain": [0, 1],
        "Mainly Clear": [1, 0],
        "Moderate Drizzle": [0, 1],
        "Moderate Rain": [0, 1],
        "Overcast": [1, 0],
        "Partly Cloudy": [1, 0],
        "rain": [0, 1],
        "sunny": [1, 0]
    }

    if "Snow" in weather_condition:
        raise ValueError("Can't embed snow!")

    return mapping[weather_condition]


class MyPrintingCallback(pl.Callback):
    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        log.info("Starting training")

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        log.info("Training done")
