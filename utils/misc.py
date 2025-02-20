import torch
import logging

import pytorch_lightning as pl
import torchvision as tv

from utils.aki_labels import get_aki_label_colors

log = logging.getLogger("rich")


def add_segmentation_mask_to_img(img: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    label[label == 255] = 0
    label_transformed = torch.nn.functional.one_hot(label, num_classes=27).permute(2, 0, 1).bool()
    img_with_segmentation_mask = tv.utils.draw_segmentation_masks(img, label_transformed, colors=get_aki_label_colors(), alpha=0.5)
    return img_with_segmentation_mask


class MyPrintingCallback(pl.Callback):
    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        log.info("Starting training")

    def on_train_epoch_end(self, trainer, pl_module):
        # Get the logged metrics
        metrics = trainer.callback_metrics
        train_loss = metrics.get("train_loss", "N/A")
        train_acc = metrics.get("train_acc", "N/A")  # Change this based on your metric name

        log.info(f"Epoch {trainer.current_epoch} - Train Loss: {train_loss}, Train Acc: {train_acc}")

    def on_validation_epoch_end(self, trainer, pl_module):
        # Get the logged metrics
        metrics = trainer.callback_metrics
        val_loss = metrics.get("val_loss", "N/A")
        val_acc = metrics.get("val_acc", "N/A")  # Change this based on your metric name

        log.info(f"Epoch {trainer.current_epoch} - Validation Loss: {val_loss}, Validation Acc: {val_acc}")

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        log.info("Training done")