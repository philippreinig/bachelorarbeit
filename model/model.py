from typing import Sequence, Union

import torchmetrics
from hydra.utils import instantiate
from omegaconf import DictConfig

import time
import timm
import torch
import torchmetrics as tm
import torchmetrics.classification as tmc

from lightning.pytorch import LightningModule

from model.decoder import LRASPPHead

torch.set_float32_matmul_precision("high")


class SegmentationModel(LightningModule):
    def __init__(
        self,
        num_classes: int,
        decoder_internal_channels: int = 128,
        compi: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.backbone = timm.create_model("resnet18", features_only=True, pretrained=True, output_stride=16)
        pyramid = self.backbone.feature_info.channels()
        self.head = LRASPPHead(pyramid, 5, num_classes)

        if compile:
            self.backbone = torch.compile(self.backbone)
            self.head = torch.compile(self.head)

        self.num_classes = num_classes

        collection = torchmetrics.MetricCollection(
            torchmetrics.classification.MulticlassAccuracy(num_classes=self.num_classes, ignore_index=255, average="micro")
        )
        self.train_metrics = collection.clone(prefix="train/")
        self.valid_metrics = collection.clone(prefix="validation/")
        self.test_metrics = collection.clone(prefix="test/")
        self.log_kwargs = dict(on_step=False, on_epoch=True, sync_dist=True)

        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255)

    def forward(self, images):
        feature_pyramid = self.backbone(images)
        logits = self.head(feature_pyramid)
        return logits

    def step(self, batch):
        images, labels = batch
        images = images.to(memory_format=torch.channels_last)
        logits = self(images)
        return self.loss_fn(logits, labels), logits, labels

    def training_step(self, batch, batch_idx):
        # batch_start = time.time()
        loss, logits, labels = self.step(batch)
        self.log("train/loss", loss, **self.log_kwargs)

        return dict(loss=loss, logits=logits)

    def validation_step(self, batch, batch_idx):
        loss, logits, labels = self.step(batch)
        self.log("validation/loss", loss, **self.log_kwargs)
        self.log_dict(
            self.valid_metrics(logits, labels),
            **self.log_kwargs,
            prog_bar=True,
        )
        return dict(loss=loss, logits=logits)

    def test_step(self, batch, batch_idx):
        loss, logits, labels = self.step(batch)

        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log_dict(
            self.test_metrics(logits, labels),
            **self.log_kwargs,
        )

        return dict(loss=loss, logits=logits)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 5000, 1e-7)
        schedule = {
            "scheduler": scheduler,
            "interval": "step",  # or 'epoch'
            "frequency": 1,
        }

        return [[opt], [schedule]] if scheduler else opt
