from typing import Sequence, Union

import time
import timm
import torch
import torchmetrics as tm
import torchvision.models as models
from torch import nn
from utils import unpack_feature_pyramid
import torch.nn.functional as F


from lightning.pytorch import LightningModule

torch.set_float32_matmul_precision("high")

class LRASPP(nn.Module):
    """
    Lite Atrous Spatial Pyramid Pooling
    Documentation: https://pytorch.org/vision/main/models/lraspp.html
    """
    def __init__(
        self,
        in_channels: list[int],
        internal_channels: int,
        num_classes: int,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        quarter, embedding = unpack_feature_pyramid(in_channels)

        self.cbr = nn.Sequential(
            nn.Conv2d(embedding, internal_channels, 1, bias=False),
            nn.BatchNorm2d(internal_channels),
            nn.ReLU(inplace=True),
        )
        self.scale = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(embedding, internal_channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.quarter_classifier = nn.Conv2d(quarter, num_classes, 1)
        self.embedding_classifier = nn.Conv2d(internal_channels, num_classes, 1)

    def forward(self, feature_pyramid: list[torch.Tensor]) -> torch.Tensor:
        [quarter, embedding] = unpack_feature_pyramid(feature_pyramid)

        x = self.cbr(embedding)
        s = self.scale(embedding)
        x = x * s

        # dont write scale=4, because irregular input sizes can round down
        #  s.t. quarter != 4 * (quarter // 4)
        B, C, H, W = quarter.shape
        # print("Quarter: ", x.shape)
        x = F.interpolate(x, size=(H, W), mode="bilinear")
        # print(x.shape)

        x = self.quarter_classifier(quarter) + self.embedding_classifier(x)
        # use scale=4 here, because original image dimension is not given
        x = F.interpolate(x, scale_factor=4, mode="bilinear")
        # print("Img: ", x.shape)
        return x

class SemanticImageSegmentationModel(LightningModule):
    def __init__(
        self,
        num_classes: int,
        ignore_index: int = 255,
        decoder_internal_channels: int = 128,
        compile: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        #self.encoder = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.encoder = timm.create_model("resnet18", features_only=True, pretrained=True, output_stride=16)
        pyramid = self.encoder.feature_info.channels()
        self.decoder = LRASPP(pyramid, 5, num_classes)

        if compile:
            self.encoder = torch.compile(self.encoder)
            self.decoder = torch.compile(self.decoder)

        self.num_classes = num_classes
        self.ignore_index = ignore_index

        collection = tm.MetricCollection(
            tm.classification.MulticlassAccuracy(num_classes=self.num_classes, ignore_index=self.ignore_index, average="micro")
        )
        self.train_metrics = collection.clone(prefix="train/")
        self.valid_metrics = collection.clone(prefix="validation/")
        self.test_metrics = collection.clone(prefix="test/")
        self.log_kwargs = dict(on_step=False, on_epoch=True, sync_dist=True)

        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255)

    def forward(self, images):
        feature_pyramid = self.encoder(images)
        logits = self.decoder(feature_pyramid)
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
