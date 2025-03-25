import timm
import torch
import torchmetrics as tm
import torchvision.models as models
from torch import nn
from models.model_utils import unpack_feature_pyramid
import torch.nn.functional as F

import logging

log = logging.getLogger(__name__)

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
        num_classes: int,
        output_image_size: tuple[int, int],
        internal_channels: int = 128,
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

        self.output_image_size = output_image_size

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
        x = F.interpolate(x, size=self.output_image_size, mode="bilinear")
        # print("Img: ", x.shape)
        return x

class SemanticImageSegmentationModel(LightningModule):
    def __init__(
        self,
        num_classes: int,
        ignore_index: int = 255,
        compile: bool = False,
        train_epochs: int = 30,
        crop_size: tuple[int, int] = (800, 1600),
        data_from_udm: bool = False,
    ):
        super().__init__()

        self.crop_size = crop_size
        self.data_from_udm = data_from_udm

        self.encoder = timm.create_model("resnet18", features_only=True, pretrained=True, output_stride=16)
        pyramid = self.encoder.feature_info.channels()
        self.decoder = LRASPP(pyramid, num_classes, self.crop_size)

        if compile:
            self.encoder = torch.compile(self.encoder)
            self.decoder = torch.compile(self.decoder)

        self.num_classes = num_classes
        self.ignore_index = ignore_index
        
        self.train_epochs = train_epochs


        self.train_acc = tm.classification.Accuracy(task="multiclass", num_classes=self.num_classes, ignore_index=ignore_index, average="micro")
        self.val_acc = tm.classification.Accuracy(task="multiclass", num_classes=self.num_classes, ignore_index=ignore_index, average="micro")
        self.test_acc = tm.classification.Accuracy(task="multiclass", num_classes=self.num_classes, ignore_index=ignore_index, average="micro")

        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.ignore_index)

    def forward(self, images):
        feature_pyramid = self.encoder(images)
        logits = self.decoder(feature_pyramid)
        return logits

    def step(self, batch):
        images, labels = (batch if not self.data_from_udm else (batch[0], batch[1])) 
        images = images.to(memory_format=torch.channels_last)
        logits = self(images)
        return self.loss_fn(logits, labels), logits, labels

    def training_step(self, batch, batch_idx):
        loss, logits, labels = self.step(batch)
        
        self.train_acc(logits, labels)

        self.log("train_loss", loss)
        self.log("train_acc", self.train_acc)

        return dict(loss=loss, logits=logits)

    def validation_step(self, batch, batch_idx):
        loss, logits, labels = self.step(batch)

        self.val_acc(logits, labels)

        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True)
        
        return dict(loss=loss, logits=logits)

    def test_step(self, batch, batch_idx):
        loss, logits, labels = self.step(batch)

        self.test_acc(logits, labels)
        
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True)

        return dict(loss=loss, logits=logits)
    
    def predict_step(self, batch, batch_idx):

        loss, logits, labels = self.step(batch)

        return dict(loss=loss, logits=logits, labels=labels)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.train_epochs, 1e-7)
        schedule = {
            "scheduler": scheduler,
            "interval": 'epoch',
            "frequency": 1,
        }

        return [[opt], [schedule]] if scheduler else opt
