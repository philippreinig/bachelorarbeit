import torch
import torchvision.models as models
from torch import nn
import pytorch_lightning as pl
import torchmetrics

import logging
from rich.logging import RichHandler
log = logging.getLogger("rich")


class WeatherClassifier(pl.LightningModule):
    def __init__(self):
        super(WeatherClassifier, self).__init__()

        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_features = self.model.fc.in_features
        log.info(f"fc in features are: {num_features}")

        self.output_classes = 2

        self.model.fc = nn.Linear(num_features, self.output_classes)

        self.criterion = nn.BCEWithLogitsLoss()
        self.softmax = torch.nn.Softmax(dim=1)

        self.train_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=self.output_classes)
        self.val_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=self.output_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch

        logits = self(images)

        loss = self.criterion(logits, labels.float())

        self.train_acc(self.softmax(logits), labels)

        self.log("train_acc", self.train_acc)
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels.float())

        self.val_acc(self.softmax(logits), labels)

        self.log("val_acc", self.val_acc)
        self.log("val_loss", loss)

        return loss

    def configure_optimizers(self):
        # Define optimizer
        return torch.optim.Adam(self.parameters())
