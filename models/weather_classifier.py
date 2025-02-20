import torch
import torchvision.models as models
from torch import nn
import pytorch_lightning as pl
import torchmetrics as tm

import logging
from rich.logging import RichHandler
log = logging.getLogger("rich")


class WeatherClassifier(pl.LightningModule):
    def __init__(self):
        super(WeatherClassifier, self).__init__()

        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        #self.model = models.resnet18()
        
        num_features = self.model.fc.in_features
        log.info(f"fc in features are: {num_features}")

        self.output_classes = 2

        self.model.fc = nn.Linear(num_features, self.output_classes)

        self.criterion = nn.CrossEntropyLoss()

        self.train_acc = tm.Accuracy(task="multiclass", num_classes=self.output_classes)
        self.val_acc = tm.Accuracy(task="multiclass", num_classes=self.output_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch

        logits = self(images)

        loss = self.criterion(logits, labels)

        preds = torch.argmax(logits, dim=1) 

        self.train_acc(preds, labels)
        
        self.log("train_acc", self.train_acc, on_step=True)
        self.log("train_loss", loss, on_step=True)
        
        rain_lbls = torch.count_nonzero(labels)
        sunny_lbls = labels.shape[0] - torch.count_nonzero(labels)

        rain_preds = torch.count_nonzero(preds)
        sunny_preds = labels.shape[0] - torch.count_nonzero(preds)

        self.log("debug/train/rain_lbls", rain_lbls.float())
        self.log("debug/train/rain_preds", rain_preds.float())
        self.log("debug/train/sunny_lbls", sunny_lbls.float())
        self.log("debug/train/sunny_preds", sunny_preds.float())

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)

        preds = torch.argmax(logits, dim=1)
        self.val_acc(preds, labels)

        self.log("val_acc", self.val_acc, on_step=True)
        self.log("val_loss", loss, on_step=True)


        rain_lbls = torch.count_nonzero(labels)
        sunny_lbls = labels.shape[0] - torch.count_nonzero(labels)

        rain_preds = torch.count_nonzero(preds)
        sunny_preds = labels.shape[0] - torch.count_nonzero(preds)

        self.log("debug/val/rain_lbls", rain_lbls.float(), on_step=True)
        self.log("debug/val/rain_preds", rain_preds.float(), on_step=True)
        self.log("debug/val/sunny_lbls", sunny_lbls.float(), on_step=True)
        self.log("debug/val/sunny_preds", sunny_preds.float(), on_step=True)

        return loss

    def configure_optimizers(self):
        # Define optimizer
        return torch.optim.Adam(self.parameters() ,lr=1e-5)
