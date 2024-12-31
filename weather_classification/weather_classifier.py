import torch
import torchvision.models as models
from torch import nn
import pytorch_lightning as pl

class WeatherClassifier(pl.LightningModule):
    def __init__(self):
        super(WeatherClassifier, self).__init__()

        self.model = models.resnet50(pretrained=True)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 1)

        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch

        outputs = self(images).squeeze() # TODO: Understand why to squeeze here?
        loss  = self.criterion(outputs, labels.float())
        self.log("train_loss", loss)

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images).squeeze()
        loss = self.criterion(outputs, labels.float())
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        # Define optimizer
        return torch.optim.Adam(self.parameters())
