import torch
import torchvision.models as models
from torch import nn
import lightning as L
import torchmetrics as tm

import logging
from rich.logging import RichHandler
log = logging.getLogger("rich")


class WeatherClassifier(L.LightningModule):
    def __init__(self,
                 data_from_udm = False):
        super(WeatherClassifier, self).__init__()

        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        num_features = self.model.fc.in_features
        log.info(f"fc in features are: {num_features}")

        self.data_from_udm = data_from_udm

        self.output_classes = 2

        self.model.fc = nn.Linear(num_features, self.output_classes)

        self.criterion = nn.CrossEntropyLoss()

        self.device_for_batches = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.train_acc = tm.Accuracy(task="multiclass", num_classes=self.output_classes)
        self.val_acc = tm.Accuracy(task="multiclass", num_classes=self.output_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch

        logits = self(images)

        loss = self.criterion(logits, labels)

        preds = torch.argmax(logits, dim=1) 

        train_acc_step = self.train_acc(preds, labels)
        
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

        log.info(f"Loss and accuracy after training step {batch_idx}: {loss}, {train_acc_step}")

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
    
    def predict_step(self, batch: torch.Tensor, batch_idx: int = None):
        images, labels = (batch if not self.data_from_udm else batch[0], batch[4])

        images = images.to(self.device_for_batches)
        logits = self(images)
        
        logits = logits.cpu()

        preds = torch.nn.Softmax(dim=1)(logits)
        rain_probabilities = preds[:, 1]

        return dict(rain_probs=rain_probabilities, labels=labels)

    def configure_optimizers(self):
        # Define optimizer
        return torch.optim.Adam(self.parameters() ,lr=1e-6)
