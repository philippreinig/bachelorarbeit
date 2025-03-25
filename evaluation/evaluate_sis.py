import logging
import torch
import lightning as L
import torch.nn.functional as F
import torchmetrics as tm

from lightning.pytorch.loggers import WandbLogger


log = logging.getLogger(__name__)

def evaluate_semantic_image_segmentation(model: L.LightningModule, data_module: L.LightningDataModule, wandb_logger: WandbLogger, split: str = "val", num_classes=27, ignore_index=255, accelerator: str = "gpu"):
    if not split in ["val", "test"]:
        raise ValueError(f"Argument split must be either val or test, but is: {split}")
    trainer = L.Trainer(logger=wandb_logger,
                        accelerator=accelerator,
                        enable_progress_bar=False)
    data_loader = data_module.val_dataloader() if split == "val" else data_module.test_dataloader()

    predictions = trainer.predict(model, data_loader)
    logits = torch.cat([pred['logits'] for pred in predictions], dim=0)
    labels = torch.cat([pred['labels'] for pred in predictions], dim=0)
    logits_reshaped = torch.permute(logits, (0,2,3,1))
    preds = torch.softmax(logits_reshaped, dim=-1)
    labels_one_hot = F.one_hot(labels, num_classes=num_classes)

    preds_n, preds_h, preds_w, preds_c = preds.shape
    preds_flattened = torch.reshape(preds, (preds_n * preds_h * preds_w, preds_c))
    labels_n, labels_h, labels_w = labels.shape
    labels_flattened = torch.flatten(labels)

    # Accuracy
    acc = tm.classification.Accuracy(task="multiclass", num_classes=num_classes, ignore_index=ignore_index, average="micro")(logits, labels)
    log.info(f"Accuracy is: {acc}")
    
    # F1
    f1_mac = tm.functional.f1_score(preds_flattened, labels_flattened, "multiclass", num_classes=num_classes, average="macro")
    f1_per_class = tm.functional.f1_score(preds_flattened, labels_flattened, "multiclass", num_classes=num_classes, average=None)
    log.info(f"Macro F1: {f1_mac}")
    log.info(f"F1 scores per class: {f1_per_class}")

    # ECE
    ece = tm.CalibrationError("multiclass", num_classes=num_classes, ignore_index=ignore_index)(preds_flattened, labels_flattened)
    log.info(f"ECE: {ece}")


    return predictions

