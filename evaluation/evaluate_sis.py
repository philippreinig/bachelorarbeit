import logging
import torch
import lightning as L
import torch.nn.functional as F

#from data_modules.data_module_utils import elems_in_dataset

log = logging.getLogger(__name__)

def evaluate_semantic_image_segmentation(model: L.LightningModule, datamodule: L.LightningDataModule):
    trainer = L.Trainer(accelerator="cpu",
                        enable_progress_bar=False)
    predictions = trainer.predict(model, datamodule.val_dataloader())
    logits = torch.cat([pred['logits'] for pred in predictions], dim=0)
    labels = torch.cat([pred['labels'] for pred in predictions], dim=0)
    first_pred_bef_reshape = logits[0, :, 0, 0]
    logits = torch.permute(logits, (0,2,3,1))
    labels_one_hot = F.one_hot(labels, num_classes=25)

    logits_n, logits_h, logits_w, logits_c = logits.shape
    logits_flattened = torch.reshape(logits, (logits_n, logits_h * logits_w, logits_c))
    labels_n, labels_h, labels_w = labels.shape
    labels_flattened = torch.reshape(labels, (labels_n, labels_h * labels_w))

    log.info(f"Logits shape: {logits.shape}")
    log.info(f"Labels shape: {labels.shape}")
    log.info(f"Logits flattened shape: {logits_flattened.shape}")
    log.info(f"Labels flattened shape: {labels_flattened.shape}")


    log.info(f"First prediction before reshape: {first_pred_bef_reshape}")
    log.info(f"First prediction after reshape: {logits_flattened[0, 0]}")

    log.info(f"First 10 labels before reshape: {labels[0, 0, :10]}")
    log.info(f"First 10 labels after reshape: {labels[0, 0:10]}")

    return predictions


if __name__ == "__main__":
    check_create_clm()