import logging
import sys
import torch

import lightning as L
import torchmetrics as tm

import torch.nn.functional as F

from typing import Optional
from lightning.pytorch.loggers import WandbLogger
from models.semantic_image_segmentation import SemanticImageSegmentationModel
from models.semantic_lidar_segmentation import PointNet2
from data_modules.unified_datamodule import UnifiedDataModule
from utils.aki_labels import get_aki_label_names
from fusion.fuse import fuse_unimodal_with_clm

logging.basicConfig(level=logging.INFO, format="[%(filename)s:%(lineno)s - %(funcName)s] %(message)s", datefmt="[%X]")

log = logging.getLogger(__name__)

def evaluate_sem_img_seg_with_udm(slurm_job_id: int, clm_path: Optional[str]):
    # DEFINE PARAMETERS
    scenario="rain"
    order_by="weather"
    datasets=["waymo"]
    classes=get_aki_label_names()
    num_workers = 1
    batch_size = 128
    val_limit = 1000
    test_limit = 1000
    void_classes = []
    crop_size=(800, 1600)
    grid_cells=(1,1)
    data_from_udm=True
    split = "test"
    accelerator="gpu" if slurm_job_id else "cpu"
    ignore_index = 255
    num_classes = len(classes)
    clm = torch.load(clm_path) if clm_path else None

    log.info(f"=== PARAMS === ")
    log.info(f"Scenario: {scenario}")
    log.info(f"CLM: {clm_path}")
    log.info(f"Split: {split}")
    log.info(f"Order by: {order_by}")
    log.info(f"Classes: {classes}")
    log.info(f"Void classes: {void_classes}")
    log.info(f"Num workers: {num_workers}")
    log.info(f"Batch size: {batch_size}")
    log.info(f"Val limit: {val_limit}")
    log.info(f"Test limit: {test_limit}")
    log.info(f"Crop size: {crop_size}")
    log.info(f"Grid cells: {grid_cells}")
    log.info(f"Data from UDM: {data_from_udm}")
    log.info(f"Accelerator: {accelerator}")
    log.info(f"Datasets: {datasets}")

    checkpoint = "checkpoints/final_models/sem_img_seg_udm_27_final-1357856.ckpt"

    wandb_logger = WandbLogger(name=f"sis-raw-{scenario}-{split}-{slurm_job_id}",
                               log_model="all",
                               save_dir="logs/wandb/eval/",
                               project="eval",
                               offline=False)

    model = SemanticImageSegmentationModel.load_from_checkpoint(checkpoint,
                                                                    num_classes=len(classes),
                                                                    crop_size=crop_size,
                                                                    data_from_udm=data_from_udm)
    data_module = UnifiedDataModule(datasets=datasets,
                                    scenario=scenario,
                                    num_workers=num_workers,
                                    order_by=order_by,
                                    val_limit=val_limit,
                                    test_limit=test_limit,
                                    crop_size=crop_size,
                                    void=void_classes,
                                    grid_cells=grid_cells,
                                    batch_size=batch_size,
                                    classes=classes)
    data_module.setup()

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

    if clm != None:
        preds_adjusted = fuse_unimodal_with_clm(preds_flattened, clm)
        preds_final_is_prob_dist = True
    else:
        preds_adjusted = preds_flattened
        preds_final_is_prob_dist = False

    #preds_final = preds_adjusted.argmax(dim=-1) if preds_final_is_prob_dist else preds_adjusted

    # Accuracy
    acc = tm.classification.Accuracy(task="multiclass", num_classes=num_classes, ignore_index=ignore_index, average="micro")(preds_adjusted, labels_flattened)
    log.info(f"Accuracy is: {acc}")

    # F1
    f1_mac = tm.functional.f1_score(preds_adjusted, labels_flattened, "multiclass", num_classes=num_classes, average="macro")
    f1_per_class = tm.functional.f1_score(preds_adjusted, labels_flattened, "multiclass", num_classes=num_classes, average=None)
    log.info(f"Macro F1: {f1_mac}")
    log.info(f"F1 scores per class: {f1_per_class}")

    # ECE
    ece = tm.CalibrationError("multiclass", num_classes=num_classes, ignore_index=ignore_index)(preds_adjusted, labels_flattened)
    log.info(f"ECE: {ece}")

    log.info(f"Total elements yielded by data module: {data_module.prepared_elems}")

def evaluate_sem_lid_seg_with_udm_raw(slurm_job_id: int):
    # DEFINE PARAMETERS
    # Dataset
    scenario="sun"
    order_by="weather"
    datasets=["waymo"]
    classes=get_aki_label_names()
    void_classes=[]
    num_workers = 1
    batch_size = 32
    val_limit= 150
    test_limit = 150
    split="test"
    crop_size = (800, 1600)
    grid_cells = (1,1)
    ignore_index=255
    num_classes=len(classes)

    # Model
    data_from_udm=True
    checkpoint = "checkpoints/final_models/sem_lid_seg_udm_27_final-1357876.ckpt"

    # Trainer
    precision="16-mixed"
    accelerator="gpu"

    # Logger
    offline=False if slurm_job_id else True

    log.info(f"=== PARAMS ===")
    log.info(f"Scenario: {scenario}")
    log.info(f"Datasets: {datasets}")
    log.info(f"Split: {split}")
    log.info(f"Order by: {order_by}")
    log.info(f"Classes: {classes}")
    log.info(f"Void classes: {void_classes}")
    log.info(f"Num workers: {num_workers}")
    log.info(f"Batch size: {batch_size}")
    log.info(f"Val limit: {val_limit}")
    log.info(f"Test limit: {test_limit}")
    log.info(f"Crop size: {crop_size}")
    log.info(f"Grid cells: {grid_cells}")
    log.info(f"Data from UDM: {data_from_udm}")
    log.info(f"Accelerator: {accelerator}")
    log.info(f"Checkpoint: {checkpoint}")

    wandb_logger = WandbLogger(name=f"sls-raw-{scenario}-{split}-{slurm_job_id}",
                               #log_model="all",
                               save_dir="logs/wandb/eval/",
                               project="eval",
                               offline=offline)

    model = PointNet2.load_from_checkpoint(checkpoint,
                                           num_classes=len(classes),
                                           data_from_udm=data_from_udm)
    
    data_module = UnifiedDataModule(scenario=scenario,
                            order_by=order_by,
                            datasets=datasets,
                            classes=classes,
                            val_limit=val_limit,
                            test_limit=test_limit,
                            void=void_classes,
                            num_workers=num_workers,
                            batch_size=batch_size)
    
    data_module.setup()
    
    trainer = L.Trainer(logger=wandb_logger,
                        precision=precision,
                        enable_progress_bar=False,
                        accelerator=accelerator)

    if not split in ["val", "test"]:
        raise ValueError(f"Argument split must be either val or test, but is: {split}")

    data_loader = data_module.val_dataloader() if split == "val" else data_module.test_dataloader()

    predictions = trainer.predict(model, data_loader)
    logits_list = [logit.view(-1, logit.shape[-1]) for pred in predictions for logit in pred['logits']]
    labels_list = [label.view(-1) for pred in predictions for label in pred['labels']]
    logits = torch.cat(logits_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    #logits_reshaped = torch.permute(logits, (0,2,3,1))
    preds = torch.softmax(logits, dim=-1)
    labels_one_hot = F.one_hot(labels, num_classes=len(classes))

    preds_n, preds_c = preds.shape
    labels_n = labels.shape

    # Accuracy
    acc = tm.classification.Accuracy(task="multiclass", num_classes=num_classes, ignore_index=ignore_index, average="micro")(logits, labels)
    log.info(f"Accuracy is: {acc}")
    
    # F1
    f1_mac = tm.functional.f1_score(preds, labels, "multiclass", num_classes=num_classes, average="macro")
    f1_per_class = tm.functional.f1_score(preds, labels, "multiclass", num_classes=num_classes, average=None)
    log.info(f"Macro F1: {f1_mac}")
    log.info(f"F1 scores per class: {f1_per_class}")

    # ECE
    ece = tm.CalibrationError("multiclass", num_classes=num_classes, ignore_index=ignore_index)(preds, labels)
    log.info(f"ECE: {ece}")

def evaluate_sem_img_seg_with_udm_unimodal_correction(slurm_job_id: int):
    pass

if __name__ == "__main__":
    #evaluate_sem_lid_seg_with_udm_raw(sys.argv[1] if len(sys.argv) > 1 else None)
    evaluate_sem_img_seg_with_udm(sys.argv[1] if len(sys.argv) > 1 else None, "clms/img_rain.pt")