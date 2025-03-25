import logging
import sys
import torch

import lightning as L
import torchmetrics as tm

import torch.nn.functional as F

from lightning.pytorch.loggers import WandbLogger
from models.semantic_image_segmentation import SemanticImageSegmentationModel
from models.semantic_lidar_segmentation import PointNet2
from data_modules.unified_datamodule import UnifiedDataModule
from utils.aki_labels import get_aki_label_names

from fusion.clm import create_unimodal_normalized_clm

logging.basicConfig(level=logging.INFO, format="[%(filename)s:%(lineno)s - %(funcName)s] %(message)s", datefmt="[%X]")

log = logging.getLogger(__name__)

def create_clm_img():
    # DEFINE PARAMETERS
    scenario="sun"
    datasets=["waymo"]
    classes=get_aki_label_names()
    num_workers = 1
    batch_size = 128
    val_limit = 1000
    crop_size=(800, 1600)
    grid_cells=(1,1)
    data_from_udm=True
    accelerator="gpu"
    num_classes = len(classes)

    log.info(f"=== PARAMS === ")
    log.info(f"Scenario: {scenario}")
    log.info(f"Datasets: {datasets}")
    log.info(f"Classes: {classes}")
    log.info(f"Num workers: {num_workers}")
    log.info(f"Batch size: {batch_size}")
    log.info(f"Val limit: {val_limit}")
    log.info(f"Crop size: {crop_size}")
    log.info(f"Grid cells: {grid_cells}")
    log.info(f"Data from UDM: {data_from_udm}")
    log.info(f"Accelerator: {accelerator}")

    checkpoint = "checkpoints/final_models/sem_img_seg_udm_27_final-1357856.ckpt"


    sis_model = SemanticImageSegmentationModel.load_from_checkpoint(checkpoint,
                                                                    num_classes=len(classes),
                                                                    crop_size=crop_size,
                                                                    data_from_udm=data_from_udm)
    data_module = UnifiedDataModule(datasets=datasets,
                                    scenario=scenario,
                                    num_workers=num_workers,
                                    val_limit=val_limit,
                                    crop_size=crop_size,
                                    grid_cells=grid_cells,
                                    batch_size=batch_size,
                                    classes=classes)
    data_module.setup()

    trainer = L.Trainer(accelerator=accelerator,
                        enable_progress_bar=False)
    
    val_data_loader = data_module.val_dataloader()

    predictions = trainer.predict(sis_model, val_data_loader)
    logits = torch.cat([pred['logits'] for pred in predictions], dim=0)
    labels = torch.cat([pred['labels'] for pred in predictions], dim=0)
    logits_reshaped = torch.permute(logits, (0,2,3,1))
    preds = torch.softmax(logits_reshaped, dim=-1)

    preds_n, preds_h, preds_w, preds_c = preds.shape
    preds_flattened = torch.reshape(preds, (preds_n * preds_h * preds_w, preds_c))
    labels_n, labels_h, labels_w = labels.shape
    labels_flattened = torch.flatten(labels)
    labels_one_hot = F.one_hot(labels_flattened, num_classes=num_classes).float()

    log.info(f"Total elements yielded by data module: {data_module.prepared_elems}")
    log.info(f"Amount of images used to create CLM: {preds_n}")

    clm =  create_unimodal_normalized_clm(preds_flattened, labels_one_hot)
    torch.save(clm, f"clms/img_{scenario}.pt")

def create_clm_pcl():
    # DEFINE PARAMETERS
    scenario="sun"
    datasets=["waymo"]
    classes=get_aki_label_names()
    num_workers = 1
    batch_size = 128
    val_limit = 1000
    crop_size=(800, 1600)
    grid_cells=(1,1)
    data_from_udm=True
    accelerator="cpu"
    num_classes = len(classes)

    log.info(f"=== PARAMS === ")
    log.info(f"Scenario: {scenario}")
    log.info(f"Datasets: {datasets}")
    log.info(f"Classes: {classes}")
    log.info(f"Num workers: {num_workers}")
    log.info(f"Batch size: {batch_size}")
    log.info(f"Val limit: {val_limit}")
    log.info(f"Crop size: {crop_size}")
    log.info(f"Grid cells: {grid_cells}")
    log.info(f"Data from UDM: {data_from_udm}")
    log.info(f"Accelerator: {accelerator}")

    checkpoint = "checkpoints/final_models/sem_lid_seg_udm_27_final-1357876.ckpt"


    model = PointNet2.load_from_checkpoint(checkpoint,
                                           num_classes=num_classes,
                                           data_from_udm=data_from_udm)
    
    data_module = UnifiedDataModule(datasets=datasets,
                                    scenario=scenario,
                                    num_workers=num_workers,
                                    val_limit=val_limit,
                                    crop_size=crop_size,
                                    grid_cells=grid_cells,
                                    batch_size=batch_size,
                                    classes=classes)
    data_module.setup()

    trainer = L.Trainer(accelerator=accelerator,
                        enable_progress_bar=False)
    
    val_data_loader = data_module.val_dataloader()

    predictions = trainer.predict(model, val_data_loader)

    logits_list = [logit.view(-1, logit.shape[-1]) for pred in predictions for logit in pred['logits']]
    labels_list = [label.view(-1) for pred in predictions for label in pred['labels']]
    logits = torch.cat(logits_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    preds = torch.softmax(logits, dim=-1)
    labels_one_hot = F.one_hot(labels, num_classes=len(classes)).float()

    log.info(f"Total elements yielded by data module: {data_module.prepared_elems}")
    #log.info(f"Amount of pcls used to create CLM: {preds.shape[0]/(crop_size[0]*crop_size[1])}")

    clm =  create_unimodal_normalized_clm(preds, labels_one_hot)
    torch.save(clm, f"clms/pcl_{scenario}.pt")

def create_clm_both():
    pass


if __name__ == "__main__":
    create_clm_img()
    #create_clm_pcl()