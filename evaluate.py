import logging
import sys
import torch

import lightning as L
import torchmetrics as tm
import pandas as pd

import torch.nn.functional as F

from typing import Optional
from lightning.pytorch.loggers import WandbLogger
from models.semantic_image_segmentation import SemanticImageSegmentationModel
from models.weather_classifier import WeatherClassifier
from models.semantic_lidar_segmentation import PointNet2
from data_modules.unified_datamodule import UnifiedDataModule
from utils.aki_labels import get_aki_label_names
from fusion.fuse import fuse_unimodal_with_clm, fuse_unimodal_with_rain_probs, fuse_multimodal, fuse_multimodal_with_rain_probs

logging.basicConfig(level=logging.INFO, format="[%(filename)s:%(lineno)s - %(funcName)s] %(message)s", datefmt="[%X]")

log = logging.getLogger(__name__)

def evaluate_sem_img_seg_with_udm(slurm_job_id: int):
    # DEFINE PARAMETERS
    scenario="combined"
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
    clm_path_rain = "clms/img_rain.pt"
    clm_path_sun = "clms/img_sun.pt"
    clm_rain = torch.load(clm_path_rain) if clm_path_rain else None
    clm_sun = torch.load(clm_path_sun) if clm_path_sun else None

    log.info(f"=== PARAMS === ")
    log.info(f"Scenario: {scenario}")
    log.info(f"CLM rain: {clm_path_rain}")
    log.info(f"CLM sun: {clm_path_sun}")
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

    data_loader = data_module.val_dataloader() if split == "val" else data_module.test_dataloader()
    predictions = []
    labels = []
    rain_labels = []
    for batch_idx, batch in enumerate(data_loader):
        with torch.no_grad():
            pred = model.predict_step(batch, batch_idx)
            predictions.append(pred['probs'])
            labels.append(pred['labels'])
            rain_labels.append(batch[4])

    predictions = torch.cat(predictions, dim=0)
    labels = torch.cat(labels, dim=0)

    predictions_n, predictions_h, predictions_w, predictions_c = predictions.shape
    predictions_flattened = torch.reshape(predictions, (predictions_n * predictions_h * predictions_w, predictions_c))
    rain_labels = torch.cat(rain_labels)
    labels = torch.flatten(labels)

    if scenario == "rain":
        if not clm_rain:
            raise ValueError("CLM rain is not available.")
        predictions_fused = fuse_unimodal_with_clm(predictions_flattened,
                                                scenario,
                                                clm_rain=clm_rain)
    elif scenario == "sun":
        if not clm_sun:
            raise ValueError("CLM sun is not available.")
        predictions_fused = fuse_unimodal_with_clm(predictions_flattened,
                                                scenario,
                                                clm_sun=clm_sun)
    elif scenario == "combined":
        predictions_fused = fuse_unimodal_with_clm(predictions_flattened,
                                                scenario,
                                                clm_rain=clm_rain,
                                                clm_sun=clm_sun,
                                                rain_labels=rain_labels)

    # Accuracy
    acc = tm.classification.Accuracy(task="multiclass", num_classes=num_classes, ignore_index=ignore_index, average="micro")(predictions_fused, labels)
    log.info(f"Accuracy is: {acc}")

    # F1
    f1_mac = tm.functional.f1_score(predictions_fused, labels, "multiclass", num_classes=num_classes, average="macro")
    f1_per_class = tm.functional.f1_score(predictions_fused, labels, "multiclass", num_classes=num_classes, average=None)
    log.info(f"Macro F1: {f1_mac}")
    log.info(f"F1 scores per class: {f1_per_class}")

    # ECE
    ece = tm.CalibrationError("multiclass", num_classes=num_classes, ignore_index=ignore_index)(predictions_fused, labels)
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

def evaluate_sem_img_seg_on_combined_scenario_with_rain_probability(slurm_job_id: int):
    # DEFINE PARAMETERS
    scenario="combined"
    datasets=["waymo"]
    classes=get_aki_label_names()
    num_workers = 1
    batch_size = 128
    void_classes = []
    crop_size=(800, 1600)
    grid_cells=(1,1)
    data_from_udm=True
    split = "test"
    accelerator="gpu" if slurm_job_id else "cpu"
    ignore_index = 255
    num_classes = len(classes)
    clm_path_rain = "clms/img_rain.pt"
    clm_path_sun = "clms/img_sun.pt"
    clm_rain = torch.load(clm_path_rain) if clm_path_rain else None
    clm_sun = torch.load(clm_path_sun) if clm_path_sun else None

    # Logger
    offline=False if slurm_job_id else True

    log.info(f"=== PARAMS === ")
    log.info(f"Scenario: {scenario}")
    log.info(f"CLM rain: {clm_path_rain}")
    log.info(f"CLM sun: {clm_path_sun}")
    log.info(f"Split: {split}")
    log.info(f"Classes: {classes}")
    log.info(f"Void classes: {void_classes}")
    log.info(f"Num workers: {num_workers}")
    log.info(f"Batch size: {batch_size}")
    log.info(f"Crop size: {crop_size}")
    log.info(f"Grid cells: {grid_cells}")
    log.info(f"Data from UDM: {data_from_udm}")
    log.info(f"Accelerator: {accelerator}")
    log.info(f"Datasets: {datasets}")

    checkpoint = "checkpoints/final_models/sem_img_seg_udm_27_final-1357856.ckpt"

    wandb_logger = WandbLogger(name=f"sis-raw-{scenario}-{split}-{slurm_job_id}",
                               log_model="all" if not offline else False,
                               save_dir="logs/wandb/eval/",
                               project="eval",
                               offline=offline)

    sis_model = SemanticImageSegmentationModel.load_from_checkpoint(checkpoint,
                                                                    num_classes=len(classes),
                                                                    crop_size=crop_size,
                                                                    data_from_udm=data_from_udm)
    
    wc_model = WeatherClassifier.load_from_checkpoint("checkpoints/final_models/weather_classifier_final-unknown.ckpt",
                                                      
                                                      data_from_udm=data_from_udm)
    
    data_module = UnifiedDataModule(datasets=datasets,
                                    scenario=scenario,
                                    num_workers=num_workers,
                                    crop_size=crop_size,
                                    void=void_classes,
                                    grid_cells=grid_cells,
                                    batch_size=batch_size,
                                    classes=classes)
    data_module.setup()

    if not split in ["val", "test"]:
        raise ValueError(f"Argument split must be either val or test, but is: {split}")
    
    sis_trainer = L.Trainer(logger=wandb_logger,
                            accelerator=accelerator,
                            enable_progress_bar=False)
    
    wc_trainer = L.Trainer(logger=wandb_logger,
                           accelerator=accelerator,
                           enable_progress_bar=False)
    
        
    
    data_loader = data_module.val_dataloader() if split == "val" else data_module.test_dataloader()

    sis_predictions = sis_trainer.predict(sis_model, data_loader)
    
    rain_predictions = wc_trainer.predict(wc_model, data_loader)
    rain_probabilities = torch.cat([pred['rain_probs'] for pred in rain_predictions], dim=0)

    sis_logits = torch.cat([pred['logits'] for pred in sis_predictions], dim=0)
    sis_labels = torch.cat([pred['labels'] for pred in sis_predictions], dim=0)
    sis_logits_reshaped = torch.permute(sis_logits, (0,2,3,1))
    sis_preds = torch.softmax(sis_logits_reshaped, dim=-1)

    sis_preds_n, sis_preds_h, sis_preds_w, sis_preds_c = sis_preds.shape
    sis_preds_rshpd = torch.reshape(sis_preds, (sis_preds_n, sis_preds_h * sis_preds_w, sis_preds_c))
    #sis_labels_n, sis_labels_h, sis_labels_w  = sis_labels.shape
    sis_labels_flattened = torch.flatten(sis_labels)

    sis_preds_fused_and_flattened = fuse_unimodal_with_rain_probs(sis_preds_rshpd, rain_probabilities, clm_rain, clm_sun)

    # Accuracy
    acc = tm.classification.Accuracy(task="multiclass", num_classes=num_classes, ignore_index=ignore_index, average="micro")(sis_preds_fused_and_flattened, sis_labels_flattened)
    log.info(f"Accuracy is: {acc}")

    # F1
    f1_mac = tm.functional.f1_score(sis_preds_fused_and_flattened, sis_labels_flattened, "multiclass", num_classes=num_classes, average="macro")
    f1_per_class = tm.functional.f1_score(sis_preds_fused_and_flattened, sis_labels_flattened, "multiclass", num_classes=num_classes, average=None)
    log.info(f"Macro F1: {f1_mac}")
    log.info(f"F1 scores per class: {f1_per_class}")

    # ECE
    ece = tm.CalibrationError("multiclass", num_classes=num_classes, ignore_index=ignore_index)(sis_preds_fused_and_flattened, sis_labels_flattened)
    log.info(f"ECE: {ece}")

def evaluate_multi_modal(slurm_job_id: Optional[int]):
    def prepare_data_for_fusion(cam_preds: list[torch.Tensor], lid_preds: list[torch.Tensor], fusable_pixels_masks: list[torch.Tensor], point_pixel_projections: list[list], rain_preds: list[torch.Tensor], seg_masks: list[torch.Tensor]):
        prepared_cam_preds = []
        prepared_lid_preds = []
        prepared_labels = []
        
        for i in range(len(cam_preds)):
            cam_pred = cam_preds[i]  # [h, w, c]
            lid_pred = lid_preds[i]  # [p, c]
            fusable_pixels_mask = fusable_pixels_masks[i]  # [h, w]
            projections = point_pixel_projections[i]  # List of [p, 5]
            seg_mask = seg_masks[i]  # [h, w]
            
            h, w, c = cam_pred.shape
            fused_lid_pred = torch.zeros((h, w, c), device=lid_pred.device)  # [h, w, c]
            count = torch.zeros((h, w), device=lid_pred.device)  # To track the number of lidar points per pixel
            
            # Iterate over all points and fuse lidar predictions per pixel
            for point_idx, proj in enumerate(projections):
                u, v = int(proj[3]), int(proj[4])  # Extract pixel coordinates
                assert(0 <= u < w and 0 <= v < h), f"Pixel coordinates must be within image dimensions, but are: {u, v, w, h}"
                if count[v, u] == 0:
                    fused_lid_pred[v, u] = lid_pred[point_idx]
                else:
                    fused_lid_pred[v, u] *= lid_pred[point_idx]  # Element-wise multiplication
                count[v, u] += 1
            
            # Filter out non-fusable pixels
            cam_lid_pred_mask = fusable_pixels_mask.bool().unsqueeze(-1).expand(-1, -1, c)  # [h, w, c]
            cam_pred_filtered = cam_pred[cam_lid_pred_mask].reshape(-1, c)  # Keep only valid pixels
            lid_pred_filtered = fused_lid_pred[cam_lid_pred_mask].reshape(-1, c)  # Keep only valid pixels
            
            prepared_labels.append(seg_mask[fusable_pixels_mask])
            
            prepared_cam_preds.append(cam_pred_filtered)
            prepared_lid_preds.append(lid_pred_filtered)

        expanded_rain_preds = [rain_pred.expand(prepared_cam_pred_tensor.shape[0]) for prepared_cam_pred_tensor, rain_pred in zip(prepared_cam_preds, rain_preds)]

        prepared_rain_preds = torch.cat(expanded_rain_preds, dim=0)
        prepared_cam_preds = torch.cat(prepared_cam_preds, dim=0)
        prepared_lid_preds = torch.cat(prepared_lid_preds, dim=0)
        prepared_labels = torch.cat(prepared_labels, dim=0)

        assert(prepared_cam_preds.shape == prepared_lid_preds.shape), f"Prepared camera and lidar predictions must have the same shape, but are: {prepared_cam_preds.shape}, {prepared_lid_preds.shape}"
        return prepared_cam_preds, prepared_lid_preds, prepared_rain_preds, prepared_labels


    # DEFINE PARAMETERS
    scenario="combined"
    datasets=["waymo"]
    classes=get_aki_label_names()
    num_workers = 1
    batch_size = 64
    void_classes = []
    crop_size=(800, 1600)
    grid_cells=(1,1)
    data_from_udm=True
    test_limit = 60 * 5
    split = "test"
    accelerator="gpu" if slurm_job_id else "cpu"
    ignore_index = 255
    num_classes = len(classes)
    clm_path_rain = "clms/multimodal_rain.pt"
    clm_path_sun = "clms/multimodal_sun.pt"
    clm_rain = torch.load(clm_path_rain) if clm_path_rain else None
    clm_sun = torch.load(clm_path_sun) if clm_path_sun else None
    output_image_size = (crop_size[0] // grid_cells[0], crop_size[1] // grid_cells[1])
    rain_probs_from_model = False

    log.info(f"=== PARAMS === ")
    log.info(f"Scenario: {scenario}")
    log.info(f"CLM rain: {clm_path_rain}")
    log.info(f"CLM sun: {clm_path_sun}")
    log.info(f"Split: {split}")
    log.info(f"Rain probs from model: {rain_probs_from_model}")
    log.info(f"Classes: {classes}")
    log.info(f"Test limit: {test_limit}")
    log.info(f"Void classes: {void_classes}")
    log.info(f"Num workers: {num_workers}")
    log.info(f"Batch size: {batch_size}")
    log.info(f"Crop size: {crop_size}")
    log.info(f"Grid cells: {grid_cells}")
    log.info(f"Size grid cell: {output_image_size}")
    log.info(f"Data from UDM: {data_from_udm}")
    log.info(f"Accelerator: {accelerator}")
    log.info(f"Datasets: {datasets}")


    checkpoint = "checkpoints/final_models/sem_img_seg_udm_27_final-1357856.ckpt"

    sis_model = SemanticImageSegmentationModel.load_from_checkpoint(checkpoint,
                                                                    num_classes=len(classes),
                                                                    output_image_size=output_image_size,
                                                                    data_from_udm=data_from_udm)
    
    sls_model = PointNet2.load_from_checkpoint("checkpoints/final_models/sem_lid_seg_udm_27_final-1357876.ckpt",
                                                  num_classes=num_classes,
                                                  data_from_udm=data_from_udm)
    
    wc_model = WeatherClassifier.load_from_checkpoint("checkpoints/final_models/weather_classifier_final-unknown.ckpt",
                                                      data_from_udm=data_from_udm)
    
    data_module = UnifiedDataModule(datasets=datasets,
                                    scenario=scenario,
                                    num_workers=num_workers,
                                    crop_size=crop_size,
                                    void=void_classes,
                                    test_limit=test_limit,
                                    grid_cells=grid_cells,
                                    batch_size=batch_size,
                                    classes=classes)
    data_module.setup()
    
    data_loader = data_module.test_dataloader()

    cam_preds = []
    lid_preds = []
    seg_masks = []
    rain_probabilities = []
    fusable_pixel_masks = []
    point_pixel_projections = []

    for batch_idx, batch in enumerate(data_loader):
        with torch.no_grad():
            cam_probs = sis_model.predict_step(batch, batch_idx)["probs"]
            lid_probs = sls_model.predict_step(batch, batch_idx)["probs"]
            if rain_probs_from_model:
                rain_probs = wc_model.predict_step(batch, batch_idx)["rain_probs"]
            else:
                rain_probs = batch[4]

            cam_preds.extend(torch.unbind(cam_probs))
            lid_preds.extend(lid_probs)
            rain_probabilities.extend(torch.unbind(rain_probs))
            seg_masks.extend(torch.unbind(batch[1]))
            fusable_pixel_masks.extend(torch.unbind(batch[5]))
            point_pixel_projections.extend(batch[6])

    cam_preds_prepared, lid_preds_prepared, rain_preds_prepared, labels_prepared = prepare_data_for_fusion(cam_preds,
                                                                                                           lid_preds,
                                                                                                           fusable_pixel_masks,
                                                                                                           point_pixel_projections,
                                                                                                           rain_probabilities,
                                                                                                           seg_masks)
    
    #preds_fused = fuse_multimodal(cam_preds_prepared, lid_preds_prepared, clm_sun if scenario == "sun" else clm_rain)
    preds_fused = fuse_multimodal_with_rain_probs(cam_preds_prepared, lid_preds_prepared, rain_preds_prepared, clm_rain, clm_sun)

    # Accuracy
    acc = tm.classification.Accuracy(task="multiclass", num_classes=num_classes, ignore_index=ignore_index, average="micro")(preds_fused, labels_prepared)
    log.info(f"Accuracy is: {acc}")

    # F1
    f1_mac = tm.functional.f1_score(preds_fused, labels_prepared, "multiclass", num_classes=num_classes, average="macro")
    f1_per_class = tm.functional.f1_score(preds_fused, labels_prepared, "multiclass", num_classes=num_classes, average=None)
    log.info(f"Macro F1: {f1_mac}")
    log.info(f"F1 scores per class: {f1_per_class}")

    # ECE
    ece = tm.CalibrationError("multiclass", num_classes=num_classes, ignore_index=ignore_index)(preds_fused, labels_prepared)
    log.info(f"ECE: {ece}")

def evaluate_sis_for_model_result_plots(slurm_job_id: int, scenario: str, grid_cells: tuple[int, int]):
    log.info(f"Evaluating SIS Model")
    df = pd.DataFrame(columns=["Scenario", "Grid", "ECE", "Macro F1", "Accuracy"])

    log.info(f"Running evaluation for scenario {scenario} and grid {grid_cells}")
    # DEFINE PARAMETERS
    order_by="weather"
    datasets=["waymo"]
    classes=get_aki_label_names()
    num_workers = 1
    batch_size = 128
    test_limit = 200*5 if scenario=="sun" else 30*5
    void_classes = []
    crop_size=(800, 1600)
    data_from_udm=True
    split = "test"
    accelerator="gpu" if slurm_job_id else "cpu"
    ignore_index = 255
    num_classes = len(classes)
    output_image_size = (crop_size[0] // grid_cells[0], crop_size[1] // grid_cells[1])

    log.info(f"=== PARAMS === ")
    log.info(f"Scenario: {scenario}")
    log.info(f"Split: {split}")
    log.info(f"Order by: {order_by}")
    log.info(f"Batch size: {batch_size}")
    log.info(f"Test limit: {test_limit}")
    log.info(f"Grid cells: {grid_cells}")
    log.info(f"Output image size: {output_image_size}")
    log.info(f"Accelerator: {accelerator}")

    checkpoint = "checkpoints/final_models/sem_img_seg_udm_27_final-1357856.ckpt"

    wandb_logger = WandbLogger(name=f"sis-raw-{scenario}-{split}-{slurm_job_id}",
                                log_model="all" if slurm_job_id else False,
                                save_dir="logs/wandb/eval/",
                                project="eval",
                                offline=False if slurm_job_id else True)

    model = SemanticImageSegmentationModel.load_from_checkpoint(checkpoint,
                                                                num_classes=len(classes),
                                                                crop_size=crop_size,
                                                                output_image_size=output_image_size,
                                                                data_from_udm=data_from_udm)

    data_module = UnifiedDataModule(datasets=datasets,
                                    scenario=scenario,
                                    num_workers=num_workers,
                                    order_by=order_by,
                                    test_limit=test_limit,
                                    crop_size=crop_size,
                                    void=void_classes,
                                    grid_cells=grid_cells,
                                    batch_size=batch_size,
                                    classes=classes)
    data_module.setup()

    trainer = L.Trainer(logger=wandb_logger,
                        accelerator=accelerator,
                        enable_progress_bar=False)
    data_loader = data_module.test_dataloader()

    predictions = trainer.predict(model, data_loader)
    preds = torch.cat([pred['probs'] for pred in predictions], dim=0)
    labels = torch.cat([pred['labels'] for pred in predictions], dim=0)

    preds_n, preds_h, preds_w, preds_c = preds.shape
    preds_flattened = torch.reshape(preds, (preds_n * preds_h * preds_w, preds_c))
    labels_flattened = torch.flatten(labels)

    # Accuracy
    acc = tm.classification.Accuracy(task="multiclass", num_classes=num_classes, ignore_index=ignore_index, average="micro")(preds_flattened, labels_flattened)
    log.info(f"Accuracy is: {acc}")

    # F1
    f1_mac = tm.functional.f1_score(preds_flattened, labels_flattened, "multiclass", num_classes=num_classes, average="macro")
    #f1_per_class = tm.functional.f1_score(preds_flattened, labels_flattened, "multiclass", num_classes=num_classes, average=None)
    log.info(f"Macro F1: {f1_mac}")
    #log.info(f"F1 scores per class: {f1_per_class}")

    # ECE
    ece = tm.CalibrationError("multiclass", num_classes=num_classes, ignore_index=ignore_index)(preds_flattened, labels_flattened)
    log.info(f"ECE: {ece}")

    log.info(f"Total elements yielded by data module: {data_module.prepared_elems}")

    df.loc[-1] = [scenario, grid_cells, ece, f1_mac, acc]

    log.info(f"All results: {df}")

def evaluate_sls_for_model_result_plots(slurm_job_id: int, scenario: str, grid_cells: tuple[int, int]):
    log.info(f"Evaluating SLS Model")
    df = pd.DataFrame(columns=["Scenario", "Grid", "ECE", "Macro F1", "Accuracy"])

    
    log.info(f"Running evaluation for scenario {scenario} and grid {grid_cells}")
    # DEFINE PARAMETERS
    order_by="weather"
    datasets=["waymo"]
    classes=get_aki_label_names()
    num_workers = 1
    batch_size = 128
    test_limit = 200*5 if scenario=="sun" else 30*5
    void_classes = []
    crop_size=(800, 1600)
    data_from_udm=True
    split = "test"
    accelerator="gpu" if slurm_job_id else "cpu"
    ignore_index = 255
    num_classes = len(classes)
    output_image_size = (crop_size[0] // grid_cells[0], crop_size[1] // grid_cells[1])

    log.info(f"=== PARAMS === ")
    log.info(f"Scenario: {scenario}")
    log.info(f"Split: {split}")
    log.info(f"Order by: {order_by}")
    log.info(f"Batch size: {batch_size}")
    log.info(f"Test limit: {test_limit}")
    log.info(f"Grid cells: {grid_cells}")
    log.info(f"Output image size: {output_image_size}")
    log.info(f"Accelerator: {accelerator}")

    checkpoint = "checkpoints/final_models/sem_lid_seg_udm_27_final-1357876.ckpt"

    wandb_logger = WandbLogger(name=f"sls-raw-{scenario}-{split}-{slurm_job_id}",
                                log_model="all" if slurm_job_id else False,
                                save_dir="logs/wandb/eval/",
                                project="eval",
                                offline=False if slurm_job_id else True)

    model = PointNet2.load_from_checkpoint(checkpoint,
                                            num_classes=len(classes),           
                                            data_from_udm=data_from_udm)

    data_module = UnifiedDataModule(datasets=datasets,
                                    scenario=scenario,
                                    num_workers=num_workers,
                                    order_by=order_by,
                                    test_limit=test_limit,
                                    crop_size=crop_size,
                                    void=void_classes,
                                    grid_cells=grid_cells,
                                    batch_size=batch_size,
                                    classes=classes)
    data_module.setup()

    trainer = L.Trainer(logger=wandb_logger,
                        accelerator=accelerator,
                        enable_progress_bar=False)
    data_loader = data_module.test_dataloader()

    predictions = trainer.predict(model, data_loader)
    probs_list = [probs.view(-1, probs.shape[-1]) for pred in predictions for probs in pred['probs']]
    labels_list = [labels.view(-1) for pred in predictions for labels in pred['labels']]
    preds = torch.cat(probs_list, dim=0)
    labels = torch.cat(labels_list, dim=0)

    log.info(f"Shape of predictions: {preds.shape}")
    log.info(f"Shape of labels: {labels.shape}")

    # Accuracy
    acc = tm.classification.Accuracy(task="multiclass", num_classes=num_classes, ignore_index=ignore_index, average="micro")(preds, labels)
    log.info(f"Accuracy is: {acc}")

    # F1
    f1_mac = tm.functional.f1_score(preds, labels, "multiclass", num_classes=num_classes, average="macro")
    #f1_per_class = tm.functional.f1_score(preds_flattened, labels_flattened, "multiclass", num_classes=num_classes, average=None)
    log.info(f"Macro F1: {f1_mac}")
    #log.info(f"F1 scores per class: {f1_per_class}")

    # ECE
    ece = tm.CalibrationError("multiclass", num_classes=num_classes, ignore_index=ignore_index)(preds, labels)
    log.info(f"ECE: {ece}")

    log.info(f"Total elements yielded by data module: {data_module.prepared_elems}")

    df.loc[-1] = [scenario, grid_cells, ece, f1_mac, acc]

    log.info(f"All results: {df}")

def evaluate_wc_for_model_result_plots(slurm_job_id:int, scenario: str, grid_cells: tuple[int, int]):
    log.info(f"Evaluating WC Model")
    df = pd.DataFrame(columns=["Scenario", "Grid", "ECE", "Macro F1", "Accuracy"])

 
    log.info(f"Running evaluation for scenario {scenario} and grid {grid_cells}")
    # DEFINE PARAMETERS
    order_by="weather"
    datasets=["waymo"]
    classes=get_aki_label_names()
    num_workers = 1
    batch_size = 128
    test_limit = 200*5 if scenario=="sun" else 30*5
    void_classes = []
    crop_size=(800, 1600)
    data_from_udm=True
    split = "test"
    accelerator="gpu" if slurm_job_id else "cpu"
    ignore_index = 255
    num_classes = len(classes)
    output_image_size = (crop_size[0] // grid_cells[0], crop_size[1] // grid_cells[1])

    log.info(f"=== PARAMS === ")
    log.info(f"Scenario: {scenario}")
    log.info(f"Split: {split}")
    log.info(f"Order by: {order_by}")
    log.info(f"Batch size: {batch_size}")
    log.info(f"Test limit: {test_limit}")
    log.info(f"Grid cells: {grid_cells}")
    log.info(f"Output image size: {output_image_size}")
    log.info(f"Accelerator: {accelerator}")

    checkpoint = "checkpoints/final_models/weather_classifier_final-unknown.ckpt"

    wandb_logger = WandbLogger(name=f"sls-raw-{scenario}-{split}-{slurm_job_id}",
                                log_model="all" if slurm_job_id else False,
                                save_dir="logs/wandb/eval/",
                                project="eval",
                                offline=False if slurm_job_id else True)

    model = WeatherClassifier.load_from_checkpoint(checkpoint,       
                                            data_from_udm=data_from_udm)

    data_module = UnifiedDataModule(datasets=datasets,
                                    scenario=scenario,
                                    num_workers=num_workers,
                                    order_by=order_by,
                                    test_limit=test_limit,
                                    crop_size=crop_size,
                                    void=void_classes,
                                    grid_cells=grid_cells,
                                    batch_size=batch_size,
                                    classes=classes)
    data_module.setup()

    trainer = L.Trainer(logger=wandb_logger,
                        accelerator=accelerator,
                        enable_progress_bar=False)
    data_loader = data_module.test_dataloader()

    predictions = trainer.predict(model, data_loader)
    preds = torch.cat([pred['rain_probs'] for pred in predictions], dim=0)
    labels = torch.cat([pred['labels'] for pred in predictions], dim=0)

    log.info(f"Shape of predictions: {preds.shape}")
    log.info(f"Shape of labels: {labels.shape}")

    # Accuracy
    acc = tm.classification.Accuracy(task="binary", ignore_index=ignore_index, average="micro")(preds, labels)
    log.info(f"Accuracy is: {acc}")

    # F1
    f1_mac = tm.functional.f1_score(preds, labels, "binary")
    #f1_per_class = tm.functional.f1_score(preds_flattened, labels_flattened, "multiclass", num_classes=num_classes, average=None)
    log.info(f"Macro F1: {f1_mac}")
    #log.info(f"F1 scores per class: {f1_per_class}")

    # ECE
    ece = tm.CalibrationError("binary", ignore_index=ignore_index)(preds, labels)
    log.info(f"ECE: {ece}")

    log.info(f"Total elements yielded by data module: {data_module.prepared_elems}")

    df.loc[-1] = [scenario, grid_cells, ece, f1_mac, acc]

    log.info(f"All results: {df}")

if __name__ == "__main__":
    slurm_job_id = sys.argv[1] if len(sys.argv) > 1 else None
    
    evaluate_sem_img_seg_with_udm(slurm_job_id)
    
    #evaluate_multi_modal(slurm_job_id)

    """
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

    #scenario = "rain"
    #grid = (1,1)
    #model = "sis"
    for model in ["sis", "sls", "wc"]:
        for scenario in ["rain", "sun", "combined"]:
            for grid in [(1,1), (2,2), (4,4)]:
                if model == "sis":
                    evaluate_sis_for_model_result_plots(slurm_job_id, scenario, grid)
                elif model == "sls":
                    evaluate_sls_for_model_result_plots(slurm_job_id, scenario, grid)
                elif model == "wc":
                    evaluate_wc_for_model_result_plots(slurm_job_id, scenario, grid)
                else:
                    raise ValueError(f"Model {model} is not supported.")"
    """
