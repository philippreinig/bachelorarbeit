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

from fusion.clm import create_unimodal_normalized_clm, create_multimodal_normalized_clm

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

def create_clm_multimodal(slurm_job_id: int):
    def prepare_data_for_multimomdal_clm_creation(_cam_preds: list[torch.Tensor], _lid_preds: list[torch.Tensor], _segmentation_masks: list[torch.Tensor], _fusable_pixel_masks: list[torch.Tensor], _point_pixel_projections: list[list]):
        assert(len(_cam_preds) == len(_lid_preds) == len(_fusable_pixel_masks) == len(_point_pixel_projections)), f"Amount of predictions, masks and projections must be equal, but are: {len(_cam_preds), len(_lid_preds), len(_fusable_pixel_masks), len(_point_pixel_projections)}"

        amt_imgs = len(_cam_preds)

        cam_preds_per_img = []
        lid_preds_per_img = []
        labels_per_img = []

        for i in range(amt_imgs):
            cam_preds_img_full = _cam_preds[i]
            lid_preds_img = _lid_preds[i]
            fusable_pixel_masks_img = _fusable_pixel_masks[i]
            point_pixel_projections_img = _point_pixel_projections[i]
            segmentation_mask = _segmentation_masks[i]

            cam_preds_img_only_projectable = []
            corresponding_labels_img = []

            for pixel_projection in point_pixel_projections_img:
                v, u = int(pixel_projection[3]), int(pixel_projection[4])
                cam_preds_img_only_projectable.append(cam_preds_img_full[u,v])
                assert(torch.isclose(cam_preds_img_full[u, v].sum(), torch.tensor(1, dtype=torch.float)))
                corresponding_labels_img.append(segmentation_mask[u,v].unsqueeze(0))

            lid_preds_per_img.append(lid_preds_img)

            cam_preds_img_only_projectable_tensor = torch.stack(cam_preds_img_only_projectable)

            cam_preds_per_img.append(cam_preds_img_only_projectable_tensor)
            
            corresponding_labels_img = torch.cat(corresponding_labels_img, dim=0)
            labels_per_img.append(corresponding_labels_img)

            assert(lid_preds_img.shape == cam_preds_img_only_projectable_tensor.shape), f"Shapes of lidar and camera predictions per img must be equal, but are: {lid_preds_img.shape, cam_preds_img_only_projectable_tensor.shape}"
            assert(lid_preds_img.shape[0] == cam_preds_img_only_projectable_tensor.shape[0] == corresponding_labels_img.shape[0]), f"Amount of predictions in lidar and camera predictions must be equal to amount of labels in img, but are: {lid_preds_img.shape[0], cam_preds_img_only_projectable_tensor.shape[0], labels_per_img.shape[0]}"
            assert (lid_preds_img.shape[0] >= fusable_pixel_masks_img.sum() and 
                    cam_preds_img_only_projectable_tensor.shape[0] >= fusable_pixel_masks_img.sum()), \
                    f"Amount of predictions in lidar and camera predictions per img must be >= sum of fusable pixel mask, but are: {lid_preds_img.shape[0], cam_preds_img_only_projectable_tensor.shape[0], fusable_pixel_masks_img.sum()}"
                    
        assert(len(cam_preds_per_img) == len(lid_preds_per_img)), f"Amount of camera and lidar predictions per img must be equal, but are: {len(cam_preds_per_img), len(lid_preds_per_img)}"

        cam_preds_tensor = torch.concat(cam_preds_per_img, dim=0)
        lid_preds_tensor = torch.concat(lid_preds_per_img, dim=0)
        labels_tensor = F.one_hot(torch.concat(labels_per_img, dim=0), num_classes=27).to(torch.float)

        assert(cam_preds_tensor.shape == lid_preds_tensor.shape), f"Shapes of camera and lidar predictions tensor must be equal, but are: {cam_preds_tensor.shape, lid_preds_tensor.shape}"

        return cam_preds_tensor, lid_preds_tensor, labels_tensor

    # DEFINE PARAMETERS
    scenario="rain"
    datasets=["waymo"]
    classes=get_aki_label_names()
    num_workers = 1
    batch_size = 128
    void_classes = []
    crop_size=(800, 1600)
    grid_cells=(1,1)
    data_from_udm=True
    val_limit = -1
    split = "val"
    accelerator="gpu" if slurm_job_id else "cpu"
    num_classes = len(classes)
    export_path = f"clms/multimodal_{scenario}.pt"

    log.info(f"=== PARAMS === ")
    log.info(f"Scenario: {scenario}")
    log.info(f"Split: {split}")
    log.info(f"Classes: {classes}")
    log.info(f"Val limit: {val_limit}")
    log.info(f"Export path: {export_path}")
    log.info(f"Void classes: {void_classes}")
    log.info(f"Num workers: {num_workers}")
    log.info(f"Batch size: {batch_size}")
    log.info(f"Crop size: {crop_size}")
    log.info(f"Grid cells: {grid_cells}")
    log.info(f"Data from UDM: {data_from_udm}")
    log.info(f"Accelerator: {accelerator}")
    log.info(f"Datasets: {datasets}")

    sis_model = SemanticImageSegmentationModel.load_from_checkpoint("checkpoints/final_models/sem_img_seg_udm_27_final-1357856.ckpt",
                                                                    num_classes=len(classes),
                                                                    crop_size=crop_size,
                                                                    data_from_udm=data_from_udm)
    
    sls_model = PointNet2.load_from_checkpoint("checkpoints/final_models/sem_lid_seg_udm_27_final-1357876.ckpt",
                                                  num_classes=num_classes,
                                                  data_from_udm=data_from_udm)
    
    data_module = UnifiedDataModule(datasets=datasets,
                                    scenario=scenario,
                                    num_workers=num_workers,
                                    crop_size=crop_size,
                                    void=void_classes,
                                    val_limit=val_limit,
                                    grid_cells=grid_cells,
                                    batch_size=batch_size,
                                    classes=classes)
    data_module.setup()

    data_loader = data_module.val_dataloader()

    sis_predictions = []
    sls_predictions = []
    segmentation_masks = []
    rain_probabilities = []
    fusable_pixel_masks = []
    point_pixel_projections = []

    for batch_idx, batch in enumerate(data_loader):
        with torch.no_grad():
            sis_probs = sis_model.predict_step(batch, batch_idx)["probs"]

            sls_probs = sls_model.predict_step(batch, batch_idx)["probs"]

            sis_predictions.extend(torch.unbind(sis_probs))
            sls_predictions.extend(sls_probs)
            segmentation_masks.extend(torch.unbind(batch[1]))
            fusable_pixel_masks.extend(torch.unbind(batch[5]))
            point_pixel_projections.extend(batch[6])

        log.info(f"Batch {batch_idx} done")

    cam_preds_fusable, lid_preds_fusable, labels_fusable = prepare_data_for_multimomdal_clm_creation(sis_predictions, sls_predictions, segmentation_masks, fusable_pixel_masks, point_pixel_projections)

    mutlimodal_clm = create_multimodal_normalized_clm(cam_preds_fusable, lid_preds_fusable, labels_fusable)

    torch.save(mutlimodal_clm, f"{export_path}")

    log.info(f"Succesfully created multimodal CLM for scenario {scenario}. Exported to {export_path}")

if __name__ == "__main__":
    slurm_job_id = sys.argv[1] if len(sys.argv) > 1 else None
    #create_clm_img()
    create_clm_multimodal(slurm_job_id)