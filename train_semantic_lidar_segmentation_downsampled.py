import torch
import logging
import sys
import uuid

import torchinfo as ti
import lightning as L

from typing import Optional
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from models.semantic_lidar_segmentation import PointNet2
from data_modules.semantic_lidar_segmentation_downsampled_clouds_datamodule import SemanticLidarSegmentationDownsampledCloudsDataModule
from utils.aki_labels import aki_labels

from utils.aki_labels import get_aki_label_names
from utils.visualization import visualize_pcl_matplotlib

logging.basicConfig(level=logging.INFO, format="[%(filename)s:%(lineno)s - %(funcName)s] %(message)s", datefmt="[%X]")

log = logging.getLogger(__name__)

def main(slurm_job_id: Optional[int]):
    log.info(f"PyTorch version: {torch.__version__}")
    log.info(f"Cuda version: {torch.version.cuda}")

    torch.set_float32_matmul_precision('medium')

    # DEFINE PARAMETERS
    # Dataset
    scenario = "all"
    datasets = ["waymo"]
    order_by = "weather"
    train_batch_size = 32
    val_batch_size = 3
    downsampled_pointcloud_size = 16000
    classes = get_aki_label_names()
    void_classes = ["void", "static"]
    train_limit = 7_000
    val_limit = 2_000
    test_limit = 1_000

    # Model
    set_abstraction_radius_1 = 0.33
    set_abstraction_radius_2 = 0.15
    set_abstration_ratio_1 = 0.33
    set_abstraction_ratio_2 = 0.15

    # Training
    max_epochs = 30

    # Create loggers
    wandb_logger = WandbLogger(name=f"ldr_smtc_sgmttn_dwnsmpld-{slurm_job_id if slurm_job_id else uuid.uuid4()}",
                               log_model="all",
                               save_dir="logs/wandb/semantic_lidar_segmentation_downsampled/",
                               project="semantic_lidar_segmentation")

    # Create data module
    datamodule = SemanticLidarSegmentationDownsampledCloudsDataModule(scenario=scenario,
                                                                      order_by=order_by,
                                                                      train_batch_size=train_batch_size,
                                                                      val_batch_size=val_batch_size,
                                                                      downsampled_pointcloud_size=downsampled_pointcloud_size,
                                                                      datasets=datasets,
                                                                      classes=classes,
                                                                      void=void_classes,
                                                                      train_limit=train_limit,
                                                                      val_limit=val_limit,
                                                                      test_limit=test_limit)

    valid_classes = datamodule.classes
    amt_valid_classes = len(valid_classes)
    
    log.info(f"There are {amt_valid_classes} valid classes of the total {len(get_aki_label_names())} classes: {valid_classes}")

    checkpoint_callback = ModelCheckpoint(dirpath="checkpoints/semantic_lidar_segmentation_downsampled/",
                                          filename=f"{slurm_job_id}"+'{epoch:02d}-{validation/loss:.5f}',
                                          monitor="validation/loss",
                                          save_top_k=-1,
                                          every_n_epochs=1)

    # Create model
    segmentation_model = PointNet2(len(aki_labels),
                                   set_abstraction_ratio_1=set_abstration_ratio_1,
                                   set_abstraction_ratio_2=set_abstraction_ratio_2,
                                   set_abstraction_radius_1=set_abstraction_radius_1,
                                   set_abstraction_radius_2=set_abstraction_radius_2,
                                   train_epochs=max_epochs)
    
    ti.summary(segmentation_model, (train_batch_size, downsampled_pointcloud_size if downsampled_pointcloud_size else 143_000, 3))
    
    # Create trainer and start training
    trainer = L.Trainer(max_epochs=max_epochs,
                        logger = wandb_logger,
                        callbacks=[checkpoint_callback],
                        precision="16-mixed",
                        enable_progress_bar=False,
                        accelerator="cpu",
                        devices=1,
                        num_nodes=1,
                        strategy="ddp")
    log.info(f"Lightning trainer uses {trainer.num_devices} gpus")

    trainer.fit(segmentation_model, datamodule)
    trainer.validate(segmentation_model, datamodule)
    trainer.test(segmentation_model, datamodule)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
