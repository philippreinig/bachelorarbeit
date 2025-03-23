import torch
import logging
import sys
import uuid

import torchinfo as ti
import lightning as L

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from models.semantic_lidar_segmentation import PointNet2
from data_modules.unified_datamodule import UnifiedDataModule
from utils.aki_labels import aki_labels
from typing import Optional

from utils.aki_labels import get_aki_label_names
from utils.visualization import visualize_pcl_matplotlib

logging.basicConfig(level=logging.INFO, format="[%(filename)s:%(lineno)s - %(funcName)s] %(message)s", datefmt="[%X]")

log = logging.getLogger(__name__)

def main(slurm_job_id: Optional[int]):
    log.info(f"PyTorch version: {torch.__version__}")
    log.info(f"Cuda version: {torch.version.cuda}")

    torch.set_float32_matmul_precision('medium')

    # DEFINE PARAMETERS
    # Dataset and dataloader
    scenario = "all"
    datasets = ["waymo"]
    order_by = "weather"
    batch_size = 32
    downsampled_pointcloud_size = None
    classes = get_aki_label_names()
    void_classes = []
    train_limit = 10_000
    val_limit = 3_000
    test_limit = 1_000
    num_workers = 16
    crop_size=(800, 1600)
    grid_cells=(1,1)

    # Model
    data_from_udm=True

    # Training
    max_epochs = 100
    precision="16-mixed"
    accelerator="gpu"
    check_val_every_n_epoch=1

    # Create loggers
    wandb_logger = WandbLogger(name=f"lid_sem_seg_udm-{slurm_job_id if slurm_job_id else uuid.uuid4()}",
                               log_model="all",
                               save_dir="logs/wandb/semantic_lidar_segmentation_with_udm/",
                               project="semantic_lidar_segmentation_with_udm")

    # Create data module
    datamodule = UnifiedDataModule(scenario=scenario,
                                   order_by=order_by,
                                   batch_size=batch_size,
                                   downsampled_pointcloud_size=downsampled_pointcloud_size,
                                   crop_size=crop_size,
                                   grid_cells=grid_cells,
                                   datasets=datasets,
                                   classes=classes,
                                   void=void_classes,
                                   train_limit=train_limit,
                                   val_limit=val_limit,
                                   test_limit=test_limit,
                                   num_workers=num_workers)

    valid_classes = datamodule.classes
    amt_valid_classes = len(valid_classes)
    
    log.info(f"There are {amt_valid_classes} valid classes of the total {len(get_aki_label_names())} classes: {valid_classes}")

    checkpoint_callback = ModelCheckpoint(dirpath="checkpoints/semantic_lidar_segmentation_with_udm/",   
                                          filename=f"{slurm_job_id}"+'{epoch:02d}-{val_loss:.5f}',
                                          monitor="val_loss",
                                          save_top_k=-1,
                                          every_n_epochs=1)

    # Create model
    #segmentation_model = PointNet2(len(aki_labels),
    #                               train_epochs=max_epochs,
    #                               data_from_udm=data_from_udm)
    segmentation_model = PointNet2.load_from_checkpoint("checkpoints/semantic_lidar_segmentation_with_udm/1357852epoch=10-val_loss=1.57201.ckpt",
                                                        num_classes=len(aki_labels),
                                                        train_epochs=max_epochs,
                                                        data_from_udm=data_from_udm)
    
    
    ti.summary(segmentation_model, (batch_size, 30_000, 3))
    
    # Create trainer and start training
    trainer = L.Trainer(max_epochs=max_epochs,
                        logger=wandb_logger,
                        callbacks=[checkpoint_callback],
                        precision=precision,
                        enable_progress_bar=False,
                        accelerator=accelerator,
                        check_val_every_n_epoch=check_val_every_n_epoch)

    trainer.fit(segmentation_model, datamodule)
    trainer.validate(segmentation_model, datamodule)
    trainer.test(segmentation_model, datamodule)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)