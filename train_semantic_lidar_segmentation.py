import torch
import logging

import torchinfo as ti
import lightning as L

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from models.semantic_lidar_segmentation import PointNet2
from data_modules.semantic_lidar_segmentation_datamodule import SemanticLidarSegmentationDataModule
from utils.aki_labels import aki_labels

from utils.aki_labels import get_aki_label_names
from utils.visualization import visualize_pcl_matplotlib

logging.basicConfig(level="INFO", format="[%(filename)s:%(lineno)s - %(funcName)s] %(message)s", datefmt="[%X]")

log = logging.getLogger("rich")

def main():
    log.info(f"Training script started")

    log.info(f"PyTorch version: {torch.__version__}")
    log.info(f"Cuda version: {torch.version.cuda}")

    # DEFINE PARAMETERS
    # Dataset
    scenario = "all"
    datasets = ["waymo"]
    order_by = "weather"
    limit = 10_000
    batch_size = 3
    downsampled_pointcloud_size = 16384
    classes = get_aki_label_names()
    void_classes = ["void", "static"]

    # Model

    # Training
    max_epochs = 10

    # Create logger
    wandb_logger = WandbLogger(log_model="all", save_dir="logs/wandb/semantic_lidar_segmentation/", project="semantic_lidar_segmentation")

    # Create data module
    datamodule = SemanticLidarSegmentationDataModule(scenario=scenario,
                                                     order_by=order_by,
                                                     limit=limit,
                                                     batch_size=batch_size,
                                                     datasets=datasets,
                                                     classes=classes,
                                                     void=void_classes)

    valid_classes = datamodule.classes
    amt_valid_classes = len(valid_classes)
    
    log.info(f"There are {amt_valid_classes} valid classes of the total {len(get_aki_label_names())} classes: {valid_classes}")

    checkpoint_callback = ModelCheckpoint(dirpath="checkpoints/semantic_lidar_segmentation/",
                                          filename='{epoch:02d}-{val_loss:.5f}',
                                          monitor='val_loss',
                                          save_top_k=-1,  # Save all checkpoints
                                          every_n_epochs=1)

    # Create model
    segmentation_model = PointNet2(len(aki_labels), train_epochs=max_epochs)
    ti.summary(segmentation_model, (batch_size, downsampled_pointcloud_size if downsampled_pointcloud_size else 143_000, 3))
    
    # Create trainer and start training
    trainer = L.Trainer(max_epochs=max_epochs, logger = wandb_logger, callbacks=[checkpoint_callback], precision="16-mixed")

    trainer.fit(segmentation_model, datamodule)
    trainer.validate(segmentation_model, datamodule)
    trainer.test(segmentation_model, datamodule)

if __name__ == "__main__":
    main()