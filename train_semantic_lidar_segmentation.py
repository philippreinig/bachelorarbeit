import torch
import logging

import torchsummary as ts

from lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from models.semantic_lidar_segmentation import PointNet2
from data_modules.semantic_lidar_segmentation_datamodule import SemanticLidarSegmentationDataModule
from utils.aki_labels import aki_labels

from utils.aki_labels import get_aki_label_names
from utils.visualization import visualize_pcl_matplotlib, visualize_pcl_open3d

logging.basicConfig(level="INFO", format="[%(filename)s:%(lineno)s - %(funcName)20s()] %(message)s", datefmt="[%X]")

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
    batch_size = 8
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

    log.info(f"Memory allocated after model initialization: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")

    valid_classes = datamodule.classes
    amt_valid_classes = len(valid_classes)
    
    log.info(f"There are {amt_valid_classes} valid classes of the total {len(get_aki_label_names())} classes: {valid_classes}")

    # Create model
    segmentation_model = PointNet2(len(aki_labels), train_epochs=max_epochs)
    #ts.summary(segmentation_model, (100000, 3))
    
    # Create trainer and start training
    trainer = Trainer(max_epochs=max_epochs, logger = wandb_logger)
    log.info(f"Starting training")
    trainer.fit(segmentation_model, datamodule)
    trainer.validate(segmentation_model, datamodule)
    trainer.test(segmentation_model, datamodule)

if __name__ == "__main__":
    main()