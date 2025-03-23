import torch
import logging
import sys
import uuid

from lightning import Trainer
from rich.logging import RichHandler
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from models.semantic_image_segmentation import SemanticImageSegmentationModel
from data_modules.unified_datamodule import UnifiedDataModule
from typing import Optional

from utils.aki_labels import get_aki_label_names

logging.basicConfig(level="INFO", format="[%(filename)s:%(lineno)s - %(funcName)20s()] %(message)s", datefmt="[%X]")

log = logging.getLogger("rich")

def main(slurm_job_id: Optional[int]):

    # DEFINE PARAMETERS
    # Dataset and dataloader
    scenario = "all"
    datasets = ["waymo"]
    order_by = "weather"
    classes = get_aki_label_names()
    void_classes = []
    train_limit = 30_000
    val_limit = 10_000
    test_limit = 10_000
    num_workers = 32
    crop_size = (800, 1600)
    batch_size = 128

    # Training
    max_epochs = 100

    # Model 
    data_from_udm = True

    # Create data module
    datamodule = UnifiedDataModule(scenario=scenario,
                                   order_by=order_by,
                                   datasets=datasets,
                                   classes=classes,
                                   void=void_classes,
                                   crop_size=crop_size,
                                   batch_size=batch_size,
                                   train_limit=train_limit,
                                   val_limit=val_limit,
                                   test_limit=test_limit,
                                   num_workers=num_workers)
    
    valid_classes = datamodule.classes

    # Create logger
    wandb_logger = WandbLogger(name=f"img_sem_seg_udm-{slurm_job_id if slurm_job_id else uuid.uuid4()}",
                               log_model="all",
                               save_dir="logs/wandb/semantic_image_segmentation/",
                               project="semantic_image_segmentation")

    # Create model
    segmentation_model = SemanticImageSegmentationModel(len(valid_classes),
                                                        data_from_udm=data_from_udm,
                                                        crop_size=crop_size,
                                                        train_epochs=max_epochs).to(memory_format=torch.channels_last)
    #segmentation_model = SemanticImageSegmentationModel.load_from_checkpoint("checkpoints/semantic_image_segmentation/sis_final_model.ckpt")

    checkpoint_callback = ModelCheckpoint(dirpath="checkpoints/semantic_image_segmentation_with_udm/",
                                          filename=f"{slurm_job_id}"+'{epoch:02d}-{val_loss:.5f}',
                                          monitor="val_loss",
                                          save_top_k=-1,
                                          every_n_epochs=1)


    # Create trainer and start training
    trainer = Trainer(max_epochs=max_epochs,
                      callbacks=[checkpoint_callback],
                      logger = wandb_logger,
                      enable_progress_bar=False)
    trainer.fit(segmentation_model, datamodule)
    trainer.validate(segmentation_model, datamodule)
    trainer.test(segmentation_model, datamodule)

if __name__ == "__main__":
   main(sys.argv[1] if len(sys.argv) > 1 else None)