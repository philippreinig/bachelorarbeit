import torch
import logging

from lightning import Trainer
from rich.logging import RichHandler
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from models.semantic_image_segmentation import SemanticImageSegmentationModel
from data_modules.semantic_image_segmentation_datamodule import SemanticImageSegmentationDataModule

from utils.aki_labels import get_aki_label_names

logging.basicConfig(level="INFO", format="[%(filename)s:%(lineno)s - %(funcName)20s()] %(message)s", datefmt="[%X]")

log = logging.getLogger("rich")

def main():
    log.info(f"Training script started")

    # DEFINE PARAMETERS
    # Dataset
    scenario = "all"
    datasets = ["waymo"]
    order_by = "weather"
    classes = get_aki_label_names()
    void_classes = ["void", "static"]
    train_limit = 7_000
    val_limit = 2_000
    test_limit = 1_000

    # Training
    max_epochs = 100

    # Create data module
    datamodule = SemanticImageSegmentationDataModule(scenario=scenario,
                                                     order_by=order_by,
                                                     datasets=datasets,
                                                     classes=classes,
                                                     void=void_classes,
                                                     train_limit=train_limit,
                                                     val_limit=val_limit,
                                                     test_limit=test_limit)
    valid_classes = datamodule.classes

    log.info(f"Length of aki_label_names: {len(get_aki_label_names())}")
    log.info(f"Akiset_datamodule.classes (valid classes): {len(datamodule.classes)}")
    log.info(f"Valid classes are: {valid_classes} ")

    # Create logger
    wandb_logger = WandbLogger(log_model="all", save_dir="logs/wandb/semantic_image_segmentation/", project="semantic_image_segmentation")

    # Create model
    #segmentation_model = SemanticImageSegmentationModel(len(valid_classes), train_epochs=max_epochs).to(memory_format=torch.channels_last)
    segmentation_model = SemanticImageSegmentationModel.load_from_checkpoint("checkpoints/semantic_image_segmentation/sis_final_model.ckpt")

    checkpoint_callback = ModelCheckpoint(dirpath="checkpoints/semantic_image_segmentation/",
                                          filename='{epoch:02d}-{val_loss:.5f}',
                                          monitor="validation/loss",
                                          save_top_k=-1,
                                          every_n_epochs=1)


    # Create trainer and start training
    trainer = Trainer(max_epochs=max_epochs,
                      callbacks=[checkpoint_callback],
                      logger = wandb_logger,
                      enable_progress_bar=False)
    log.info(f"Starting training")
  #  trainer.fit(segmentation_model, datamodule)
    trainer.validate(segmentation_model, datamodule)
    trainer.test(segmentation_model, datamodule)

if __name__ == "__main__":
    main()