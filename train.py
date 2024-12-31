import data
from model.model import SegmentationModel
from lightning import Trainer
from akiset_datamodule import AkisetDataModule
import torch
from data import get_aki_label_names
import logging
from rich.logging import RichHandler

logging.basicConfig(level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()])

log = logging.getLogger("rich")

def main():
    log.info(f"Training script started")

    akiset_datamodule = AkisetDataModule("rain",
                                         ["all"],
                                         classes=data.get_aki_label_names(["void", "static"]),
                                         void=["void", "static"])
    log.info(f"Akiset datamodule created")

    valid_classes = akiset_datamodule.classes
    log.info(f"Valid classes are: {valid_classes} ")

    segmentation_model = SegmentationModel(len(valid_classes)).to(memory_format=torch.channels_last)
    log.info(f"Segmentation model created")


    trainer = Trainer()
    log.info(f"Starting training")
    trainer.fit(segmentation_model, akiset_datamodule)
    trainer.validate(segmentation_model, akiset_datamodule)
    trainer.test(segmentation_model, akiset_datamodule)

if __name__ == "__main__":
    main()