import torch
import logging

from lightning import Trainer
from rich.logging import RichHandler
from models.semantic_image_segmentation import SemanticImageSegmentationModel
from data_modules.semantic_image_segmentation_datamodule import SemanticImageSegmentationDataModule

from aki_labels import get_aki_label_names

logging.basicConfig(level="INFO", format="[%(filename)s:%(lineno)s - %(funcName)20s()] %(message)s", datefmt="[%X]", handlers=[RichHandler()])

log = logging.getLogger("rich")

def main():
    log.info(f"Training script started")

    scenario = "dayrain"
    datasets = ["all"]
    classes = get_aki_label_names()
    void_classes = ["void", "static"]

    datamodule = SemanticImageSegmentationDataModule(scenario=scenario,
                                         datasets=datasets,
                                         classes=classes,
                                         void=void_classes)
    log.info(f"Akiset datamodule created")

    log.info(f"Length of aki_label_names: {len(get_aki_label_names())}")
    log.info(f"Akiset_datamodule.classes (valid classes): {len(datamodule.classes)}")

    valid_classes = datamodule.classes
    log.info(f"Valid classes are: {valid_classes} ")

    segmentation_model = SemanticImageSegmentationModel(len(valid_classes)).to(memory_format=torch.channels_last)
    log.info(f"Segmentation model created")


    trainer = Trainer()
    log.info(f"Starting training")
    trainer.fit(segmentation_model, datamodule)
    trainer.validate(segmentation_model, datamodule)
    trainer.test(segmentation_model, datamodule)

if __name__ == "__main__":
    main()