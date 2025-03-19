import logging 

import lightning as L

from evaluation.evaluate_sis import evaluate_semantic_image_segmentation
from models.semantic_image_segmentation import SemanticImageSegmentationModel
from data_modules.semantic_image_segmentation_datamodule import SemanticImageSegmentationDataModule
from utils.aki_labels import get_aki_label_names


logging.basicConfig(level=logging.INFO, format="[%(filename)s:%(lineno)s - %(funcName)s] %(message)s", datefmt="[%X]")

log = logging.getLogger(__name__)

def evaluate():
    # DEFINE PARAMETERS
    scenario="all"
    order_by="weather"
    datasets=["waymo"]
    classes=get_aki_label_names()
    void_classes=["void", "static"]
    num_workers = 4
    batch_size = 4
    val_limit= 32  

    checkpoint = "checkpoints/semantic_image_segmentation/sis_final_model.ckpt"
    sis_model = SemanticImageSegmentationModel.load_from_checkpoint(checkpoint)
    data_module = SemanticImageSegmentationDataModule(scenario=scenario,
                                                      order_by=order_by,
                                                      datasets=datasets,
                                                      classes=classes,
                                                      val_limit=val_limit,
                                                      void=void_classes,
                                                      num_workers=num_workers,
                                                      batch_size=batch_size)
                                
    data_module.setup()
    evaluate_semantic_image_segmentation(sis_model, data_module)


if __name__ == "__main__":
    evaluate()