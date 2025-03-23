import logging

import lightning as L

from lightning.pytorch.loggers import WandbLogger
from evaluation.evaluate_sis import evaluate_semantic_image_segmentation
from models.semantic_image_segmentation import SemanticImageSegmentationModel
from models.semantic_lidar_segmentation import PointNet2
from data_modules.semantic_image_segmentation_datamodule import SemanticImageSegmentationDataModule
from data_modules.unified_datamodule import UnifiedDataModule
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

def evaluate_with_udm():
    # DEFINE PARAMETERS
    # Dataset
    scenario="rain"
    order_by="weather"
    datasets=["waymo"]
    classes=get_aki_label_names()
    void_classes=[]
    num_workers = 16
    batch_size = 32
    train_limit = 100_000
    val_limit= 100_000
    test_limit = 100_000

    # Model
    data_from_udm=True

    # Training
    precision="16-mixed"
    accelerator="gpu"

    wandb_logger = WandbLogger(name=f"eval_lig_sem_seg_with_on_udm_with_down_model",
                               log_model="all",
                               save_dir="logs/wandb/eval/",
                               project="eval")

    checkpoint = "checkpoints/final_models/lid_sem_seg_down_final-1351811.ckpt"
    model = PointNet2.load_from_checkpoint(checkpoint,
                                           num_classes=len(classes),
                                           data_from_udm=data_from_udm)
    udm = UnifiedDataModule(scenario=scenario,
                            order_by=order_by,
                            datasets=datasets,
                            classes=classes,
                            train_limit=train_limit,
                            val_limit=val_limit,
                            test_limit=test_limit,
                            void=void_classes,
                            num_workers=num_workers,
                            batch_size=batch_size)
    


     # Create trainer and start training
    trainer = L.Trainer(logger=wandb_logger,
                        precision=precision,
                        enable_progress_bar=False,
                        accelerator=accelerator)

    trainer.validate(model, udm)
    trainer.test(model, udm)

    #udm.setup()
    #for batch in udm.val_dataloader():
    #    for elem in batch:
    #        img, seg_mask, point_cloud, pc_labels, weather_condition, fusable_pixel_mask, points_pixel_projection = elem




if __name__ == "__main__":
    evaluate_with_udm()