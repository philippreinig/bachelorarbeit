import logging 

from rich.logging import RichHandler
from exploration.explore_dataset import explore_aki_dataset, explore_sis_dm_train_dataloader
from akiset import AKIDataset
from aki_labels import get_aki_label_names, get_aki_label_colors_rgb
from data_modules.semantic_image_segmentation_datamodule import SemanticImageSegmentationDataModule


logging.basicConfig(level="INFO", format="[%(filename)s:%(lineno)s - %(funcName)s] %(message)s", datefmt="[%X]", handlers=[RichHandler()])

log = logging.getLogger("rich")

def main():
    log.info(f"Exploration script started")
    # 1) Create image of aki label colors, names and ids
    
    #2) Explore AkiDataset
    log.info(f"Exporting images with their semantic segmentation masks from the AKIDataset class")
    aki_ds = AKIDataset({"camera": ["image"], "camera_segmentation": ["camera_segmentation"]},
                        scenario="all",
                        datasets=["all"])
    explore_aki_dataset(aki_ds, colors=get_aki_label_colors_rgb())

    #3) Explore train loader of SemanticImageSegmentationDatamModule
    log.info(f"Exporting images with their semantic segmentation masks from the train data loader of the SemanticImageSegmentationDataModule")
    sis_dm = SemanticImageSegmentationDataModule(scenario="all",
                                            datasets=["waymo"],
                                            classes=get_aki_label_names(),
                                            void=["void", "static"])
    sis_dm.setup()
    explore_sis_dm_train_dataloader(sis_dm, colors=get_aki_label_colors_rgb())

if __name__ == "__main__":
    main()