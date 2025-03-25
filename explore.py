import logging 

from rich.logging import RichHandler
from exploration.explore_data import explore_aki_dataset, \
                                     explore_sis_dm_train_dataloader, \
                                     explore_weather_classification_train_dataloader, \
                                     explore_aki_ds_for_weather_classification, \
                                     explore_waymo_rain_images, \
                                     waymo_sunny_vs_rainy_images, \
                                     calc_waymo_rainy_vs_sunny_image_stats, \
                                     explore_unified_datamodule, \
                                     get_waymo_img_label_distribution
from akiset import AKIDataset
from utils.aki_labels import get_aki_label_names, get_aki_label_colors_rgb
from data_modules.semantic_image_segmentation_datamodule import SemanticImageSegmentationDataModule
from data_modules.weather_classification_datamodule import WeatherClassificationDataModule


logging.basicConfig(level="INFO", format="[%(levelname)s] - %(filename)s:%(lineno)s - %(message)s", datefmt="[%X]")

log = logging.getLogger("rich")

def main():
    log.info(f"Exploration script started")
    # 1) Create image of aki label colors, names and ids
    
    #2) Explore AkiDataset
    #log.info(f"Exporting images with their semantic segmentation masks from the AKIDataset class")
    #aki_ds = AKIDataset({"camera": ["image"], "camera_segmentation": ["camera_segmentation"]},
    #                    scenario="all",
    #                    datasets=["waymo"])
    #explore_aki_dataset(aki_ds, colors=get_aki_label_colors_rgb())

    # 3) Explore train loader of SemanticImageSegmentationDataModule
    #log.info(f"Exporting images with their semantic segmentation masks from the train data loader of the SemanticImageSegmentationDataModule")
    #sis_dm = SemanticImageSegmentationDataModule(scenario="all",
    #                                        datasets=["waymo"],
    #                                        classes=get_aki_label_names(),
    #                                        void=["void", "static"])
    #sis_dm.setup()
    #explore_sis_dm_train_dataloader(sis_dm, colors=get_aki_label_colors_rgb())

    #4) Explore train loader of WeatherClassificationDataModule

    #wc_dm = WeatherClassificationDataModule(scenario="all",
    #                                        batch_size=128,
    #                                        datasets=["waymo"],
    #                                        order_by="weather",
    #                                        num_workers=1,
    #                                        limit=10_000,
    #                                        shuffle=True)
    #wc_dm.setup()
    #explore_weather_classification_train_dataloader(wc_dm)
    #explore_aki_ds_for_weather_classification()
    
    #5) Show waymo rain images
    #explore_waymo_rain_images()

    #6) Compare sunny vs rainy images of the waymo dataset
    #waymo_sunny_vs_rainy_images()

    #7) Calc stats of sunny vs rainy images of the waymo dataset
    #calc_waymo_rainy_vs_sunny_image_stats()

    #8) Explore point cloud img projection datamodule
    #explore_unified_datamodule()

    #9) Get waymo img label distribution
    get_waymo_img_label_distribution()




if __name__ == "__main__":
    main()