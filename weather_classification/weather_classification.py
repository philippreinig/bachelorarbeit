import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from weather_classifier import WeatherClassifier
from weather_classification_datamodule import WeatherClassificationDataModule

import logging
from rich.logging import RichHandler

logging.basicConfig(level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()])
log = logging.getLogger("rich")

def train_weather_classifier():
    weather_classification_model = WeatherClassifier()

    data_module = WeatherClassificationDataModule(dbtype="psycopg@aki")

    trainer = Trainer()
    log.info(f"Starting training")
    trainer.fit(weather_classification_model, datamodule=data_module)
    trainer.validate(datamodule=data_module)
    trainer.test(datamodule=data_module)

if __name__ == "__main__":
    train_weather_classifier()