import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from models.weather_classifier import WeatherClassifier
from data_modules.weather_classification_datamodule import WeatherClassificationDataModule

import logging
from rich.logging import RichHandler

from utils import MyPrintingCallback

logging.basicConfig(level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()])
log = logging.getLogger("rich")

torch.set_float32_matmul_precision("high")

def train_weather_classifier():
    weather_classification_model = WeatherClassifier()

    db_connection = "psycopg@ants" # or "psycopg@aki"
    data_module = WeatherClassificationDataModule(dbtype=db_connection)
    data_module.setup()
    
    train_ds_size = data_module.train_ds.count
    dm_batch_size = data_module.batch_size
    
    amount_of_batches = train_ds_size // dm_batch_size if train_ds_size % dm_batch_size == 0 else train_ds_size // dm_batch_size + 1
    log.info(f"Amount of batches: {amount_of_batches}")

    wandb_logger = WandbLogger(log_model="all", save_dir="logs/wandb/weather_classification/resnet18/")

    checkpoint_callback = ModelCheckpoint(dirpath="checkpoints/weather_classification/",
                                          filename='epoch={epoch:02d}-val_loss={val_loss:.2f}',
                                          monitor='val_loss',
                                          save_top_k=-1,  # Save all checkpoints
                                          every_n_epochs=1)

    trainer = pl.Trainer(max_epochs=100,
                         callbacks=[pl.callbacks.progress.RichProgressBar(), MyPrintingCallback(), checkpoint_callback],
                         logger=wandb_logger)

    log.info(f"Starting training")
    trainer.fit(weather_classification_model, datamodule=data_module)
    log.info(f"Training done")

    trainer.validate(model=weather_classification_model, datamodule=data_module)

if __name__ == "__main__":
    train_weather_classifier()