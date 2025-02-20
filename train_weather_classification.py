import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from models.weather_classifier import WeatherClassifier
from data_modules.weather_classification_datamodule import WeatherClassificationDataModule

import logging

from utils.misc import MyPrintingCallback

logging.basicConfig(level="INFO",  format="[%(levelname)s] - %(filename)s:%(lineno)s - %(message)s", datefmt="[%X]")
log = logging.getLogger("my_logger")

torch.set_float32_matmul_precision("high")

def train_weather_classifier():
    # Define params
    scenario = "all"
    datasets = ["waymo"]
    db_connection = "psycopg@ants" # or "psycopg@aki"
    order_by = "weather"
    limit = 10000
    batch_size = 32
    num_workers = 1
    shuffle=True

    max_epochs = 100

    wandb_logger = WandbLogger(log_model="all",
                               save_dir="logs/wandb/weather_classification/",
                               project="weather_classification")


    # Create model
    log.info(f"Constructing data module, models, etc.")
    weather_classification_model = WeatherClassifier()

    
    # Create data module
    data_module = WeatherClassificationDataModule(dbtype=db_connection,
                                                  datasets=datasets,
                                                  batch_size=batch_size,
                                                  num_workers=num_workers,
                                                  order_by=order_by,
                                                  shuffle=shuffle,
                                                  limit=limit)
    data_module.setup()
    
    train_ds_size = data_module.train_ds.count
    dm_batch_size = data_module.batch_size
    
    amount_of_batches = train_ds_size // dm_batch_size if train_ds_size % dm_batch_size == 0 else train_ds_size // dm_batch_size + 1
    log.info(f"Amount of batches: {amount_of_batches}")


    # Create logger and callbacks
    
    checkpoint_callback = ModelCheckpoint(dirpath="checkpoints/weather_classification/",
                                          filename='{epoch:02d}-{val_loss:.5f}',
                                          monitor='val_loss',
                                          save_top_k=-1,  # Save all checkpoints
                                          every_n_epochs=1)


    # Run training
    trainer = pl.Trainer(max_epochs=max_epochs,
                         callbacks=[MyPrintingCallback(), checkpoint_callback],
                         logger=wandb_logger,
                         enable_progress_bar=False)

    trainer.fit(weather_classification_model, datamodule=data_module)

    trainer.validate(model=weather_classification_model, datamodule=data_module)

if __name__ == "__main__":
    train_weather_classifier()