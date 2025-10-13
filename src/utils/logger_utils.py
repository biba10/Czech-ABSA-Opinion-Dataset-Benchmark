import argparse
import logging
import os

import wandb
from pytorch_lightning.loggers import Logger, WandbLogger, CSVLogger

from src.utils.config import WANDB_ENTITY, WANDB_PROJECT_NAME


def get_trainer_logger(args: argparse.Namespace, wandb_entity=WANDB_ENTITY, wandb_project=WANDB_PROJECT_NAME) -> Logger:
    """
    Initialize logger for Trainer – WandB if api key is available, CSV otherwise.

    :param wandb_project:
    :param wandb_entity: wandb entity for logging
    :param args: parsed arguments used for WandB logger config
    :return: logger (WandB or CSV)
    """
    if args.no_wandb:
        logging.info("Using CSV logger")
        return CSVLogger(save_dir=os.getcwd(), name="logs")

    logging.info("Trying to use wandb logger")
    try:
        logger = WandbLogger(
            project=wandb_project,
            entity=wandb_entity,
            config=vars(args),
            tags=[args.tag] if args.tag else [],
        )
    except Exception:
        # Log the exception
        logging.error("Wandb exception: ", exc_info=True)
        logging.info("WandB login failed, using CSV logger")
        logger = CSVLogger(save_dir=os.getcwd(), name="logs")

    return logger


def init_logging() -> None:
    """Initialize logging."""
    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
