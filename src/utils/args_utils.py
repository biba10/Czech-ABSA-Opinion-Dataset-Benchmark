import argparse
import logging

from src.utils.config import (TEST_LANG_OPTIONS, TRAIN_LANG_OPTIONS, MODE_OPTIONS, ADAFACTOR_OPTIMIZER, ADAMW_OPTIMIZER,
                              MODE_DEV, LANG_CZECH)
from src.utils.tasks import Task


def init_args() -> argparse.Namespace:
    """
    Initialize arguments for the script.

    :return: parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="t5-base",
        help="Path to pre-trained model or shortcut name.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size."
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument(
        "--target_language",
        type=str,
        default=LANG_CZECH,
        help="Language of test dataset (target language).",
        choices=TEST_LANG_OPTIONS,
    )
    parser.add_argument(
        "--source_language",
        type=str,
        default=LANG_CZECH,
        help="Language of training dataset (source language).",
        choices=TRAIN_LANG_OPTIONS,
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        choices=[ADAMW_OPTIMIZER, ADAFACTOR_OPTIMIZER],
        default=ADAMW_OPTIMIZER,
        help="Optimizer.",
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=MODE_OPTIONS,
        default=MODE_DEV,
        help="Mode - 'dev' uses validation data for selecting the best model, 'test' uses the exact number of epochs.",

    )
    parser.add_argument(
        "--checkpoint_monitor",
        type=str,
        choices=["val_loss", "f1"],
        default="f1",
        help="Metric based on which the best model will be stored according to the performance on validation data in "
             "'dev' mode",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="Accumulate gradient batches. It is used when there is insufficient memory for training"
             " for the required effective batch size.",
    )
    parser.add_argument("--beam_size", type=int, default=1, help="Beam size for beam search decoding.")
    parser.add_argument("--task", type=Task, choices=list(Task), default=Task.ACOS, help="Task.")

    parser.add_argument(
        "--max_data",
        type=int,
        default=0,
        help="Number of examples for training for target language. 0 means all examples.",
    )

    parser.add_argument(
        "--target_language_few_shot",
        type=int,
        default=None,
        help="Number of examples for training for target language. None means no examples, 0 means all examples.",
    )
    parser.add_argument("--no_wandb", action="store_true", help="Do not use WandB.")
    parser.add_argument("--tag", type=str, default="opinion", help="Tag for WandB.")
    parser.add_argument(
        "--constrained_decoding",
        action="store_true",
        help="Use constrained decoding. It has an effect only when used with sequence-to-sequence models.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Token for the model. It is used for loading the model and tokenizer.",
    )
    parser.add_argument(
        "--few_shot_prompt",
        action="store_true",
        help="Use few-shot prompt for training.",
    )

    parser.add_argument(
        "--instruction_tuning",
        action="store_true",
        help="Use instruction tuning for training.",
    )

    parser.add_argument(
        "--train_translated",
        action="store_true",
        help="Train on translated data.",
    )

    parser.add_argument(
        "--lora_r",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
    )

    args = parser.parse_args()

    logging.info("Arguments: %s", args)

    logging.info("Train language: %s", args.source_language)
    logging.info("Test language: %s", args.target_language)

    return args
