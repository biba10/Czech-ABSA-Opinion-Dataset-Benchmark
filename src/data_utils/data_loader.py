import logging
import os
from functools import partial

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast

from src.data_utils.dataset_seq2seq import Seq2SeqDataset, data_collate_fn
from src.utils.config import ABSA_TRAIN, ABSA_TEST, DATA_DIR_PATH, ABSA_DEV, LANG_ENGLISH, TRANSLATED_DATA_PATH
from src.utils.tasks import Task


class SADataLoader(pl.LightningDataModule):
    """Data loader for sentiment analysis."""

    def __init__(
            self,
            source_language: str,
            target_language: str,
            batch_size: int,
            tokenizer: PreTrainedTokenizerFast,
            task: Task,
            max_data: int,
            target_language_few_shot: int | None,
            train_translated: bool = False,
    ) -> None:
        """
        Initialize data loader for ABSA dataset with given arguments.

        :param source_language: language of the train dataset
        :param target_language: language of the test dataset
        :param batch_size: train and validation batch size
        :param tokenizer: tokenizer
        :param task: task to solve
        :param max_data: number of examples for few-shot learning
        :param target_language_few_shot: number of examples for few-shot learning in target language
        :param train_translated: if True, train data will be translated
        """
        super().__init__()
        self._train_dataset = None
        self._dev_dataset = None
        self._test_dataset = None
        self._tokenizer = tokenizer
        self._batch_size = batch_size
        self._source_language = source_language
        self._target_language = target_language
        self._task = task
        self._max_data = max_data
        self._target_language_few_shot = target_language_few_shot
        self._train_translated = train_translated

    def setup(self, stage=None) -> None:
        """
        Setup data loader.

        :param stage: stage ('fit' for training or 'test' for testing)
        :return: None
        """
        if stage == "fit" or stage is None:
            # Load train dataset
            data_path_train = os.path.join(DATA_DIR_PATH, self._source_language, str(self._task), ABSA_TRAIN)
            data_path_dev = os.path.join(DATA_DIR_PATH, self._source_language, str(self._task), ABSA_DEV)
            train_dataset = Seq2SeqDataset(
                data_path=data_path_train,
                task=self._task,
                few_shot=self._max_data,
            )
            dev_dataset = Seq2SeqDataset(
                data_path=data_path_dev,
                task=self._task,
                few_shot=self._max_data,
            )

            if self._source_language == LANG_ENGLISH and self._train_translated:
                translated_data_path = os.path.join(DATA_DIR_PATH, TRANSLATED_DATA_PATH, str(self._task), ABSA_TRAIN)
                translated_dataset = Seq2SeqDataset(
                    data_path=translated_data_path,
                    task=self._task,
                    few_shot=self._max_data,
                )

                self._train_dataset = torch.utils.data.ConcatDataset(
                    [
                        train_dataset,
                        translated_dataset,
                    ]
                )

            if self._target_language_few_shot is not None:
                data_path_train = os.path.join(DATA_DIR_PATH, self._target_language, str(self._task), ABSA_TRAIN)
                train_dataset_target = Seq2SeqDataset(
                    data_path=data_path_train,
                    task=self._task,
                    few_shot=self._target_language_few_shot,
                )

                self._train_dataset = torch.utils.data.ConcatDataset(
                    [
                        train_dataset,
                        train_dataset_target,
                    ]
                )

                data_path_dev = os.path.join(DATA_DIR_PATH, self._target_language, str(self._task), ABSA_DEV)
                dev_dataset_target = Seq2SeqDataset(
                    data_path=data_path_dev,
                    task=self._task,
                    few_shot=self._max_data,
                )

                self._dev_dataset = torch.utils.data.ConcatDataset(
                    [
                        dev_dataset,
                        dev_dataset_target,
                    ]
                )
            else:
                self._train_dataset = train_dataset
                self._dev_dataset = dev_dataset

            logging.info("Train data all length: %d", len(self._train_dataset))

        # Load test dataset
        if stage == "test" or stage is None:
            data_path_test = os.path.join(DATA_DIR_PATH, self._target_language, str(self._task), ABSA_TEST)
            self._test_dataset = Seq2SeqDataset(
                data_path=data_path_test,
                task=self._task,
                few_shot=self._max_data,
            )
            logging.info("Test data length: %d", len(self._test_dataset))

    def train_dataloader(self) -> DataLoader:
        """
        Get train data loader.

        :return: train data loader
        """
        return DataLoader(
            self._train_dataset,
            batch_size=self._batch_size,
            num_workers=0,
            shuffle=True,
            collate_fn=partial(data_collate_fn, tokenizer=self._tokenizer),
        )

    def val_dataloader(self) -> DataLoader:
        """
        Get dev data loader.

        :return: dev data loader
        """
        return DataLoader(
            self._dev_dataset,
            batch_size=self._batch_size,
            num_workers=0,
            shuffle=False,
            collate_fn=partial(data_collate_fn, tokenizer=self._tokenizer),
        )

    def test_dataloader(self):
        """
        Get test data loader.

        :return: test data loader
        """
        return DataLoader(
            self._test_dataset,
            batch_size=self._batch_size,
            num_workers=0,
            shuffle=False,
            collate_fn=partial(data_collate_fn, tokenizer=self._tokenizer),
        )
