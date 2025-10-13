import ast
import logging

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast

from src.utils.config import (SEPARATOR_SENTENCES, SENTIMENT_ELEMENT_PARTS, NULL_ASPECT_TERM, NULL_SPECIAL_TOKEN,
                              SENTIMENT_MAPPING)
from src.utils.tasks import Task


def data_collate_fn(batch: list[dict], tokenizer: PreTrainedTokenizerFast) -> dict:
    texts = [item["texts"] for item in batch]
    labels = [item["labels"] for item in batch]

    encoded_inputs = tokenizer(
        texts,
        truncation=True,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )

    encoded_labels = tokenizer(
        labels,
        truncation=True,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )

    encoded_labels["input_ids"][encoded_labels["input_ids"] == tokenizer.pad_token_id] = -100

    return {
        "input_text_ids": encoded_inputs["input_ids"],
        "input_attention_mask": encoded_inputs["attention_mask"],
        "labels_ids": encoded_labels["input_ids"],
        "labels_attention_mask": encoded_labels["attention_mask"],
        "labels": labels,
    }


class Seq2SeqDataset(Dataset):
    """Dataset for ABSA and Seq2Seq models."""

    def __init__(
            self,
            data_path: str,
            task: Task,
            few_shot: int = 0,
    ) -> None:
        """
        Initialize dataset for Seq2Seq ABSA with given arguments.

        :param data_path: path to the data file
        :param task: task
        :param few_shot: number of samples to use for few shot learning, 0 means all samples
        """

        self._data_path = data_path
        self._few_shot = few_shot
        self._task = task

        self._texts = []
        self._labels = []

        self._load_data()

    def __len__(self) -> int:
        """
        Return the length of the dataset.

        :return: length of the dataset
        """
        return len(self._labels)

    def __getitem__(self, index: int) -> dict:
        """
        Return the dictionary for item at the given index.
        Dictionary contains the following keys:
        - texts: input text
        - labels: label text

        :param index: index of the item
        :return: dictionary containing input text and label for item at the given index
        """

        return {
            "texts": self._texts[index],
            "labels": self._labels[index],
        }

    def _load_data(self) -> None:
        """
        Load data from the dataset file. Convert text and label into token IDs and attention masks.

        :return: None
        """
        with open(self._data_path, "r", encoding="utf-8") as file:
            lines = file.readlines()

        for line in lines:
            text, label = line.split("####")
            text = text.lower().strip()
            labels = ast.literal_eval(label.strip())

            input_labels = []
            for label in labels:
                aspect_term = label[0]
                if aspect_term == NULL_ASPECT_TERM:
                    aspect_term = NULL_SPECIAL_TOKEN
                else:
                    aspect_term = aspect_term.lower()

                if self._task == Task.ASTE:
                    sentiment = SENTIMENT_MAPPING[label[2]]
                    opinion_term = label[1].lower()
                    input_labels.append(
                        f"{SENTIMENT_ELEMENT_PARTS['AT']} {aspect_term} {SENTIMENT_ELEMENT_PARTS['OT']} {opinion_term} {SENTIMENT_ELEMENT_PARTS['SP']} {sentiment}"
                    )
                    continue

                if self._task == Task.ACOS and label[3] == NULL_ASPECT_TERM:
                    opinion_term = NULL_SPECIAL_TOKEN
                else:
                    opinion_term = label[3].lower()
                sentiment = SENTIMENT_MAPPING[label[2]]

                input_labels.append(
                    f"{SENTIMENT_ELEMENT_PARTS['AT']} {aspect_term} {SENTIMENT_ELEMENT_PARTS['OT']} {opinion_term} {SENTIMENT_ELEMENT_PARTS['AC']} {label[1]} {SENTIMENT_ELEMENT_PARTS['SP']} {sentiment}"
                )

            self._texts.append(text)
            labels_joined = f" {SEPARATOR_SENTENCES} ".join(input_labels)
            self._labels.append(labels_joined)

            if 0 < self._few_shot <= len(self._labels):
                break

        logging.info("Example of first sentence: %s", str(self._texts[0]))

        logging.info("Example of first label: %s", str(self._labels[0]))
        logging.info("Number of samples: %d", len(self._labels))
