import logging
import re

import numpy as np

from src.evaluation.f1_score import F1ScoreSeq2Seq
from src.utils.config import NULL_ASPECT_TERM, SEPARATOR_SENTENCES, SENTIMENT_ELEMENT_PARTS, NULL_SPECIAL_TOKEN


def normalize_spaces(text):
    """
    Normalize spaces around punctuation, ensuring each punctuation is separated by spaces.
    """
    # Add spaces around punctuation
    text = re.sub(r"([.,!?;:/()\"'])", r" \1 ", text)
    # Remove extra spaces created by multiple punctuations
    text = re.sub(r"\s+", " ", text)
    # Trim leading and trailing spaces
    return text.strip()


def _parse_sentence(sentence: str) -> tuple[str, str, str, str]:
    """
    Parse sentence to retrieve sentiment elements.

    :param sentence: sentence to parse
    :return: aspect category, aspect term and sentiment
    """
    for part in SENTIMENT_ELEMENT_PARTS.values():
        if part not in sentence:
            sentence += f"{part} null"
    indexes = [sentence.index(part) for part in SENTIMENT_ELEMENT_PARTS.values()]
    arg_index_list = list(np.argsort(indexes))

    result = []
    for i in range(len(indexes)):
        start = indexes[i] + len(list(SENTIMENT_ELEMENT_PARTS.values())[i])
        sort_index = arg_index_list.index(i)
        if sort_index < len(indexes) - 1:
            next_ = arg_index_list[sort_index + 1]
            re = sentence[start:indexes[next_]]
        else:
            re = sentence[start:]
        result.append(re.strip())

    aspect_term, opinion_term, aspect_category, sentiment = result

    # normalize aspect term and opinion term
    aspect_term = normalize_spaces(aspect_term)
    opinion_term = normalize_spaces(opinion_term)

    return aspect_term, opinion_term, aspect_category, sentiment


def retrieve_slots(sample: str) -> set[tuple[str, str, str, str]]:
    """
    Retrieve slots from data sample.

    :param sample: data sample
    :return: slots from sample
    """
    # Get slots if sample is not empty
    try:
        result = set()
        if sample.strip():
            sentences = [sent.strip() for sent in sample.split(SEPARATOR_SENTENCES)]

            for sentence in sentences:
                try:
                    aspect_term, opinion_term, aspect_category, sentiment = _parse_sentence(sentence)

                except Exception as e:
                    logging.error("Exception: %s - %s", str(sentence), str(e))
                    aspect_category, opinion_term, aspect_term, sentiment = "", "", "", ""

                if aspect_term == NULL_SPECIAL_TOKEN:
                    aspect_term = NULL_ASPECT_TERM
                if opinion_term == NULL_SPECIAL_TOKEN:
                    opinion_term = NULL_ASPECT_TERM
                result.add((aspect_term, opinion_term, aspect_category, sentiment))
        return result
    except IndexError as e:
        logging.error("ValueError: %s", str(e))
        return set()


def process_batch_for_evaluation(
        decoded_predictions: list[str],
        labels: list[str],
        f1_score: F1ScoreSeq2Seq,
) -> None:
    """
    Process batch for evaluation. Convert predictions and labels to slots and update metrics.

    :param decoded_predictions: decoded predictions
    :param labels: labels
    :param f1_score: f1 score
    :return: None
    """
    for decoded_prediction, label in zip(decoded_predictions, labels):
        logging.info("Decoded prediction: %s", str(decoded_prediction))
        logging.info("Label: %s", str(label))
        slots_predictions = retrieve_slots(decoded_prediction)
        slots_labels = retrieve_slots(label)
        f1_score.update(predictions=slots_predictions, labels=slots_labels)
