from functools import partial

import pytorch_lightning as pl
import torch
from torch.optim import Optimizer
from transformers import Adafactor, PreTrainedModel, PreTrainedTokenizerFast
from transformers.modeling_outputs import Seq2SeqLMOutput

from src.evaluation.evaluation import process_batch_for_evaluation
from src.evaluation.f1_score import F1ScoreSeq2Seq
from src.utils.config import (SENTIMENT_MAPPING, CATEGORIES, SENTIMENT_ELEMENT_PARTS, NULL_SPECIAL_TOKEN,
                              SEPARATOR_SENTENCES)
from src.utils.tasks import Task


class ABSAModelGenerative(pl.LightningModule):
    """Generative model for Aspect Based Sentiment Analysis."""

    def __init__(
            self,
            learning_rate: float,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizerFast,
            optimizer: str,
            beam_size: int,
            constrained_decoding: bool,
            max_length: int,
            task: Task,
    ) -> None:
        """
        Initialize the model.

        :param learning_rate: learning rate
        :param model: pre-trained model (expecting a Seq2Seq model)
        :param tokenizer: pre-trained tokenizer
        :param optimizer: optimizer
        :param beam_size: beam size
        :param constrained_decoding: whether to use constrained decoding
        :param max_length: maximum sequence length for generation
        :param task: task
        """
        super().__init__()

        self._learning_rate = learning_rate
        self._model = model
        self._tokenizer = tokenizer
        self._optimizer = optimizer
        self._beam_size = beam_size
        self._max_length = max_length
        self._constrained_decoding = constrained_decoding
        self._task = task

        self._f1_score = F1ScoreSeq2Seq()

        if self._constrained_decoding:
            self._force_tokens = create_force_tokens(self._tokenizer)

        self.save_hyperparameters(ignore=["model"])

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            decoder_attention_mask: torch.Tensor,
            labels: torch.Tensor,
    ) -> Seq2SeqLMOutput:
        """
        Perform forward pass through the model.

        :param input_ids: input ids
        :param attention_mask: attention mask
        :param decoder_attention_mask: decoder attention mask
        :param labels: labels
        :return: model output
        """
        output = self._model(
            input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )
        return output

    def _compute_loss(self, batch: dict) -> torch.Tensor:
        """
        Compute loss for a batch.

        :param batch: batch
        :return: loss
        """
        out = self(
            input_ids=batch["input_text_ids"],
            attention_mask=batch["input_attention_mask"],
            labels=batch["labels_ids"],
            decoder_attention_mask=batch["labels_attention_mask"],
        )
        loss = out.loss
        return loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """
        Perform training step for a single batch. Compute loss and log it.

        :param batch: batch
        :param batch_idx: batch index
        :return: loss
        """
        loss = self._compute_loss(batch)

        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def _generate_output_and_update_metrics(self, batch: dict) -> torch.Tensor:
        """
        Generate output and predictions, calculate loss, and update metrics for a single batch.

        :param batch: batch
        :return: loss
        """
        loss = self._compute_loss(batch)

        generated_ids = self._model.generate(
            input_ids=batch["input_text_ids"],
            attention_mask=batch["input_attention_mask"],
            max_length=self._max_length,
            num_beams=self._beam_size,
            prefix_allowed_tokens_fn=partial(
                prefix_allowed_tokens_fn,
                self._tokenizer,
                self._force_tokens,
                batch["input_text_ids"],
                self._task,
            ) if self._constrained_decoding else None,
        )

        decoded_predictions = self._tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        gold_labels = batch["labels"]

        process_batch_for_evaluation(
            decoded_predictions=decoded_predictions,
            labels=gold_labels,
            f1_score=self._f1_score,
        )

        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        """
        Perform validation step for a single batch. Generate output and predictions, calculate loss, and update metrics.
        Log loss and metrics.

        :param batch: batch
        :param batch_idx: batch index
        :return: validation loss
        """
        loss = self._generate_output_and_update_metrics(batch)

        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("f1", self._f1_score, prog_bar=True, sync_dist=True)
        return {"val_loss": loss}

    def test_step(self, batch: dict, batch_idx: int) -> dict:
        """
        Perform test step for a single batch. Generate output and predictions, calculate loss, and update metrics.
        Log loss and metrics.

        :param batch: batch
        :param batch_idx: batch index
        :return: test loss
        """
        loss = self._generate_output_and_update_metrics(batch)

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_f1", self._f1_score, prog_bar=True)
        return {"test_loss": loss}

    def configure_optimizers(self) -> Optimizer:
        """
        Configure optimizer.

        :return: optimizer
        """
        model = self._model
        if self._optimizer == "AdamW":
            optimizer = torch.optim.AdamW(model.parameters(), lr=self._learning_rate)
        elif self._optimizer == "adafactor":
            optimizer = Adafactor(
                model.parameters(),
                lr=self._learning_rate,
                scale_parameter=False,
                relative_step=False,
            )
        else:
            raise ValueError(f"Optimizer {self._optimizer} not implemented.")
        return optimizer


def create_force_tokens(tokenizer: PreTrainedTokenizerFast) -> dict[str, list[int] | dict[str, int]]:
    force_tokens = {}

    category_tokens = []
    for category in CATEGORIES:
        category_tokens.extend(tokenizer.encode(category, add_special_tokens=False))
    force_tokens["category_tokens"] = category_tokens

    sentiment_tokens = []
    for sentiment in SENTIMENT_MAPPING.values():
        sentiment_tokens.extend(tokenizer.encode(sentiment, add_special_tokens=False))
    force_tokens["sentiment_tokens"] = sentiment_tokens

    special_tokens = {}
    for key, value in SENTIMENT_ELEMENT_PARTS.items():
        special_tokens[key] = tokenizer.encode(value, add_special_tokens=False)[-1]
    special_tokens["null"] = tokenizer.encode(NULL_SPECIAL_TOKEN, add_special_tokens=False)[-1]
    special_tokens["separator"] = tokenizer.encode(SEPARATOR_SENTENCES, add_special_tokens=False)[-1]
    force_tokens["special_tokens"] = special_tokens

    return force_tokens


def prefix_allowed_tokens_fn(
        tokenizer: PreTrainedTokenizerFast,
        force_tokens: dict[str, list[int] | dict[str, int]],
        source_ids: torch.Tensor,
        task: Task,
        batch_id: int,
        input_ids: torch.Tensor,
) -> list[int]:
    # if empty return <aspect>
    if len(input_ids) == 0:
        return [force_tokens["special_tokens"]["AT"]]

    aspect_index = (input_ids == force_tokens["special_tokens"]["AT"]).nonzero()

    # if there is no aspect token, return <aspect>
    if len(aspect_index) == 0:
        return [force_tokens["special_tokens"]["AT"]]

    opinion_index = (input_ids == force_tokens["special_tokens"]["OT"]).nonzero()
    category_index = (input_ids == force_tokens["special_tokens"]["AC"]).nonzero()
    polarity_index = (input_ids == force_tokens["special_tokens"]["SP"]).nonzero()
    separator_index = (input_ids == force_tokens["special_tokens"]["separator"]).nonzero()

    aspect_last_index = aspect_index[-1] if len(aspect_index) > 0 else -1
    opinion_last_index = opinion_index[-1] if len(opinion_index) > 0 else -1
    category_last_index = category_index[-1] if len(category_index) > 0 else -1
    polarity_last_index = polarity_index[-1] if len(polarity_index) > 0 else -1
    separator_last_index = separator_index[-1] if len(separator_index) > 0 else -1

    last_special_token_index = max(
        (aspect_last_index, "AT"),
        (opinion_last_index, "OT"),
        (category_last_index, "AC"),
        (polarity_last_index, "SP"),
        (separator_last_index, "separator"),
        key=lambda x: x[0],
    )

    # if last index is separator, return <aspect>
    if last_special_token_index[1] == "separator":
        return [force_tokens["special_tokens"]["AT"]]

    if last_special_token_index[1] == "AT":
        return_set = set(source_ids[batch_id].tolist())
        if task != Task.ASTE:
            return_set.add(force_tokens["special_tokens"]["null"])
        # if aspect term is not the last token, add also <opinion>
        if aspect_index[-1] != len(input_ids) - 1:
            return_set.add(force_tokens["special_tokens"]["OT"])
        return_set.discard(tokenizer.eos_token_id)
        return list(return_set)

    if last_special_token_index[1] == "OT":
        return_set = set(source_ids[batch_id].tolist())
        if task == Task.ACOS:
            return_set.add(force_tokens["special_tokens"]["null"])
        if opinion_index[-1] != len(input_ids) - 1:
            if task != Task.ASTE:
                return_set.add(force_tokens["special_tokens"]["AC"])
            else:
                return_set.add(force_tokens["special_tokens"]["SP"])
        return_set.discard(tokenizer.eos_token_id)
        return list(return_set)

    if task != Task.ASTE and last_special_token_index[1] == "AC":
        return_set = set(force_tokens["category_tokens"])
        if category_index[-1] != len(input_ids) - 1:
            return_set.add(force_tokens["special_tokens"]["SP"])
        # remove eos token if in the list
        return_set.discard(tokenizer.eos_token_id)
        return list(return_set)

    if last_special_token_index[1] == "SP":
        return_set = set(force_tokens["sentiment_tokens"])
        if polarity_index[-1] != len(input_ids) - 1:
            return_set.add(force_tokens["special_tokens"]["separator"])
            # add also eos token
            return_set.add(tokenizer.eos_token_id)
        else:
            return_set.discard(tokenizer.eos_token_id)
        return list(return_set)
