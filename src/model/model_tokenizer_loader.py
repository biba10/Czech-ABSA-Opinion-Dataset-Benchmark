import pytorch_lightning as pl
import torch
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM, PreTrainedTokenizerFast, BitsAndBytesConfig,
                          AutoModelForCausalLM, Gemma3ForConditionalGeneration)

from src.model.model_generative import ABSAModelGenerative
from src.utils.config import SENTIMENT_ELEMENT_PARTS, SEPARATOR_SENTENCES, NULL_SPECIAL_TOKEN
from src.utils.tasks import Task


def load_model_and_tokenizer(
        model_path: str,
        max_seq_length_label: int,
        optimizer: str,
        learning_rate: float,
        beam_size: int,
        task: Task,
        constrained_decoding: bool,
) -> tuple[pl.LightningModule, PreTrainedTokenizerFast]:
    """
    Load model and tokenizer from path. Add special tokens to tokenizer.

    :param model_path: path to pre-trained model or shortcut name
    :param max_seq_length_label: maximal length of the label to generate
    :param optimizer: optimizer
    :param learning_rate: learning rate
    :param beam_size: beam size
    :param task: task
    :param constrained_decoding: if True, constrained decoding is used
    :return: model and tokenizer
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, model_max_length=512)

    additional_tokens = list(SENTIMENT_ELEMENT_PARTS.values()) + [SEPARATOR_SENTENCES]

    if task == Task.ASTE:
        additional_tokens.remove(SENTIMENT_ELEMENT_PARTS["AC"])
    else:
        additional_tokens.append(NULL_SPECIAL_TOKEN)

    tokenizer.add_tokens(additional_tokens)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    model.resize_token_embeddings(len(tokenizer))

    absa_model = ABSAModelGenerative(
        learning_rate=learning_rate,
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        beam_size=beam_size,
        constrained_decoding=constrained_decoding,
        max_length=max_seq_length_label,
        task=task,
    )

    return absa_model, tokenizer


def llm_model_tokenizer_loader(model_name, token):
    use_cpu = True if not torch.cuda.is_available() else False
    device = torch.device("cpu" if use_cpu else "cuda" if torch.cuda.is_available() else "cpu")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # load model in 4-bit precision
        bnb_4bit_quant_type="nf4",  # pre-trained model should be quantized in 4-bit NF format
        bnb_4bit_use_double_quant=True,  # Using double quantization as mentioned in QLoRA paper
        bnb_4bit_compute_dtype=torch.bfloat16,  # During computation, pre-trained model should be loaded in BF16 format
    )

    if "gemma-3" in model_name.lower():
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=bnb_config if not use_cpu else None,
            device_map="auto" if not use_cpu else "cpu",
            low_cpu_mem_usage=True,
            token=token,
            attn_implementation="eager",
            torch_dtype=torch.bfloat16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config if not use_cpu else None,
            device_map="auto" if not use_cpu else "cpu",
            use_cache=False,
            low_cpu_mem_usage=True,
            token=token,
        )

    use_fast = True
    if "orca" in model_name.lower():
        use_fast = False

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token, use_fast=use_fast)
    return model, tokenizer, use_cpu, device, bnb_config
