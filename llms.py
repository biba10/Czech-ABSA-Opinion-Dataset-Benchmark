import argparse
import functools
import logging
import os

import torch
import wandb
from peft import AutoPeftModelForCausalLM, LoraConfig, PeftModel
from torch.utils.data import DataLoader
from transformers import Gemma3ForConditionalGeneration
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer, SFTConfig

from src.data_utils.data_utils import data_collate_llm_dataset
from src.data_utils.llm_dataset import LLMDataset
from src.llm_prompting.llm_classifier import llm_classify
from src.llm_prompting.templates import INSTRUCTIONS
from src.model.model_tokenizer_loader import llm_model_tokenizer_loader
from src.utils.args_utils import init_args
from src.utils.config import WANDB_ENTITY, ABSA_TRAIN, ABSA_DEV, ABSA_TEST, WANDB_PROJECT_NAME, TRANSLATED_DATA_PATH
from src.utils.logger_utils import init_logging

SUPPORTED_GPUS_TF32 = ["A100", "A6000", "RTX 30", "RTX 40", "A30", "A40"]


def check_ampere_gpu() -> bool:
    """Check if the GPU supports NVIDIA Ampere or later and enable FP32 in PyTorch if it does."""
    # Check if CUDA is available
    if not torch.cuda.is_available():
        logging.info("No GPU detected, running on CPU.")
        return False
    # Get current device named
    device_name = torch.cuda.get_device_name(torch.cuda.current_device())
    for supported_gpu in SUPPORTED_GPUS_TF32:
        if supported_gpu in device_name:
            logging.info("Detected ampere GPU: %s", device_name)
            return True
    logging.info("Detected non-ampere GPU: %s", device_name)
    return False

def find_target_modules(model) -> list[str]:
    # Initialize a Set to Store Unique Layers
    unique_layers = set()

    # Iterate Over All Named Modules in the Model
    for name, module in model.named_modules():
        # Check if the Module Type Contains 'Linear4bit'
        if "Linear4bit" in str(type(module)):
            # Extract the Type of the Layer
            layer_type = name.split('.')[-1]

            # Add the Layer Type to the Set of Unique Layers
            unique_layers.add(layer_type)

    # Return the Set of Unique Layers Converted to a List
    return list(unique_layers)


def instruction_tuning(args: argparse.Namespace):
    model, tokenizer, use_cpu, device, bnb_config = llm_model_tokenizer_loader(args.model, args.token)

    model.config.pretraining_tp = 1
    # tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    resized_embeddings = False
    if tokenizer.pad_token is None:
        if args.batch_size == 1:
            logging.info("Setting pad token to eos token...")
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        elif "Llama-3.1" in args.model:
            logging.info("Setting pad token to <|finetune_right_pad_id|>...")
            tokenizer.pad_token = "<|finetune_right_pad_id|>"
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        else:
            logging.info("Resizing embeddings...")
            tokenizer.pad_token = "[PAD]"
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            model.resize_token_embeddings(len(tokenizer))
            resized_embeddings = True

    peft_config = LoraConfig(
        lora_alpha=args.lora_alpha,
        lora_dropout=0.1,
        r=args.lora_r,
        target_modules=find_target_modules(model),
        modules_to_save=None if not resized_embeddings else ["lm_head", "embed_tokens"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    if tokenizer.chat_template is None:
        if "microsoft/Orca" in args.model:
            tokenizer.chat_template = "{{ bos_token }} {% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

    instruction = INSTRUCTIONS.get(str(args.task), None)
    if instruction is None:
        raise ValueError(f"Instruction not found for {args.task}")

    data_path_train = os.path.join("data", args.source_language, str(args.task), ABSA_TRAIN)

    train_dataset = LLMDataset(
        data_path=str(data_path_train),
        tokenizer=tokenizer,
        max_data=args.max_data,
        instruction_tuning=True,
        instruction=instruction,
        testing=False,
    )

    if args.train_translated:
        data_path_train_translated = os.path.join("data", TRANSLATED_DATA_PATH, str(args.task), ABSA_TRAIN)
        train_dataset_translated = LLMDataset(
            data_path=str(data_path_train_translated),
            tokenizer=tokenizer,
            max_data=args.max_data,
            instruction_tuning=True,
            instruction=instruction,
            testing=False,
        )
        train_dataset = torch.utils.data.ConcatDataset([train_dataset, train_dataset_translated])

    if args.target_language_few_shot is not None:
        data_path_train_target = os.path.join("data", args.target_language, str(args.task), ABSA_TRAIN)
        train_dataset_target = LLMDataset(
            data_path=str(data_path_train_target),
            tokenizer=tokenizer,
            max_data=args.target_language_few_shot,
            instruction_tuning=True,
            instruction=instruction,
            testing=False,
        )

        train_dataset = torch.utils.data.ConcatDataset([train_dataset, train_dataset_target])

    data_path_dev = os.path.join("data", args.source_language, str(args.task), ABSA_DEV)
    dev_dataset = LLMDataset(
        data_path=str(data_path_dev),
        tokenizer=tokenizer,
        instruction_tuning=True,
        max_data=args.max_data,
        instruction=instruction,
        testing=False,
    )

    if args.target_language_few_shot is not None:
        data_path_dev_target = os.path.join("data", args.target_language, str(args.task), ABSA_DEV)
        dev_dataset_target = LLMDataset(
            data_path=str(data_path_dev_target),
            tokenizer=tokenizer,
            instruction_tuning=True,
            max_data=args.max_data,
            instruction=instruction,
            testing=False,
        )

        dev_dataset = torch.utils.data.ConcatDataset([dev_dataset, dev_dataset_target])

    data_path_test = os.path.join("data", args.target_language, str(args.task), ABSA_TEST)
    test_dataset = LLMDataset(
        data_path=str(data_path_test),
        tokenizer=tokenizer,
        max_data=args.max_data,
        instruction_tuning=True,
        instruction=instruction,
        testing=True,
    )

    output_dir = "output"

    ampere_gpu = False
    if not use_cpu:
        ampere_gpu = check_ampere_gpu()

    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accumulate_grad_batches,
        learning_rate=2e-4,
        logging_steps=10,
        num_train_epochs=args.epochs,
        optim="paged_adamw_32bit",
        report_to=["wandb"] if not args.no_wandb else [],
        lr_scheduler_type="constant",
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        bf16=ampere_gpu,
        tf32=ampere_gpu,
        save_strategy="epoch",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="epoch",
        use_cpu=use_cpu,
        remove_unused_columns=True,
        load_best_model_at_end=True,
        save_total_limit=1,
        metric_for_best_model="eval_loss",
        disable_tqdm=True,
        group_by_length=True,
        dataloader_drop_last=False,
        packing=False,
        dataset_text_field="input_ids",
        max_seq_length=1024,
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    if "orca" in args.model.lower():
        response_template = tokenizer.encode("\n<|im_start|>assistant\n", add_special_tokens=False)[2:]
        assistant_text = "<|im_start|> assistant\n"
    elif "llama-3" in args.model.lower():
        response_template = tokenizer.encode(
            "<|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False
            )
        assistant_text = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    elif "llama" in args.model.lower():
        response_template = tokenizer.encode(" [/INST]", add_special_tokens=False)[1:]
        assistant_text = "[/INST]"
    elif "aya" in args.model.lower():
        response_template = tokenizer.encode("<|CHATBOT_TOKEN|>", add_special_tokens=False)
        assistant_text = "<|CHATBOT_TOKEN|>"
    elif "gemma-3" in args.model.lower():
        response_template = tokenizer.encode("<start_of_turn>model\n", add_special_tokens=False)
        assistant_text = "<start_of_turn>model\n"
    else:
        raise ValueError("Response template not defined for this model.")

    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    trainer = SFTTrainer(
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        model=model,
        peft_config=peft_config,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=collator,
    )

    best_model_dir = "best_model"
    logging.info("Training...")
    trainer.train()
    trainer.save_model(best_model_dir)
    logging.info("Training finished")

    # load best model

    if "gemma-3" in args.model.lower():
        base_model = Gemma3ForConditionalGeneration.from_pretrained(
            args.model,
            quantization_config=bnb_config if not use_cpu else None,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16 if not use_cpu else torch.float32,
            attn_implementation="eager",
        )
        peft_model = PeftModel.from_pretrained(base_model, best_model_dir)
        model = peft_model.merge_and_unload()
    else:
        model = AutoPeftModelForCausalLM.from_pretrained(
            best_model_dir,
            quantization_config=bnb_config,
            low_cpu_mem_usage=True,
        )

    tokenizer.padding_side = "left"
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        collate_fn=functools.partial(data_collate_llm_dataset, tokenizer=tokenizer),
        num_workers=0,
        shuffle=False,
        drop_last=False,
    )

    llm_classify(
        model=model,
        tokenizer=tokenizer,
        data_loader=test_dataloader,
        no_wandb=args.no_wandb,
        assistant_text=assistant_text,
        device=device,
        task=args.task,
    )


def main():
    init_logging()
    args = init_args()
    args.instruction_tuning = True
    # Set system env variable HF_TOKEN to args.token
    if args.token is not None:
        os.environ["HF_TOKEN"] = args.token

    if not args.no_wandb:
        wandb.init(
            project=WANDB_PROJECT_NAME,
            entity=WANDB_ENTITY,
            config=vars(args),
            tags=[args.tag] if args.tag else [],
            settings=wandb.Settings(x_service_wait=300),
        )

    instruction_tuning(args)
    if not args.no_wandb:
        wandb.finish()
    logging.info("This is the end...")


if __name__ == '__main__':
    main()
