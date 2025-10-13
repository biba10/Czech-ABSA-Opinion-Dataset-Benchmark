import functools
import logging
import os

import wandb
from torch.utils.data import DataLoader

from src.data_utils.data_utils import data_collate_llm_dataset
from src.data_utils.llm_dataset import LLMDataset
from src.llm_prompting.llm_classifier import llm_classify
from src.llm_prompting.templates import INSTRUCTIONS, append_few_shot_examples_to_instruction
from src.model.model_tokenizer_loader import llm_model_tokenizer_loader
from src.utils.args_utils import init_args
from src.utils.config import WANDB_PROJECT_NAME, WANDB_ENTITY, ABSA_TEST, ABSA_TRAIN
from src.utils.logger_utils import init_logging

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # For FastTokenizerspip


def main():
    init_logging()
    args = init_args()

    if not args.few_shot_prompt:
        args.zero_shot = True

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

    logging.info("Loading tokenizer and model...")
    model, tokenizer, _, device, _ = llm_model_tokenizer_loader(args.model, args.token)
    logging.info("Tokenizer and model loaded")

    tokenizer.padding_side = "left"

    if tokenizer.chat_template is None:
        if "microsoft/Orca" in args.model:
            tokenizer.chat_template = "{{ bos_token }} {% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

    if "orca" in args.model.lower():
        assistant_text = "<|im_start|> assistant\n"
    elif "llama-3" in args.model.lower():
        assistant_text = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    elif "llama" in args.model.lower():
        assistant_text = "[/INST]"
    elif "aya" in args.model.lower():
        assistant_text = "<|CHATBOT_TOKEN|>"
    elif "gemma-3" in args.model.lower():
        assistant_text = "<start_of_turn>model\n"
    else:
        raise ValueError("Response template not defined for this model.")

    instruction = INSTRUCTIONS.get(str(args.task), None)
    if instruction is None:
        raise ValueError(f"Instruction not found for {args.task}")

    if args.few_shot_prompt:
        data_path = os.path.join("data", args.target_language, str(args.task), ABSA_TRAIN)
        instruction = append_few_shot_examples_to_instruction(instruction, data_path)

    logging.info("Creating Data loader")
    data_path_test = os.path.join("data", args.target_language, str(args.task), ABSA_TEST)
    test_dataset = LLMDataset(
        data_path=str(data_path_test),
        tokenizer=tokenizer,
        max_data=args.max_data,
        instruction_tuning=True,
        instruction=instruction,
        testing=True,
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

    logging.info("Data loader created")

    llm_classify(
        model=model,
        tokenizer=tokenizer,
        data_loader=test_dataloader,
        no_wandb=args.no_wandb,
        assistant_text=assistant_text,
        device=device,
        task=args.task,
    )

    if not args.no_wandb:
        wandb.finish()
    logging.info("This is the end...")


if __name__ == '__main__':
    main()
