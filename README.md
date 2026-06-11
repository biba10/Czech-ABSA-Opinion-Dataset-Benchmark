# Czech_Opinion_ABSA

This project contains a dataset and code for Aspect-Based Sentiment Analysis (ABSA) in the Czech language, focusing on restaurant reviews. The dataset includes annotations of opinion terms. Published as <a href="https://lrec.elra.info/conference/2026">LREC 2026</a> paper title "Extending Czech Aspect-Based Sentiment Analysis with Opinion Terms: Dataset and LLM Benchmarks".

---

## 🐍 Requirements

- **Python ≥ 3.10**
- Install dependencies via:
  ```bash
  pip install -r requirements.txt
  
## 🚀 Scripts Overview
### `main.py`

For training sequence-to-sequence models such as google/mt5-large.

Example:
`python main.py --model google/mt5-large --lr 0.0001 --optimizer AdamW --mode dev --epochs 20 --batch_size 4 --accumulate_grad_batches 4 --task aste --checkpoint_monitor f1 --source_language en --no_wandb`
  
### `llm.py`

For fine-tuning large language models (LLMs) such as LLaMA or similar instruction-tuned models.

Example:

`python llms.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --task asqp --epochs 5 --no_wandb`

### `llm_zero.py`

For zero-shot or few-shot inference with large language models (LLMs) such as LLaMA or similar instruction-tuned models.

Example:
`python llm_zero.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --task acos --no_wandb`

### ⚙️ Main Arguments

| Argument                     | Type    | Default   | Description                                                                    |
| ---------------------------- | ------- | --------- | ------------------------------------------------------------------------------ |
| `--model`                    | `str`   | `t5-base` | Path or name of the pre-trained model.                                         |
| `--batch_size`               | `int`   | `16`      | Training batch size.                                                           |
| `--lr`                       | `float` | `1e-4`    | Learning rate.                                                                 |
| `--epochs`                   | `int`   | `10`      | Number of training epochs.                                                     |
| `--optimizer`                | `str`   | `AdamW`   | Optimiser (`AdamW` or `Adafactor`).                                            |
| `--mode`                     | `str`   | `dev`     | Training mode (`dev` for validation-based selection, `test` for fixed epochs). |
| `--checkpoint_monitor`       | `str`   | `f1`      | Metric to monitor for saving the best model.                                   |
| `--accumulate_grad_batches`  | `int`   | `1`       | Number of batches to accumulate gradients over.                                |
| `--beam_size`                | `int`   | `1`       | Beam size for beam search decoding.                                            |
| `--task`                     | `Task`  | `acos`    | Task type (enum).                                                              |
| `--source_language`          | `str`   | `cs`      | Training dataset language.                                                     |
| `--target_language`          | `str`   | `cs`      | Test dataset language.                                                         |
| `--target_language_few_shot` | `int`   | `None`    | Number of few-shot examples (None = no examples, 0 = all examples).            |
| `--max_data`                 | `int`   | `0`       | Number of training examples (0 = all).                                         |
| `--train_translated`         | flag    | —         | Train on translated data.                                                      |
| `--constrained_decoding`     | flag    | —         | Use constrained decoding (for seq2seq models).                                 |
| `--few_shot_prompt`          | flag    | —         | Use few-shot prompting during training.                                        |
| `--instruction_tuning`       | flag    | —         | Enable instruction tuning.                                                     |
| `--no_wandb`                 | flag    | —         | Disable Weights & Biases logging.                                              |
| `--tag`                      | `str`   | `opinion` | WandB tag.                                                                     |
| `--token`                    | `str`   | `None`    | Token for model access (e.g. Hugging Face).                                    |
| `--lora_r`                   | `int`   | `64`      | LoRA rank parameter.                                                           |
| `--lora_alpha`               | `int`   | `16`      | LoRA alpha parameter.                                                          |

## Citation
If you find this repository helpful for your research, please cite our paper as follows:
```
@inproceedings{smid-etal-2026-extending,
  title = {Extending Czech Aspect-Based Sentiment Analysis with Opinion Terms: Dataset and LLM Benchmarks},
  author = {\v{S}}m{\'i}íd, Jakub and Priban, Pavel and Kral, Pavel},
  booktitle = {Proceedings of the Fifteenth Language Resources and Evaluation Conference (LREC 2026)},
  month = {May},
  year = {2026},
  pages = {7973--7984},
  address = {Palma, Mallorca, Spain},
  publisher = {European Language Resources Association (ELRA)},
  editor = {Piperidis, Stelios and Bel, Núria and van den Heuvel, Henk and Ide, Nancy and Krek, Simon and Toral, Antonio},
  doi = {10.63317/4hkzdnwfztkz},
  abstract = {This paper introduces a novel Czech dataset in the restaurant domain for aspect-based sentiment analysis (ABSA), enriched with annotations of opinion terms. The dataset supports three distinct ABSA tasks involving opinion terms, accommodating varying levels of complexity. Leveraging this dataset, we conduct extensive experiments using modern Transformer-based models, including large language models (LLMs), in monolingual, cross-lingual, and multilingual settings. To address cross-lingual challenges, we propose a translation and label alignment methodology leveraging LLMs, which yields consistent improvements. Our results highlight the strengths and limitations of state-of-the-art models, especially when handling the linguistic intricacies of low-resource languages like Czech. A detailed error analysis reveals key challenges, including the detection of subtle opinion terms and nuanced sentiment expressions. The dataset establishes a new benchmark for Czech ABSA, and our proposed translation–alignment approach offers a scalable solution for adapting ABSA resources to other low-resource languages.}
}
```
