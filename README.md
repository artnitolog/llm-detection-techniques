# Text Generated by Large Language Models: Detection Techniques

* **Dataset page: https://huggingface.co/datasets/artnitolog/llm-generated-texts**
* **Paper: [llm-detection-techniques.pdf](llm-detection-techniques.pdf)**

### Benchmarking

To conduct a pairwise fine-tuning experiment, launch `python run.py` with the following hydra config:

```yaml
mode: benchmark
method: finetune_pairwise
source_model: "GPT4 Turbo 2024-04-09"
source_dataset: essay
dataset_seed: 0
preprocess_args:
  process_spaces: true
  words: 50
eval_models:
  - "GPT4 Turbo 2024-04-09"
  - "GPT4 Omni"
  - "Claude 3 Opus"
  - "YandexGPT 3 Pro"
  - "GigaChat Pro"
  - "Llama3 70B"
  - "Command R+"
eval_datasets:
  - essay
  - wp
  - reuters
result_dir: "results"
exp_prefix: "pairwise_run"
exp_suffix: "alpha0.1"
final_result_dir: ""
ft_args:
  max_length: 512
  batch_size: 16
  pairwise_alpha: 0.1
```

This config will train a classifier on `GPT4 Turbo 2024-04-09` vs `human` pairs on `essay` dataa and evaluate on all other llm-generated subsets. Results (metrics, tensorboard logs) will be stored in `result_dir`. To use vanilla BCE fine-tuning, replace `method: finetune_pairwise` with `method: finetune`.

For metric-based methods 2 stages are required: feature generation and benchmarking.

1. To generate features, replace `mode: benchmark` with `mode: get_features`.
2. Run `benchmark` mode with `metric_based_args.features_input` as features path.

Supported metric-based methods: `log_proba`, `log_perplexity`, `entropy`, `rank`, `log_rank`, `gltr`, `lrr`, `fastdetectgpt_sampled`, `fastdetectgpt_analytical`, `metric_ensemble`.

### Environment

```bash
pip install -r requirements.txt
```
