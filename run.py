import json
from pathlib import Path

import hydra
from data_utils import get_dataset, read_dataset
from methods.finetune import benchmark_finetune
from methods.metric_based import benchmark_metric_based, get_features_metric_based
from omegaconf import DictConfig, OmegaConf
from utils import final_dir


def benchmark(cfg: DictConfig) -> None:
    cfg.final_result_dir = final_dir(cfg)
    if cfg.method in {"finetune", "finetune_pairwise"}:
        dataset = get_dataset(cfg)
        results = benchmark_finetune(cfg, dataset)
    elif cfg.method in {
        "log_proba",
        "log_perplexity",
        "entropy",
        "rank",
        "log_rank",
        "gltr",
        "lrr",
        "fastdetectgpt_sampled",
        "fastdetectgpt_analytical",
    }:
        dataset = get_dataset(cfg, cfg.metric_based_args.features_input)
        results = benchmark_metric_based(cfg, dataset)
    else:
        raise ValueError(f"Unknown benchmark method: {cfg.method}")
    with open(cfg.final_result_dir / "metrics.json", "w") as fout:
        json.dump(results, fout)


def get_features(cfg: DictConfig) -> None:
    dataset = read_dataset(cfg)
    if cfg.method == "metric_based":
        results = get_features_metric_based(cfg, dataset)
    else:
        raise ValueError(f"Unknown features method: {cfg.method}")
    out_dir = Path(cfg.result_dir) / f"features#{cfg.method}#{cfg.exp_suffix}.jsonl"
    results.to_json(out_dir, lines=True, orient="records")


@hydra.main(version_base=None, config_path=".", config_name="config")
def run(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    if cfg.mode == "get_features":
        get_features(cfg)
    elif cfg.mode == "benchmark":
        benchmark(cfg)
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")


if __name__ == "__main__":
    run()
