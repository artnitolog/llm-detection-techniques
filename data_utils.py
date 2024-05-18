import re

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from omegaconf import DictConfig


def preprocess_text(text: str, cfg: DictConfig) -> str:
    if cfg.preprocess_args.process_spaces:
        text = (
            text.replace("\n\n", "\n")
            .replace(" ,", ",")
            .replace(" .", ".")
            .replace(" ?", "?")
            .replace(" !", "!")
            .replace(" ;", ";")
            .replace(" '", "'")
            .replace(" ’ ", "'")
            .replace(" :", ":")
            .replace("<newline>", "\n")
            .replace("`` ", '"')
            .replace(" ''", '"')
            .replace("''", '"')
            .replace(".. ", "... ")
            .replace(" )", ")")
            .replace("( ", "(")
            .replace(" n't", "n't")
            .replace(" i ", " I ")
            .replace(" i'", " I'")
            .replace("\\'", "'")
            .replace("\n ", "\n")
            .strip()
        )
    if cfg.preprocess_args.words > 0:
        n_groups = cfg.preprocess_args.words * 2
        text = "".join(re.split("(\W+)", text)[:n_groups])
    text = text.strip()
    return text


def get_train_test_ids(total_ids: int = 1000, train_frac: float = 0.8, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    ids = np.arange(1, 1 + total_ids)
    rng.shuffle(ids)
    n_train_ids = int(train_frac * total_ids)
    return ids[:n_train_ids], ids[n_train_ids:]


def read_dataset(cfg: DictConfig) -> pd.DataFrame:
    df = load_dataset("artnitolog/llm-generated-texts")["train"].to_pandas()
    cols = cfg.eval_models + ["human"]
    df.loc[:, cols] = df.loc[:, cols].applymap(lambda x: preprocess_text(x, cfg))
    return df


def read_features(path: str) -> pd.DataFrame:
    df = pd.read_json(path, lines=True)
    return df


def split_dataset_sources(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    results = {}
    for source in df.dataset_name.unique():
        results[source] = df[df.dataset_name == source]
    return results


def split_dataset_to_hf(
    dataset: dict[str, pd.DataFrame], total_ids: int = 1000, train_frac: float = 0.8, seed: int = 0
) -> dict[str, dict[str, pd.DataFrame]]:
    results = {}
    train_ids, test_ids = get_train_test_ids(total_ids, train_frac, seed)
    for source, df in dataset.items():
        results[source] = {}
        results[source]["train"] = Dataset.from_pandas(df[df.id.isin(train_ids)], preserve_index=False)
        results[source]["test"] = Dataset.from_pandas(df[df.id.isin(test_ids)], preserve_index=False)
    return DatasetDict(results)


def get_dataset(cfg: DictConfig, features_path: str | None = None) -> DatasetDict:
    if not features_path:
        dataset = read_dataset(cfg)
    else:
        dataset = read_features(features_path)
    dataset = split_dataset_sources(dataset)
    dataset = split_dataset_to_hf(dataset, seed=cfg.dataset_seed)
    return dataset
