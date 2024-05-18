import numpy as np
import pandas as pd
import torch
from datasets import DatasetDict
from evaluate import eval_metrics
from omegaconf import DictConfig
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_model_tokenizer(model_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype="auto",
        attn_implementation="flash_attention_2",
        # torch_dtype=torch.float,
        # attn_implementation="sdpa",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    tokenizer.model_max_length = model.config.max_position_embeddings
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def fastdetectgpt_features(logsoftmax, labels, log_perplexity):
    dist = torch.distributions.Categorical(logits=logsoftmax)
    samples = dist.sample([10000])
    logits_tilde = torch.gather(logsoftmax, 1, samples.T).mean(0)
    discrepancy_sampled = (log_perplexity - logits_tilde.mean()) / logits_tilde.std()
    std_analytical = ((dist.probs * (dist.logits**2)).sum(-1) - (dist.entropy() ** 2)).sum().sqrt() / len(labels)
    discrepancy_analytical = (log_perplexity + dist.entropy().mean()) / std_analytical
    result = {}
    result["fastdetectgpt_sampled"] = discrepancy_sampled.item()
    result["fastdetectgpt_analytical"] = discrepancy_analytical.item()
    return result


def logits_to_features(logits, labels):
    result = {}
    logsoftmax = logits.log_softmax(dim=-1)
    gathered_logsoftmax = torch.gather(logsoftmax, 1, labels.view(-1, 1))
    result["log_proba"] = gathered_logsoftmax.sum().item()
    result["log_perplexity"] = gathered_logsoftmax.mean().item()
    result["entropy"] = torch.distributions.Categorical(logits=logsoftmax).entropy().mean().item()
    ranks = (logsoftmax.argsort(dim=-1, descending=True) == labels.view(-1, 1)).nonzero()[:, 1]
    result["rank"] = (ranks + 1).float().mean().item()
    result["log_rank"] = (ranks + 1).log().mean().item()
    counts = torch.tensor(
        [
            (ranks < 10).sum(),
            ((ranks >= 10) & (ranks < 100)).sum(),
            ((ranks >= 100) & (ranks < 1000)).sum(),
            (ranks >= 1000).sum(),
        ]
    )
    assert (counts.sum() == logsoftmax.shape[0]).item()
    result["gltr"] = (counts / counts.sum()).tolist()
    result["lrr"] = abs(result["log_perplexity"] / result["log_rank"])
    result.update(fastdetectgpt_features(logsoftmax, labels, result["log_perplexity"]))
    return result


@torch.inference_mode()
def texts_to_features(texts, model, tokenizer, batch_size, max_length, post_processing=logits_to_features):
    features = []
    for text_batch in tqdm(DataLoader(texts, shuffle=False, batch_size=batch_size)):
        tokenized = tokenizer(
            text_batch, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt"
        ).to("cuda")
        logits_batch = model(**tokenized).logits
        for logits, input_ids, attention_mask in zip(logits_batch, tokenized["input_ids"], tokenized["attention_mask"]):
            mask = attention_mask.to(bool)
            labels = input_ids[mask][1:]
            logits = logits[mask][:-1]
            features.append(post_processing(logits, labels))
    return features


def get_features_metric_based(cfg: DictConfig, dataset: pd.DataFrame):
    model, tokenizer = get_model_tokenizer("google/gemma-2b")
    model.to("cuda")
    for eval_model in cfg.eval_models + ["human"]:
        texts = list(dataset.loc[:, eval_model])
        features = texts_to_features(
            texts,
            model,
            tokenizer,
            batch_size=cfg.metric_based_args.batch_size,
            max_length=cfg.metric_based_args.max_length,
        )
        dataset.loc[:, eval_model] = features
    return dataset


def preprocess(
    data_model: list[dict[str, float | list[float]]], data_human: list[dict[str, float | list[float]]], method: str
):
    features_model = np.asarray([row[method] for row in data_model])
    features_human = np.asarray([row[method] for row in data_human])
    assert len(features_model) == len(features_human)
    features = np.concatenate([features_model, features_human], 0)
    if len(features.shape) == 1:
        features = features.reshape(-1, 1)

    labels = np.zeros(len(features_model) + len(features_human), dtype=int)
    labels[: len(features_model)] = 1

    return features, labels


def benchmark_metric_based(cfg: DictConfig, dataset: DatasetDict):
    X_train, y_train = preprocess(
        dataset[cfg.source_dataset]["train"][cfg.source_model],
        dataset[cfg.source_dataset]["train"]["human"],
        cfg.method,
    )
    model = LogisticRegression().fit(X_train, y_train)

    results = {"models": {}, "datasets": {}}
    for model_name in tqdm(cfg.eval_models):
        X_test, y_test = preprocess(
            dataset[cfg.source_dataset]["test"][model_name], dataset[cfg.source_dataset]["test"]["human"], cfg.method
        )
        proba = model.predict_proba(X_test)[:, 1]
        predicted_labels = (proba > 0.5).astype(int)
        results["models"][model_name] = eval_metrics(proba, predicted_labels, y_test)
    for dataset_name in tqdm(cfg.eval_datasets):
        X_test, y_test = preprocess(
            dataset[dataset_name]["test"][cfg.source_model], dataset[cfg.source_dataset]["test"]["human"], cfg.method
        )
        proba = model.predict_proba(X_test)[:, 1]
        predicted_labels = (proba > 0.5).astype(int)
        results["datasets"][dataset_name] = eval_metrics(proba, predicted_labels, y_test)

    return results
