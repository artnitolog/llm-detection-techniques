from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from datasets import Dataset, DatasetDict
from evaluate import eval_metrics
from omegaconf import DictConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel, Trainer, TrainingArguments
from transformers.trainer_pt_utils import nested_detach
from trl import RewardConfig, RewardTrainer
from dataclasses import dataclass

@dataclass
class PairwiseConfig(RewardConfig):
    pairwise_alpha: float = 0.0

class PairwiseTrainer(RewardTrainer):
    def compute_loss(
        self,
        model: Union[PreTrainedModel, torch.nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        rewards_chosen = model(
            input_ids=inputs["input_ids_chosen"],
            attention_mask=inputs["attention_mask_chosen"],
            return_dict=True,
        )["logits"]
        rewards_rejected = model(
            input_ids=inputs["input_ids_rejected"],
            attention_mask=inputs["attention_mask_rejected"],
            return_dict=True,
        )["logits"]
        # calculate loss, optionally modulate with margin
        logits = torch.cat([rewards_chosen, rewards_rejected], dim=0)
        labels = torch.zeros_like(logits)
        labels[:rewards_chosen.shape[0]] = 1.0
        loss_ce = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        
        if "margin" in inputs:
            loss_pairwise = -torch.nn.functional.logsigmoid(rewards_chosen - rewards_rejected - inputs["margin"]).mean()
        else:
            loss_pairwise = -torch.nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()

        loss = loss_ce + loss_pairwise * self.args.pairwise_alpha

        if return_outputs:
            return loss, {
                "rewards_chosen": rewards_chosen,
                "rewards_rejected": rewards_rejected,
            }
        return loss

    def prediction_step(
        self,
        model: Union[PreTrainedModel, torch.nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        with torch.no_grad():
            loss, logits_dict = self.compute_loss(model, inputs, return_outputs=True)

        logits = torch.cat([logits_dict["rewards_chosen"], logits_dict["rewards_rejected"]], dim=0)
        labels = torch.zeros_like(logits)
        labels[: logits_dict["rewards_chosen"].shape[0]] = 1.0

        if prediction_loss_only:
            return (loss, None, None)
        return loss, logits, labels


def get_model_tokenizer(model_name):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=1, problem_type="multi_label_classification"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def to_vanilla(encodings, size):
    labels = [[1.0] for _ in range(size)] + [[0.0] for _ in range(size)]
    result_dict = dict(
        input_ids=encodings["input_ids"],
        attention_mask=encodings["attention_mask"],
        labels=labels,
    )
    return Dataset.from_dict(result_dict)


from tokenizers import Encoding


def to_pairwise(encodings, size):
    chosen = encodings[:size]
    rejected = encodings[size:]
    result_dict = dict(
        input_ids_chosen=[row.ids for row in chosen],
        attention_mask_chosen=[row.attention_mask for row in chosen],
        input_ids_rejected=[row.ids for row in rejected],
        attention_mask_rejected=[row.attention_mask for row in rejected],
    )
    return Dataset.from_dict(result_dict)


def preprocess(dataset, tokenizer, generated_name, human_name, max_length=512, pairwise=False):
    size = len(dataset[generated_name])
    encodings = tokenizer(
        dataset[generated_name] + dataset[human_name], truncation=True, padding="max_length", max_length=max_length
    )
    if pairwise:
        return to_pairwise(encodings, size)
    else:
        return to_vanilla(encodings, size)


def compute_metrics(inputs):
    logits = inputs.predictions
    predicted_labels = logits > 0.0
    labels = inputs.label_ids.astype(int)
    return eval_metrics(logits, predicted_labels, labels)


def args_cls(pairwise: bool):
    if pairwise:
        return PairwiseConfig
    else:
        return TrainingArguments


def trainer_cls(pairwise: bool):
    if pairwise:
        return PairwiseTrainer
    else:
        return Trainer


def benchmark_finetune(cfg: DictConfig, dataset: DatasetDict):
    is_pairwise = cfg.method.endswith("pairwise")
    model, tokenizer = get_model_tokenizer("distilbert-base-uncased")
    train = preprocess(
        dataset=dataset[cfg.source_dataset]["train"],
        tokenizer=tokenizer,
        generated_name=cfg.source_model,
        human_name="human",
        max_length=cfg.ft_args.max_length,
        pairwise=is_pairwise,
    )
    val = preprocess(
        dataset=dataset[cfg.source_dataset]["test"],
        tokenizer=tokenizer,
        generated_name=cfg.source_model,
        human_name="human",
        max_length=cfg.ft_args.max_length,
        pairwise=is_pairwise,
    )
    if is_pairwise:
        batch_size = cfg.ft_args.batch_size // 2
        kwargs = dict(
            max_length=cfg.ft_args.max_length,
            pairwise_alpha=cfg.ft_args.pairwise_alpha,
        )
    else:
        batch_size = cfg.ft_args.batch_size
        kwargs = {}
    args = args_cls(is_pairwise)(
        cfg.final_result_dir,
        save_strategy="no",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=64,
        learning_rate=1e-5,
        weight_decay=0.01,
        warmup_ratio=0.0,
        logging_steps=20,
        eval_steps=20,
        evaluation_strategy="steps",
        max_steps=300,
        report_to="tensorboard",
        **kwargs
    )
    trainer = trainer_cls(is_pairwise)(
        model=model.to("cuda"),
        tokenizer=tokenizer,
        args=args,
        train_dataset=train,
        eval_dataset=val,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    model.eval()
    results = {"models": {}, "datasets": {}}
    for model_name in cfg.eval_models:
        ood_data = preprocess(
            dataset=dataset[cfg.source_dataset]["test"],
            tokenizer=tokenizer,
            generated_name=model_name,
            human_name="human",
            pairwise=is_pairwise,
        )
        preds = trainer.predict(ood_data)
        results["models"][model_name] = compute_metrics(preds)
    for dataset_name in cfg.eval_datasets:
        ood_data = preprocess(
            dataset=dataset[dataset_name]["test"],
            tokenizer=tokenizer,
            generated_name=cfg.source_model,
            human_name="human",
            pairwise=is_pairwise,
        )
        preds = trainer.predict(ood_data)
        results["datasets"][dataset_name] = compute_metrics(preds)

    return results
