from pathlib import Path
from omegaconf import DictConfig


def final_dir(cfg: DictConfig):
    name = f"{cfg.source_dataset}#{cfg.source_model}"
    dir_name = f"{cfg.method}#{cfg.exp_suffix}"
    path = Path(cfg.result_dir) / cfg.exp_prefix / dir_name / name
    path.mkdir(parents=True, exist_ok=True)
    return path
