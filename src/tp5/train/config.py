"""
tp5.train.config
================
Carga, resolución de valores por defecto y validación de esquemas de configuración YAML.
"""

import copy
import pathlib
from typing import Dict, Any
import yaml


DEFAULT_CONFIG: Dict[str, Any] = {
    "run_name": "run",
    "seed": 42,
    "data": {
        "dataset": "font",
        "path": "data/processed/font.npz",
        "val_split": 0.0,
    },
    "model": {
        "type": "autoencoder",
        "latent_dim": 2,
        "init": "xavier",
        "encoder": {"layers": []},
        "decoder": {"layers": []},
    },
    "loss": {
        "recon": "bce",
        "beta": 1.0,
        "beta_warmup_epochs": 0,
    },
    "noise": {
        "kind": "flip",
        "level": 0.0,
        "regenerate_each_epoch": True,
    },
    "optimizer": {
        "name": "adam",
        "lr": 0.001,
        "momentum": 0.9,
        "beta1": 0.9,
        "beta2": 0.999,
        "decay": 0.9,
        "schedule": {"name": "constant"},
        "grad_clip": None,
    },
    "train": {
        "epochs": 1000,
        "batch_size": "full",
        "eval_every": 10,
        "early_stop": None,
    },
    "output": {
        "dir": "results/runs",
        "summary_dir": "results/summary",
        "save_weights": True,
    },
}


def _deep_update(source: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Actualiza recursivamente un diccionario preservando valores anidados por defecto."""
    result = copy.deepcopy(source)
    for key, value in overrides.items():
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Carga un archivo YAML y fusiona con los valores por defecto del sistema.
    """
    path = pathlib.Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Archivo de configuración no encontrado: {path}")

    with open(path, "r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}

    # Si run_name no está definido, usar el nombre del archivo sin extensión
    if "run_name" not in user_cfg:
        user_cfg["run_name"] = path.stem

    resolved_cfg = _deep_update(DEFAULT_CONFIG, user_cfg)
    return resolved_cfg
