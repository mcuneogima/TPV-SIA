"""
loader.py
=========
Interfaz unificada de carga de datasets.

`load_dataset(cfg) -> dict` es el único punto de entrada para el trainer.
Retorna siempre:
  X_train      : np.ndarray float64 (n_train, features)
  X_val        : np.ndarray float64 (n_val, features) | None
  labels       : list[str] | None
  image_shape  : tuple — shape original de cada muestra (H, W) o (H, W, C)
"""

import pathlib
import numpy as np


def load_dataset(cfg: dict) -> dict:
    """
    Carga el dataset especificado en cfg['data'].

    cfg['data'] puede contener:
      dataset   : 'font' | 'sprites'
      path      : ruta al .npz procesado
      val_split : fracción para validación (default 0.0, solo para font)
      seed      : semilla del split (default 42)
    """
    data_cfg = cfg.get("data", {})
    dataset = data_cfg.get("dataset", "font")

    if dataset == "font":
        return _load_font(data_cfg)
    elif dataset == "sprites":
        return _load_sprites(data_cfg)
    else:
        raise ValueError(f"Dataset desconocido: '{dataset}'. Opciones: font, sprites")


# ---------------------------------------------------------------------------
# Loaders internos
# ---------------------------------------------------------------------------

def _load_font(data_cfg: dict) -> dict:
    path = data_cfg.get("path", "data/processed/font.npz")
    path = pathlib.Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró {path}. "
            "Ejecutá primero: python scripts/preprocess_font.py"
        )

    npz = np.load(path, allow_pickle=True)
    X = npz["X"].astype(np.float64)
    labels = list(npz["labels"])
    shape = tuple(npz["shape"])

    val_split = float(data_cfg.get("val_split", 0.0))
    seed = int(data_cfg.get("seed", 42))

    if val_split > 0.0:
        X_train, X_val = _split(X, val_split, seed)
    else:
        X_train, X_val = X, None

    return {
        "X_train": X_train,
        "X_val": X_val,
        "labels": labels,
        "image_shape": shape,
    }


def _load_sprites(data_cfg: dict) -> dict:
    path = data_cfg.get("path", "data/processed/sprites.npz")
    path = pathlib.Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró {path}. "
            "Ejecutá primero: python scripts/preprocess_sprites.py"
        )

    npz = np.load(path, allow_pickle=True)
    X_train = npz["X_train"].astype(np.float64)
    X_val   = npz["X_val"].astype(np.float64) if "X_val" in npz else None
    shape   = tuple(npz["shape"])

    return {
        "X_train": X_train,
        "X_val": X_val,
        "labels": None,
        "image_shape": shape,
    }


def _split(X: np.ndarray, val_split: float, seed: int) -> tuple:
    """Divide X en train y val de forma reproducible."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    n_val = max(1, int(len(X) * val_split))
    return X[idx[n_val:]], X[idx[:n_val]]
