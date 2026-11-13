"""
tp5.analysis.latent
===================
Utilidades para análisis y exploración del espacio latente 2D.
Carga modelos entrenados, mapeos latentes y evaluación de variedades (manifolds).
"""

import pathlib
from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
import yaml

from ..models.registry import build_model
from ..train.io import load_weights_npz


def load_model_from_run(run_dir: str | pathlib.Path) -> Tuple[Any, Dict[str, Any]]:
    """
    Carga el modelo y la configuración serializada de una corrida específica.
    Restaura los pesos guardados en weights.npz.
    """
    run_path = pathlib.Path(run_dir)
    cfg_path = run_path / "config_resolved.yaml"
    weights_path = run_path / "weights.npz"

    if not cfg_path.exists():
        raise FileNotFoundError(f"No se encontró config_resolved.yaml en {run_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"No se encontró weights.npz en {run_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # font dataset tiene entrada 35
    input_dim = 35
    if cfg.get("data", {}).get("dataset") == "sprites":
        channels = cfg.get("data", {}).get("channels", "rgb")
        input_dim = 768 if channels == "rgb" else 256

    model = build_model(cfg, input_dim=input_dim)
    weights = load_weights_npz(str(weights_path))
    model.set_weights(weights)

    return model, cfg


def load_latent_data(run_dir: str | pathlib.Path) -> pd.DataFrame:
    """Carga el archivo latent.csv de la corrida."""
    run_path = pathlib.Path(run_dir)
    latent_path = run_path / "latent.csv"
    if not latent_path.exists():
        raise FileNotFoundError(f"No se encontró latent.csv en {run_path}")
    return pd.read_csv(latent_path)


def evaluate_decoder_grid(
    model: Any,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    steps: int = 15,
    shape: Tuple[int, int] = (7, 5),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evalúa el decoder en una grilla regular 2D dentro de los límites del espacio latente.
    Retorna (grid_x, grid_y, decoded_images) donde decoded_images tiene forma (steps, steps, 7, 5).
    """
    grid_x = np.linspace(x_min, x_max, steps)
    grid_y = np.linspace(y_min, y_max, steps)

    mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)
    z_points = np.stack([mesh_x.ravel(), mesh_y.ravel()], axis=1)

    # Evaluar decoder
    decoded = model.decode(z_points)  # shape (steps*steps, 35)
    decoded_images = decoded.reshape(steps, steps, shape[0], shape[1])

    return grid_x, grid_y, decoded_images


def find_nearest_character(
    df_latent: pd.DataFrame,
    z_query: np.ndarray,
) -> Tuple[str, float, int]:
    """
    Encuentra el caracter del dataset más cercano a una coordenada latente de consulta.
    Retorna (label, distancia, index).
    """
    coords = df_latent[["z1", "z2"]].values
    dists = np.linalg.norm(coords - z_query.reshape(1, 2), axis=1)
    best_idx = int(np.argmin(dists))
    return str(df_latent.iloc[best_idx]["label"]), float(dists[best_idx]), best_idx
