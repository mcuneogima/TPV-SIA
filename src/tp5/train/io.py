"""
tp5.train.io
============
Manejo de entrada/salida para persistencia de resultados, métricas y pesos.
Genera los archivos descriptos en PLAN.md (sección 2.7).
"""

import os
import pathlib
import hashlib
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import yaml


def generate_run_id(cfg: Dict[str, Any]) -> str:
    """
    Genera un identificador único para la corrida combinando:
    run_name + hash corto de la configuración + semilla.
    """
    run_name = cfg.get("run_name", "run")
    seed = cfg.get("seed", 42)
    cfg_str = yaml.dump(cfg, sort_keys=True)
    hash_short = hashlib.sha1(cfg_str.encode("utf-8")).hexdigest()[:8]
    return f"{run_name}_{hash_short}_s{seed}"


def ensure_run_dir(base_dir: str, run_id: str) -> pathlib.Path:
    """Crea y retorna el directorio específico de la corrida."""
    run_dir = pathlib.Path(base_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_config_resolved(run_dir: pathlib.Path, cfg: Dict[str, Any]) -> pathlib.Path:
    """Guarda el archivo config_resolved.yaml en el directorio de la corrida."""
    out_path = run_dir / "config_resolved.yaml"
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    return out_path


def save_epochs_csv(run_dir: pathlib.Path, history: List[Dict[str, Any]]) -> pathlib.Path:
    """Guarda el historial de épocas en epochs.csv."""
    out_path = run_dir / "epochs.csv"
    df = pd.DataFrame(history)
    df.to_csv(out_path, index=False)
    return out_path


def save_weights_npz(run_dir: pathlib.Path, weights: Dict[str, np.ndarray]) -> pathlib.Path:
    """Guarda los pesos del modelo en weights.npz."""
    out_path = run_dir / "weights.npz"
    np.savez_compressed(out_path, **weights)
    return out_path


def load_weights_npz(weights_path: str) -> Dict[str, np.ndarray]:
    """Carga los pesos del modelo desde weights.npz."""
    npz = np.load(weights_path, allow_pickle=True)
    return {k: npz[k] for k in npz.files}


def save_latent_csv(
    run_dir: pathlib.Path,
    latent_data: np.ndarray,
    labels: Optional[List[str]] = None,
    logvar_data: Optional[np.ndarray] = None,
) -> pathlib.Path:
    """
    Guarda las coordenadas latentes en latent.csv:
    index, label, z1, z2 ... (y logvar1, logvar2 ... si es VAE).
    """
    out_path = run_dir / "latent.csv"
    n_samples, latent_dim = latent_data.shape

    cols: Dict[str, Any] = {
        "index": np.arange(n_samples),
        "label": labels if labels is not None else [f"sample_{i}" for i in range(n_samples)],
    }

    for d in range(latent_dim):
        cols[f"z{d + 1}"] = latent_data[:, d]

    if logvar_data is not None:
        for d in range(logvar_data.shape[1]):
            cols[f"logvar{d + 1}"] = logvar_data[:, d]

    df = pd.DataFrame(cols)
    df.to_csv(out_path, index=False)
    return out_path


def save_reconstructions_npz(
    run_dir: pathlib.Path,
    X_clean: np.ndarray,
    X_hat: np.ndarray,
    X_noisy: Optional[np.ndarray] = None,
) -> pathlib.Path:
    """Guarda las entradas originales, reconstrucciones y ruidosas en reconstructions.npz."""
    out_path = run_dir / "reconstructions.npz"
    payload = {
        "X_clean": X_clean,
        "X_hat": X_hat,
    }
    if X_noisy is not None:
        payload["X_noisy"] = X_noisy
    np.savez_compressed(out_path, **payload)
    return out_path


def append_summary_run(summary_dir: str, run_summary: Dict[str, Any]) -> pathlib.Path:
    """
    Agrega una fila al CSV consolidado de resultados: results/summary/runs.csv.
    Crea el archivo si no existe, o añade la fila si ya existe.
    """
    sum_dir = pathlib.Path(summary_dir)
    sum_dir.mkdir(parents=True, exist_ok=True)
    csv_path = sum_dir / "runs.csv"

    df_row = pd.DataFrame([run_summary])
    if not csv_path.exists():
        df_row.to_csv(csv_path, index=False)
    else:
        df_row.to_csv(csv_path, mode="a", header=False, index=False)

    return csv_path
