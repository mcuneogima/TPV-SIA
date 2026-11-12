"""
tp5.train
=========
Módulo de entrenamiento, métricas, IO, configuración y sweeps.
"""

from .config import load_config
from .metrics import compute_metrics
from .io import (
    generate_run_id,
    ensure_run_dir,
    save_config_resolved,
    save_epochs_csv,
    save_weights_npz,
    load_weights_npz,
    save_latent_csv,
    save_reconstructions_npz,
    append_summary_run,
)
from .trainer import Trainer
from .sweep import expand_sweep_grid, run_sweep

__all__ = [
    "load_config",
    "compute_metrics",
    "generate_run_id",
    "ensure_run_dir",
    "save_config_resolved",
    "save_epochs_csv",
    "save_weights_npz",
    "load_weights_npz",
    "save_latent_csv",
    "save_reconstructions_npz",
    "append_summary_run",
    "Trainer",
    "expand_sweep_grid",
    "run_sweep",
]
