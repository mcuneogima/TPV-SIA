"""
tp5.train.metrics
=================
Cálculo de métricas de desempeño y evaluación.
Métricas para datos binarios (font) y continuos (sprites).
"""

from typing import Dict, Any, Optional
import numpy as np


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    recon_loss: float,
    kl_loss: float = 0.0,
    is_binary: bool = True,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Calcula métricas globales de evaluación sobre un dataset completo.
    
    y_true: Matriz ground-truth (N, features).
    y_pred: Matriz predicha por el modelo (N, features).
    recon_loss: Pérdida de reconstrucción promedio.
    kl_loss: Divergencia KL promedio (0.0 si no es VAE).
    is_binary: True para patrones tipo font (0/1), False para sprites continuos.
    threshold: Umbral de binarización para métricas discretas de píxel.
    """
    total_loss = float(recon_loss + kl_loss)
    metrics: Dict[str, Any] = {
        "loss": total_loss,
        "recon_loss": float(recon_loss),
        "kl_loss": float(kl_loss),
    }

    if is_binary:
        pred_bin = (y_pred >= threshold).astype(np.int64)
        true_bin = (y_true >= threshold).astype(np.int64)

        # Diferencia de píxeles por muestra (distancia de Hamming individual)
        diff_per_sample = np.sum(pred_bin != true_bin, axis=1)

        metrics["max_pixel_errors"] = int(np.max(diff_per_sample))
        metrics["mean_pixel_errors"] = float(np.mean(diff_per_sample))
        metrics["n_patterns_ok"] = int(np.sum(diff_per_sample <= 1))
        metrics["n_patterns_perfect"] = int(np.sum(diff_per_sample == 0))
    else:
        # Métricas para datos continuos (ej. sprites)
        mse_pixel = float(np.mean((y_pred - y_true) ** 2))
        mae_pixel = float(np.mean(np.abs(y_pred - y_true)))
        metrics["mse"] = mse_pixel
        metrics["mae"] = mae_pixel
        metrics["max_pixel_errors"] = None
        metrics["mean_pixel_errors"] = None
        metrics["n_patterns_ok"] = None
        metrics["n_patterns_perfect"] = None

    return metrics
