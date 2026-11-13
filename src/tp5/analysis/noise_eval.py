"""Reproducible robustness evaluation for binary font autoencoders."""

from typing import Any, Dict, Iterable, List

import numpy as np

from ..data.noise import corrupt


def reconstruction_metrics(X_clean: np.ndarray, X_pred: np.ndarray) -> Dict[str, float]:
    """Measure pixel and pattern reconstruction quality against clean targets."""
    clean_bits = X_clean >= 0.5
    predicted_bits = X_pred >= 0.5
    errors = np.count_nonzero(clean_bits != predicted_bits, axis=1)
    return {
        "mean_pixel_errors": float(np.mean(errors)),
        "max_pixel_errors": float(np.max(errors)),
        "frac_perfect": float(np.mean(errors == 0)),
        "frac_ok": float(np.mean(errors <= 1)),
    }


def reconstruct(model: Any, X: np.ndarray) -> np.ndarray:
    """Use a deterministic latent mean when evaluating a variational model."""
    if hasattr(model, "encode_mean"):
        return model.decode(model.encode_mean(X))
    return model.forward(X)


def evaluate_noise_levels(
    model: Any | None,
    X_clean: np.ndarray,
    noise_kind: str,
    levels: Iterable[float],
    n_trials: int,
    seed: int,
) -> List[Dict[str, float]]:
    """Evaluate denoising quality over fixed noise levels and independent trials."""
    if n_trials < 1:
        raise ValueError("n_trials debe ser al menos 1.")

    rng = np.random.default_rng(seed)
    results = []
    for level in levels:
        for trial in range(1, n_trials + 1):
            X_noisy = corrupt(X_clean, noise_kind, float(level), rng)
            X_pred = X_noisy if model is None else reconstruct(model, X_noisy)
            results.append({
                "kind": noise_kind,
                "level": float(level),
                "trial": trial,
                **reconstruction_metrics(X_clean, X_pred),
            })
    return results
