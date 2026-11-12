"""
noise.py
========
Funciones de corrupción de datos para el Denoising Autoencoder.

Todas las funciones son puras: reciben X y un rng explícito,
no modifican el array original, y son reproducibles.
"""

import numpy as np


def corrupt(X: np.ndarray, kind: str, level: float, rng: np.random.Generator) -> np.ndarray:
    """
    Corrompe el array X según el tipo y nivel de ruido.

    Parámetros
    ----------
    X     : ndarray de shape (n, features) o (features,)
            Para 'flip' y 'mask' se esperan valores en {0, 1}.
            Para 'gaussian' se esperan valores continuos en [0, 1].
    kind  : 'flip' | 'mask' | 'gaussian'
    level : probabilidad de corrupción (flip/mask) o desviación estándar (gaussian)
    rng   : np.random.Generator pasado explícitamente (para reproducibilidad)

    Retorna
    -------
    X_corrupted : ndarray del mismo dtype y shape que X
    """
    X = np.array(X, copy=True)

    if kind == "flip":
        # Cada bit se invierte con probabilidad `level`
        mask = rng.random(X.shape) < level
        X = np.where(mask, 1 - X, X).astype(X.dtype)

    elif kind == "mask":
        # Cada pixel se fuerza a 0 con probabilidad `level`
        mask = rng.random(X.shape) < level
        X = np.where(mask, 0, X).astype(X.dtype)

    elif kind == "gaussian":
        # Ruido aditivo gaussiano (para imágenes continuas), clipeado a [0, 1]
        noise = rng.normal(0.0, level, size=X.shape)
        X = np.clip(X.astype(np.float64) + noise, 0.0, 1.0)

    else:
        raise ValueError(f"Tipo de ruido desconocido: '{kind}'. Opciones: flip, mask, gaussian")

    return X


def corrupt_batch(X: np.ndarray, kind: str, level: float, rng: np.random.Generator,
                  n_trials: int = 1) -> np.ndarray:
    """
    Genera n_trials versiones ruidosas de cada patrón en X.

    Retorna array de shape (n_trials, n, features).
    Útil para evaluar robustez promediando sobre muchas corrupciones.
    """
    trials = [corrupt(X, kind, level, rng) for _ in range(n_trials)]
    return np.stack(trials, axis=0)
