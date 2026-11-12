"""
tp5.nn.losses
=============
Funciones de pérdida y cálculo de gradientes.
Convención (según PLAN.md 2.1):
  - Pérdida por muestra = suma sobre píxeles (features).
  - Pérdida del batch   = promedio sobre muestras.
"""

from typing import Optional
import numpy as np


class Loss:
    """Clase base para funciones de pérdida."""

    def __init__(self):
        self._y_pred_cache: Optional[np.ndarray] = None
        self._y_true_cache: Optional[np.ndarray] = None

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        raise NotImplementedError

    def backward(self) -> np.ndarray:
        raise NotImplementedError

    def sample_losses(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return self.forward(y_pred, y_true)


class MSELoss(Loss):
    """
    Error Cuadrático Medio.
    L_i = sum_j (y_pred_ij - y_true_ij)^2
    L = (1 / N) * sum_i L_i
    
    Gradiente: dL / dy_pred = (2 / N) * (y_pred - y_true)
    """

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        pred = np.asarray(y_pred, dtype=np.float64)
        true = np.asarray(y_true, dtype=np.float64)
        if pred.ndim == 1:
            pred = pred.reshape(1, -1)
        if true.ndim == 1:
            true = true.reshape(1, -1)

        self._y_pred_cache = pred
        self._y_true_cache = true

        # Suma sobre features, promedio sobre el batch
        diff = pred - true
        per_sample = np.sum(diff ** 2, axis=1)
        return float(np.mean(per_sample))

    def backward(self) -> np.ndarray:
        if self._y_pred_cache is None or self._y_true_cache is None:
            raise RuntimeError("Llamada a backward() sin forward() previo.")

        n_samples = self._y_pred_cache.shape[0]
        return (2.0 / n_samples) * (self._y_pred_cache - self._y_true_cache)

    def sample_losses(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        pred = np.asarray(y_pred, dtype=np.float64)
        true = np.asarray(y_true, dtype=np.float64)
        if pred.ndim == 1:
            pred = pred.reshape(1, -1)
        if true.ndim == 1:
            true = true.reshape(1, -1)
        return np.sum((pred - true) ** 2, axis=1)


class BCELoss(Loss):
    """
    Entropía Cruzada Binaria (Binary Cross-Entropy).
    L_i = - sum_j [y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred)]
    L = (1 / N) * sum_i L_i
    
    Gradiente: dL / dy_pred = (1 / N) * (y_pred - y_true) / (y_pred * (1 - y_pred))
    Incluye recorte por eps para estabilidad numérica.
    """

    def __init__(self, eps: float = 1e-12):
        super().__init__()
        self.eps = float(eps)

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        pred = np.asarray(y_pred, dtype=np.float64)
        true = np.asarray(y_true, dtype=np.float64)
        if pred.ndim == 1:
            pred = pred.reshape(1, -1)
        if true.ndim == 1:
            true = true.reshape(1, -1)

        # Recorte numérico
        pred_clipped = np.clip(pred, self.eps, 1.0 - self.eps)
        self._y_pred_cache = pred_clipped
        self._y_true_cache = true

        per_sample = -np.sum(
            true * np.log(pred_clipped) + (1.0 - true) * np.log(1.0 - pred_clipped),
            axis=1,
        )
        return float(np.mean(per_sample))

    def backward(self) -> np.ndarray:
        if self._y_pred_cache is None or self._y_true_cache is None:
            raise RuntimeError("Llamada a backward() sin forward() previo.")

        pred = self._y_pred_cache
        true = self._y_true_cache
        n_samples = pred.shape[0]

        denom = pred * (1.0 - pred)
        denom = np.maximum(denom, self.eps)
        return (1.0 / n_samples) * ((pred - true) / denom)

    def sample_losses(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        pred = np.clip(np.asarray(y_pred, dtype=np.float64), self.eps, 1.0 - self.eps)
        true = np.asarray(y_true, dtype=np.float64)
        if pred.ndim == 1:
            pred = pred.reshape(1, -1)
        if true.ndim == 1:
            true = true.reshape(1, -1)
        return -np.sum(
            true * np.log(pred) + (1.0 - true) * np.log(1.0 - pred),
            axis=1,
        )


LOSSES = {
    "mse": MSELoss,
    "bce": BCELoss,
}


def get_loss(name: str, **kwargs) -> Loss:
    """Instancia y retorna la función de pérdida solicitada por nombre."""
    name_clean = name.lower().strip()
    if name_clean not in LOSSES:
        raise ValueError(
            f"Función de pérdida desconocida: '{name}'. "
            f"Opciones disponibles: {list(LOSSES.keys())}"
        )
    return LOSSES[name_clean](**kwargs)
