"""
tp5.nn.activations
==================
Funciones de activación y sus derivadas para retropropagación.
Todas las operaciones trabajan sobre matrices (batch, features) en float64.
"""

from typing import List, Any
import numpy as np


class Activation:
    """Clase base para funciones de activación."""

    def __init__(self):
        self._cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def params(self) -> List[Any]:
        return []

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class Identity(Activation):
    """Activación identidad f(x) = x."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._cache = x
        return x

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        return grad_out.copy()


class Sigmoid(Activation):
    """
    Sigmoide logística: f(x) = 1 / (1 + exp(-x)).
    Incluye recorte (clipping) para prevenir desbordes numéricos (overflow).
    Derivada: f'(x) = f(x) * (1 - f(x)).
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        x_clipped = np.clip(x, -500.0, 500.0)
        out = 1.0 / (1.0 + np.exp(-x_clipped))
        self._cache = out
        return out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        out = self._cache
        return grad_out * out * (1.0 - out)


class Tanh(Activation):
    """
    Tangente hiperbólica: f(x) = tanh(x).
    Derivada: f'(x) = 1 - tanh(x)^2.
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = np.tanh(x)
        self._cache = out
        return out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        out = self._cache
        return grad_out * (1.0 - out ** 2)


class ReLU(Activation):
    """
    Rectified Linear Unit: f(x) = max(0, x).
    Derivada: 1 si x > 0, 0 en caso contrario.
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._cache = x
        return np.maximum(0.0, x)

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        x = self._cache
        return grad_out * (x > 0.0).astype(np.float64)


class LeakyReLU(Activation):
    """
    Leaky ReLU: f(x) = x si x > 0 else alpha * x.
    Derivada: 1 si x > 0 else alpha.
    """

    def __init__(self, alpha: float = 0.01):
        super().__init__()
        self.alpha = float(alpha)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._cache = x
        return np.where(x > 0.0, x, self.alpha * x)

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        x = self._cache
        dx = np.where(x > 0.0, 1.0, self.alpha)
        return grad_out * dx


ACTIVATIONS = {
    "identity": Identity,
    "linear": Identity,
    "sigmoid": Sigmoid,
    "tanh": Tanh,
    "relu": ReLU,
    "leaky_relu": LeakyReLU,
}


def get_activation(name: str, **kwargs) -> Activation:
    """Instancia y retorna la activación solicitada por nombre."""
    name_clean = name.lower().strip()
    if name_clean not in ACTIVATIONS:
        raise ValueError(
            f"Activación desconocida: '{name}'. "
            f"Opciones disponibles: {list(ACTIVATIONS.keys())}"
        )
    return ACTIVATIONS[name_clean](**kwargs)
