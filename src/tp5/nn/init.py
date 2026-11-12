"""
tp5.nn.init
===========
Funciones de inicialización de pesos y sesgos.
Garantizan reproducibilidad vía Generator de numpy y precisión float64.
"""

from typing import Tuple, Optional
import numpy as np


def xavier_init(
    shape: Tuple[int, int],
    rng: Optional[np.random.Generator] = None,
    mode: str = "uniform"
) -> np.ndarray:
    """
    Inicialización de Xavier / Glorot.
    Apropiada para activaciones simétricas (tanh, sigmoid, identity).
    
    shape: (in_features, out_features)
    mode: 'uniform' o 'normal'
    """
    if rng is None:
        rng = np.random.default_rng()
    
    fan_in, fan_out = shape
    if mode == "uniform":
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return rng.uniform(-limit, limit, size=shape).astype(np.float64)
    else:
        std = np.sqrt(2.0 / (fan_in + fan_out))
        return rng.normal(0.0, std, size=shape).astype(np.float64)


def he_init(
    shape: Tuple[int, int],
    rng: Optional[np.random.Generator] = None,
    mode: str = "normal"
) -> np.ndarray:
    """
    Inicialización de He / Kaiming.
    Apropiada para activaciones no lineales tipo ReLU y LeakyReLU.
    
    shape: (in_features, out_features)
    mode: 'normal' o 'uniform'
    """
    if rng is None:
        rng = np.random.default_rng()
    
    fan_in, _ = shape
    if mode == "normal":
        std = np.sqrt(2.0 / fan_in)
        return rng.normal(0.0, std, size=shape).astype(np.float64)
    else:
        limit = np.sqrt(6.0 / fan_in)
        return rng.uniform(-limit, limit, size=shape).astype(np.float64)


def normal_small_init(
    shape: Tuple[int, int],
    rng: Optional[np.random.Generator] = None,
    std: float = 0.01
) -> np.ndarray:
    """
    Inicialización normal pequeña N(0, std^2).
    Útil como baseline para contrastar con Xavier/He.
    """
    if rng is None:
        rng = np.random.default_rng()
    return rng.normal(0.0, std, size=shape).astype(np.float64)


def zeros_init(shape: Tuple[int, ...]) -> np.ndarray:
    """Inicialización de ceros para sesgos (biases)."""
    return np.zeros(shape, dtype=np.float64)


INITIALIZERS = {
    "xavier": xavier_init,
    "glorot": xavier_init,
    "he": he_init,
    "kaiming": he_init,
    "normal_small": normal_small_init,
}


def get_initializer(name: str):
    """Retorna la función inicializadora correspondiente."""
    name_clean = name.lower().strip()
    if name_clean not in INITIALIZERS:
        raise ValueError(
            f"Inicializador desconocido: '{name}'. "
            f"Opciones disponibles: {list(INITIALIZERS.keys())}"
        )
    return INITIALIZERS[name_clean]
