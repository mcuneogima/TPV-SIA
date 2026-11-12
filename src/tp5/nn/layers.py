"""
tp5.nn.layers
=============
Definición de capas densas (fully connected) y estructuras de parámetros.
Garantiza cálculo matricial en float64 y propagación exacta de gradientes.
"""

from typing import List, Optional, Union
import numpy as np

from .init import get_initializer, zeros_init


class Parameter:
    """
    Contenedor de parámetros entrenables (pesos y sesgos).
    Mantiene sincronizados los valores numéricos y sus gradientes acumulados.
    """

    def __init__(self, data: np.ndarray, name: str = ""):
        self.data: np.ndarray = np.asarray(data, dtype=np.float64)
        self.grad: np.ndarray = np.zeros_like(self.data, dtype=np.float64)
        self.name: str = name

    def zero_grad(self) -> None:
        """Reinicia el gradiente a ceros."""
        self.grad.fill(0.0)

    @property
    def shape(self):
        return self.data.shape

    def __repr__(self) -> str:
        return f"Parameter(name='{self.name}', shape={self.shape}, dtype={self.data.dtype})"


class Layer:
    """Clase base abstracta para capas de red neuronal."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def params(self) -> List[Parameter]:
        return []

    def zero_grad(self) -> None:
        for p in self.params():
            p.zero_grad()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class Dense(Layer):
    """
    Capa totalmente conectada (Dense / Linear):
    y = x @ W + b
    
    Dimensiones:
      x:        (batch, in_features)
      W:        (in_features, out_features)
      b:        (1, out_features)
      y:        (batch, out_features)
      grad_out: (batch, out_features)
      dW:       (in_features, out_features) = x.T @ grad_out
      db:       (1, out_features) = sum(grad_out, axis=0)
      grad_in:  (batch, in_features) = grad_out @ W.T
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        init: str = "xavier",
        rng: Optional[np.random.Generator] = None,
        name: Optional[str] = None,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.name = name or f"dense_{in_features}x{out_features}"

        init_fn = get_initializer(init)
        w_data = init_fn((self.in_features, self.out_features), rng=rng)
        b_data = zeros_init((1, self.out_features))

        self.W = Parameter(w_data, name=f"{self.name}.W")
        self.b = Parameter(b_data, name=f"{self.name}.b")

        self._x_cache: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        x_mat = np.asarray(x, dtype=np.float64)
        if x_mat.ndim == 1:
            x_mat = x_mat.reshape(1, -1)
        self._x_cache = x_mat
        return x_mat @ self.W.data + self.b.data

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        if self._x_cache is None:
            raise RuntimeError("Llamada a backward() sin forward() previo.")

        grad_out_mat = np.asarray(grad_out, dtype=np.float64)
        if grad_out_mat.ndim == 1:
            grad_out_mat = grad_out_mat.reshape(1, -1)

        # Gradientes de los parámetros
        self.W.grad = self._x_cache.T @ grad_out_mat
        self.b.grad = np.sum(grad_out_mat, axis=0, keepdims=True)

        # Gradiente con respecto a la entrada
        grad_in = grad_out_mat @ self.W.data.T
        return grad_in

    def params(self) -> List[Parameter]:
        return [self.W, self.b]


class Sequential(Layer):
    """Contenedor secuencial de capas."""

    def __init__(self, layers: Optional[List[Layer]] = None):
        super().__init__()
        self.layers: List[Layer] = list(layers) if layers else []

    def add(self, layer: Layer) -> None:
        self.layers.append(layer)

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        grad = grad_out
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def params(self) -> List[Parameter]:
        all_p = []
        for layer in self.layers:
            all_p.extend(layer.params())
        return all_p

    def __iter__(self):
        return iter(self.layers)

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, idx: int) -> Layer:
        return self.layers[idx]
