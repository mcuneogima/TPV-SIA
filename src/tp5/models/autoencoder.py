"""
tp5.models.autoencoder
======================
Arquitectura de Autoencoder estándar (soporta autoencoder básico y denoising).
Consiste en un codificador (encoder) y un decodificador (decoder) modulares.
"""

from typing import List, Dict, Optional, Any
import numpy as np

from ..nn.layers import Layer, Parameter, Sequential


class Autoencoder(Layer):
    """
    Autoencoder clásico feed-forward.
    
    Estructura:
      X -> [Encoder] -> Z (espacio latente) -> [Decoder] -> X_hat
    """

    def __init__(
        self,
        encoder: Sequential,
        decoder: Sequential,
        latent_dim: int,
        input_dim: int,
        output_dim: int,
        name: str = "autoencoder",
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = int(latent_dim)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.name = name

        self._latent_cache: Optional[np.ndarray] = None

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Pasa la entrada por el codificador y retorna la representación latente Z."""
        return self.encoder.forward(x)

    def decode(self, z: np.ndarray) -> np.ndarray:
        """Pasa la representación latente Z por el decodificador y retorna X_hat."""
        return self.decoder.forward(z)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Pasa la entrada por codificador y decodificador."""
        z = self.encode(x)
        self._latent_cache = z
        x_hat = self.decode(z)
        return x_hat

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        """
        Propaga el gradiente de la reconstrucción hacia atrás por el decodificador
        y luego por el codificador.
        """
        grad_z = self.decoder.backward(grad_out)
        grad_in = self.encoder.backward(grad_z)
        return grad_in

    def params(self) -> List[Parameter]:
        """Retorna todos los parámetros entrenables (pesos y sesgos)."""
        return self.encoder.params() + self.decoder.params()

    def zero_grad(self) -> None:
        for p in self.params():
            p.zero_grad()

    def get_weights(self) -> Dict[str, np.ndarray]:
        """Extrae un diccionario {nombre: array} con todos los pesos del modelo."""
        weights = {}
        for p in self.params():
            weights[p.name] = p.data.copy()
        return weights

    def set_weights(self, weights: Dict[str, np.ndarray]) -> None:
        """Carga los pesos desde un diccionario."""
        for p in self.params():
            if p.name in weights:
                p.data[...] = weights[p.name]
            else:
                raise KeyError(f"Parámetro '{p.name}' no encontrado en el diccionario de pesos.")

    def count_params(self) -> int:
        """Calcula la cantidad total de parámetros escalares entrenables."""
        return sum(p.data.size for p in self.params())

    def arch_str(self) -> str:
        """
        Retorna un string descriptivo de la arquitectura de capas,
        por ejemplo '35-24-12-2-12-24-35'.
        """
        dims = [self.input_dim]
        # Extraer dimensiones de las capas densas del encoder
        for layer in self.encoder.layers:
            if hasattr(layer, "out_features"):
                dims.append(layer.out_features)
        # Extraer dimensiones del decoder
        for layer in self.decoder.layers:
            if hasattr(layer, "out_features"):
                dims.append(layer.out_features)
        return "-".join(str(d) for d in dims)
