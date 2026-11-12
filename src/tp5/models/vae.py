"""
tp5.models.vae
==============
Autoencoder Variacional (VAE) con truco de reparametrización.
Cálculo analítico del gradiente de la divergencia KL y de reconstrucción.
"""

from typing import List, Dict, Optional, Tuple, Any
import numpy as np

from ..nn.layers import Layer, Parameter, Dense, Sequential


class VAE(Layer):
    """
    Autoencoder Variacional (VAE).
    
    Estructura:
      X -> [Encoder Trunk] -> h
      h -> [mu_head] -> mu
      h -> [logvar_head] -> log(sigma^2)
      z = mu + exp(0.5 * logvar) * eps   (eps ~ N(0, I))
      z -> [Decoder] -> X_hat
    
    Pérdida:
      L = L_recon + beta * KL
      KL = -0.5 * (1 / N) * sum(1 + logvar - mu^2 - exp(logvar))
    """

    def __init__(
        self,
        encoder_trunk: Sequential,
        mu_head: Dense,
        logvar_head: Dense,
        decoder: Sequential,
        latent_dim: int,
        input_dim: int,
        output_dim: int,
        beta: float = 1.0,
        name: str = "vae",
    ):
        super().__init__()
        self.encoder_trunk = encoder_trunk
        self.mu_head = mu_head
        self.logvar_head = logvar_head
        self.decoder = decoder
        self.latent_dim = int(latent_dim)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.beta = float(beta)
        self.name = name

        # Caches para retropropagación
        self._h_cache: Optional[np.ndarray] = None
        self._mu_cache: Optional[np.ndarray] = None
        self._logvar_cache: Optional[np.ndarray] = None
        self._std_cache: Optional[np.ndarray] = None
        self._eps_cache: Optional[np.ndarray] = None
        self._z_cache: Optional[np.ndarray] = None

    def encode(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Pasa x por el tronco y retorna (mu, logvar)."""
        h = self.encoder_trunk.forward(x)
        mu = self.mu_head.forward(h)
        logvar = self.logvar_head.forward(h)
        # Acotar logvar para evitar inestabilidad en exp()
        logvar = np.clip(logvar, -20.0, 20.0)
        return mu, logvar

    def encode_mean(self, x: np.ndarray) -> np.ndarray:
        """Modo determinista: retorna la media mu (útil para análisis latente)."""
        mu, _ = self.encode(x)
        return mu

    def reparameterize(
        self,
        mu: np.ndarray,
        logvar: np.ndarray,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Aplica el truco de reparametrización:
        z = mu + std * eps,  donde std = exp(0.5 * logvar) y eps ~ N(0, I).
        """
        if rng is None:
            rng = np.random.default_rng()

        std = np.exp(0.5 * logvar)
        eps = rng.normal(0.0, 1.0, size=mu.shape).astype(np.float64)
        z = mu + std * eps
        return z, std, eps

    def decode(self, z: np.ndarray) -> np.ndarray:
        """Pasa el vector latente z por el decodificador."""
        return self.decoder.forward(z)

    def forward(
        self,
        x: np.ndarray,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Paso hacia adelante completo del VAE:
        x -> h -> (mu, logvar) -> z -> x_hat.
        """
        h = self.encoder_trunk.forward(x)
        mu = self.mu_head.forward(h)
        logvar = self.logvar_head.forward(h)
        logvar = np.clip(logvar, -20.0, 20.0)

        z, std, eps = self.reparameterize(mu, logvar, rng=rng)
        x_hat = self.decode(z)

        # Guardar en caché para backward
        self._h_cache = h
        self._mu_cache = mu
        self._logvar_cache = logvar
        self._std_cache = std
        self._eps_cache = eps
        self._z_cache = z

        return x_hat

    def kl_loss(self) -> float:
        """
        Calcula la divergencia KL analítica promedio del batch actual:
        KL = -0.5 * (1 / N) * sum(1 + logvar - mu^2 - exp(logvar))
        """
        if self._mu_cache is None or self._logvar_cache is None:
            return 0.0

        mu = self._mu_cache
        logvar = self._logvar_cache
        n_samples = mu.shape[0]

        kl_per_sample = -0.5 * np.sum(1.0 + logvar - (mu ** 2) - np.exp(logvar), axis=1)
        return float(np.mean(kl_per_sample))

    def sample_kl(self) -> np.ndarray:
        """Retorna la divergencia KL por cada muestra individual en el batch."""
        if self._mu_cache is None or self._logvar_cache is None:
            raise RuntimeError("No hay caché de forward() para calcular sample_kl.")
        return -0.5 * np.sum(
            1.0 + self._logvar_cache - (self._mu_cache ** 2) - np.exp(self._logvar_cache),
            axis=1,
        )

    def backward(self, grad_recon: np.ndarray) -> np.ndarray:
        """
        Retropropagación completa del VAE:
          1. dL_recon/dz vía decoder.backward(grad_recon)
          2. dL/dmu = dL_recon/dz + (beta / N) * mu
          3. dL/dlogvar = dL_recon/dz * eps * 0.5 * std + (beta / (2 * N)) * (exp(logvar) - 1)
          4. grad_h = mu_head.backward(dL/dmu) + logvar_head.backward(dL/dlogvar)
          5. grad_in = encoder_trunk.backward(grad_h)
        """
        if self._mu_cache is None:
            raise RuntimeError("Llamada a backward() sin forward() previo en VAE.")

        mu = self._mu_cache
        logvar = self._logvar_cache
        std = self._std_cache
        eps = self._eps_cache
        n_samples = mu.shape[0]

        # 1. Gradiente del decoder hacia el espacio latente z
        grad_z = self.decoder.backward(grad_recon)

        # 2. Gradiente respecto a mu
        grad_mu = grad_z + (self.beta / n_samples) * mu

        # 3. Gradiente respecto a logvar
        grad_logvar = (
            grad_z * eps * 0.5 * std
            + (self.beta / (2.0 * n_samples)) * (np.exp(logvar) - 1.0)
        )

        # 4. Gradientes a través de las cabezas mu y logvar
        grad_h_mu = self.mu_head.backward(grad_mu)
        grad_h_logvar = self.logvar_head.backward(grad_logvar)
        grad_h = grad_h_mu + grad_h_logvar

        # 5. Gradiente hacia el tronco del encoder
        grad_in = self.encoder_trunk.backward(grad_h)
        return grad_in

    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """
        Genera nuevas muestras a partir de la distribución a priori:
        z ~ N(0, I) -> Decoder -> X_hat.
        """
        if rng is None:
            rng = np.random.default_rng()
        z = rng.normal(0.0, 1.0, size=(n, self.latent_dim)).astype(np.float64)
        return self.decode(z)

    def params(self) -> List[Parameter]:
        """Retorna todos los parámetros de tronco, cabezas y decodificador."""
        return (
            self.encoder_trunk.params()
            + self.mu_head.params()
            + self.logvar_head.params()
            + self.decoder.params()
        )

    def zero_grad(self) -> None:
        for p in self.params():
            p.zero_grad()

    def get_weights(self) -> Dict[str, np.ndarray]:
        weights = {}
        for p in self.params():
            weights[p.name] = p.data.copy()
        return weights

    def set_weights(self, weights: Dict[str, np.ndarray]) -> None:
        for p in self.params():
            if p.name in weights:
                p.data[...] = weights[p.name]
            else:
                raise KeyError(f"Parámetro '{p.name}' no encontrado en el diccionario de pesos.")

    def count_params(self) -> int:
        return sum(p.data.size for p in self.params())

    def arch_str(self) -> str:
        """Resumen de arquitectura, ej: 768-256-128-[2x16]-128-256-768."""
        trunk_dims = [self.input_dim]
        for layer in self.encoder_trunk.layers:
            if hasattr(layer, "out_features"):
                trunk_dims.append(layer.out_features)
        
        dec_dims = [self.latent_dim]
        for layer in self.decoder.layers:
            if hasattr(layer, "out_features"):
                dec_dims.append(layer.out_features)

        return (
            "-".join(str(d) for d in trunk_dims)
            + f"-[2x{self.latent_dim}]-"
            + "-".join(str(d) for d in dec_dims[1:])
        )
