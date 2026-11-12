"""
tp5.nn.optimizers
=================
Algoritmos de optimización para actualización de parámetros.
Soporta SGD, Momentum, RMSprop y Adam, con recorte de gradientes (grad_clip).
"""

from typing import List, Optional, Dict, Any
import numpy as np

from .layers import Parameter


def clip_grad_norm(params: List[Parameter], max_norm: float) -> float:
    """
    Recorta la norma global L2 de los gradientes acumulados en los parámetros.
    Retorna la norma previa al recorte.
    """
    total_sq = sum(np.sum(p.grad ** 2) for p in params)
    total_norm = float(np.sqrt(total_sq))

    if max_norm is not None and max_norm > 0.0 and total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-12)
        for p in params:
            p.grad *= scale

    return total_norm


class Optimizer:
    """Clase base abstracta para optimizadores."""

    def __init__(self, params: List[Parameter], lr: float = 0.001, grad_clip: Optional[float] = None):
        self.params: List[Parameter] = list(params)
        self.lr: float = float(lr)
        self.grad_clip: Optional[float] = float(grad_clip) if grad_clip is not None else None
        self.last_grad_norm: float = 0.0

    def zero_grad(self) -> None:
        """Pone a cero los gradientes de todos los parámetros."""
        for p in self.params:
            p.zero_grad()

    def set_lr(self, lr: float) -> None:
        """Actualiza la tasa de aprendizaje (usado por schedulers)."""
        self.lr = float(lr)

    def pre_step(self) -> float:
        """Aplica grad_clip si está configurado y calcula la norma total."""
        norm = clip_grad_norm(self.params, self.grad_clip)
        self.last_grad_norm = norm
        return norm

    def step(self) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    """
    Descenso de Gradiente Estocástico estándar:
    w <- w - lr * grad
    """

    def step(self) -> None:
        self.pre_step()
        for p in self.params:
            p.data -= self.lr * p.grad


class Momentum(Optimizer):
    """
    SGD con Momentum clásico:
    v <- momentum * v + grad
    w <- w - lr * v
    """

    def __init__(
        self,
        params: List[Parameter],
        lr: float = 0.001,
        momentum: float = 0.9,
        grad_clip: Optional[float] = None,
    ):
        super().__init__(params, lr=lr, grad_clip=grad_clip)
        self.momentum = float(momentum)
        self.v: Dict[int, np.ndarray] = {
            id(p): np.zeros_like(p.data, dtype=np.float64) for p in self.params
        }

    def step(self) -> None:
        self.pre_step()
        for p in self.params:
            pid = id(p)
            v = self.v[pid]
            v[...] = self.momentum * v + p.grad
            p.data -= self.lr * v


class RMSprop(Optimizer):
    """
    RMSprop con promedio móvil del cuadrado del gradiente:
    s <- decay * s + (1 - decay) * grad^2
    w <- w - lr * grad / sqrt(s + eps)
    """

    def __init__(
        self,
        params: List[Parameter],
        lr: float = 0.001,
        decay: float = 0.9,
        eps: float = 1e-8,
        grad_clip: Optional[float] = None,
    ):
        super().__init__(params, lr=lr, grad_clip=grad_clip)
        self.decay = float(decay)
        self.eps = float(eps)
        self.s: Dict[int, np.ndarray] = {
            id(p): np.zeros_like(p.data, dtype=np.float64) for p in self.params
        }

    def step(self) -> None:
        self.pre_step()
        for p in self.params:
            pid = id(p)
            s = self.s[pid]
            s[...] = self.decay * s + (1.0 - self.decay) * (p.grad ** 2)
            p.data -= self.lr * (p.grad / (np.sqrt(s) + self.eps))


class Adam(Optimizer):
    """
    Optimizador Adam (Adaptive Moment Estimation):
    m <- beta1 * m + (1 - beta1) * grad
    v <- beta2 * v + (1 - beta2) * grad^2
    m_hat <- m / (1 - beta1^t)
    v_hat <- v / (1 - beta2^t)
    w <- w - lr * m_hat / (sqrt(v_hat) + eps)
    """

    def __init__(
        self,
        params: List[Parameter],
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        grad_clip: Optional[float] = None,
    ):
        super().__init__(params, lr=lr, grad_clip=grad_clip)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.t: int = 0
        self.m: Dict[int, np.ndarray] = {
            id(p): np.zeros_like(p.data, dtype=np.float64) for p in self.params
        }
        self.v: Dict[int, np.ndarray] = {
            id(p): np.zeros_like(p.data, dtype=np.float64) for p in self.params
        }

    def step(self) -> None:
        self.pre_step()
        self.t += 1
        b1_corr = 1.0 - (self.beta1 ** self.t)
        b2_corr = 1.0 - (self.beta2 ** self.t)

        for p in self.params:
            pid = id(p)
            m = self.m[pid]
            v = self.v[pid]

            # Actualización de momentos
            m[...] = self.beta1 * m + (1.0 - self.beta1) * p.grad
            v[...] = self.beta2 * v + (1.0 - self.beta2) * (p.grad ** 2)

            # Corrección de sesgo (bias correction)
            m_hat = m / b1_corr
            v_hat = v / b2_corr

            # Paso
            p.data -= self.lr * (m_hat / (np.sqrt(v_hat) + self.eps))


OPTIMIZERS = {
    "sgd": SGD,
    "momentum": Momentum,
    "rmsprop": RMSprop,
    "adam": Adam,
}


def build_optimizer(cfg_opt: Dict[str, Any], params: List[Parameter]) -> Optimizer:
    """
    Construye un optimizador a partir del diccionario de configuración.
    
    Ejemplo YAML:
      optimizer:
        name: adam
        lr: 0.001
        beta1: 0.9
        beta2: 0.999
        grad_clip: 1.0
    """
    opt_name = cfg_opt.get("name", "adam").lower().strip()
    if opt_name not in OPTIMIZERS:
        raise ValueError(
            f"Optimizador desconocido: '{opt_name}'. "
            f"Opciones: {list(OPTIMIZERS.keys())}"
        )

    lr = float(cfg_opt.get("lr", 0.001))
    grad_clip = cfg_opt.get("grad_clip", None)

    if opt_name == "sgd":
        return SGD(params, lr=lr, grad_clip=grad_clip)
    elif opt_name == "momentum":
        mom = float(cfg_opt.get("momentum", 0.9))
        return Momentum(params, lr=lr, momentum=mom, grad_clip=grad_clip)
    elif opt_name == "rmsprop":
        decay = float(cfg_opt.get("decay", 0.9))
        eps = float(cfg_opt.get("eps", 1e-8))
        return RMSprop(params, lr=lr, decay=decay, eps=eps, grad_clip=grad_clip)
    elif opt_name == "adam":
        b1 = float(cfg_opt.get("beta1", 0.9))
        b2 = float(cfg_opt.get("beta2", 0.999))
        eps = float(cfg_opt.get("eps", 1e-8))
        return Adam(params, lr=lr, beta1=b1, beta2=b2, eps=eps, grad_clip=grad_clip)
