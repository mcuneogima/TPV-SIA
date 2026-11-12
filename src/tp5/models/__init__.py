"""
tp5.models
==========
Modelos de Autoencoders y registros de construcción.
"""

from .autoencoder import Autoencoder
from .vae import VAE
from .registry import MODELS, build_model

__all__ = ["Autoencoder", "VAE", "MODELS", "build_model"]
