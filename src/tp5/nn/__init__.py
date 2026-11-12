"""
tp5.nn
======
Módulo de redes neuronales implementado puramente sobre NumPy.
"""

from .init import xavier_init, he_init, normal_small_init, zeros_init, get_initializer
from .activations import (
    Activation,
    Identity,
    Sigmoid,
    Tanh,
    ReLU,
    LeakyReLU,
    get_activation,
    ACTIVATIONS,
)
from .layers import Parameter, Layer, Dense, Sequential
from .losses import Loss, MSELoss, BCELoss, get_loss, LOSSES
from .schedules import (
    LRScheduler,
    ConstantSchedule,
    ExpDecaySchedule,
    StepSchedule,
    build_schedule,
    SCHEDULES,
)
from .optimizers import (
    Optimizer,
    SGD,
    Momentum,
    RMSprop,
    Adam,
    clip_grad_norm,
    build_optimizer,
    OPTIMIZERS,
)

__all__ = [
    "xavier_init",
    "he_init",
    "normal_small_init",
    "zeros_init",
    "get_initializer",
    "Activation",
    "Identity",
    "Sigmoid",
    "Tanh",
    "ReLU",
    "LeakyReLU",
    "get_activation",
    "ACTIVATIONS",
    "Parameter",
    "Layer",
    "Dense",
    "Sequential",
    "Loss",
    "MSELoss",
    "BCELoss",
    "get_loss",
    "LOSSES",
    "LRScheduler",
    "ConstantSchedule",
    "ExpDecaySchedule",
    "StepSchedule",
    "build_schedule",
    "SCHEDULES",
    "Optimizer",
    "SGD",
    "Momentum",
    "RMSprop",
    "Adam",
    "clip_grad_norm",
    "build_optimizer",
    "OPTIMIZERS",
]
