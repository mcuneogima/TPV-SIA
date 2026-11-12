"""
tp5.nn.schedules
================
Esquemas de decaimiento del learning rate (schedules).
Permiten ajustar dinámicamente la tasa de aprendizaje durante el entrenamiento.
"""

from typing import Dict, Any


class LRScheduler:
    """Clase base para esquemas de tasa de aprendizaje."""

    def __init__(self, initial_lr: float):
        self.initial_lr = float(initial_lr)

    def get_lr(self, epoch: int) -> float:
        raise NotImplementedError

    def __call__(self, epoch: int) -> float:
        return self.get_lr(epoch)


class ConstantSchedule(LRScheduler):
    """Tasa de aprendizaje constante."""

    def get_lr(self, epoch: int) -> float:
        return self.initial_lr


class ExpDecaySchedule(LRScheduler):
    """
    Decaimiento exponencial: lr = initial_lr * (gamma ** epoch).
    """

    def __init__(self, initial_lr: float, gamma: float = 0.9995):
        super().__init__(initial_lr)
        self.gamma = float(gamma)

    def get_lr(self, epoch: int) -> float:
        return float(self.initial_lr * (self.gamma ** epoch))


class StepSchedule(LRScheduler):
    """
    Decaimiento por pasos: lr = initial_lr * (gamma ** (epoch // step_size)).
    """

    def __init__(self, initial_lr: float, step_size: int = 1000, gamma: float = 0.5):
        super().__init__(initial_lr)
        self.step_size = max(1, int(step_size))
        self.gamma = float(gamma)

    def get_lr(self, epoch: int) -> float:
        steps = epoch // self.step_size
        return float(self.initial_lr * (self.gamma ** steps))


SCHEDULES = {
    "constant": ConstantSchedule,
    "exp_decay": ExpDecaySchedule,
    "step": StepSchedule,
}


def build_schedule(cfg_opt: Dict[str, Any]) -> LRScheduler:
    """
    Construye un scheduler a partir del bloque de configuración del optimizador.
    
    Ejemplo YAML:
      optimizer:
        lr: 0.001
        schedule: {name: exp_decay, gamma: 0.9995}
    """
    base_lr = float(cfg_opt.get("lr", 0.001))
    sched_info = cfg_opt.get("schedule", None)

    if sched_info is None or not sched_info:
        return ConstantSchedule(base_lr)

    if isinstance(sched_info, str):
        sched_name = sched_info
        sched_params = {}
    elif isinstance(sched_info, dict):
        sched_name = sched_info.get("name", "constant")
        sched_params = {k: v for k, v in sched_info.items() if k != "name"}
    else:
        raise TypeError(f"Formato de schedule inválido: {type(sched_info)}")

    name_clean = sched_name.lower().strip()
    if name_clean not in SCHEDULES:
        raise ValueError(
            f"Schedule desconocido: '{sched_name}'. "
            f"Opciones: {list(SCHEDULES.keys())}"
        )

    return SCHEDULES[name_clean](initial_lr=base_lr, **sched_params)
