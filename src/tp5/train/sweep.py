"""
tp5.train.sweep
===============
Expansión de grillas de hiperparámetros (Sweeps) para exploración sistemática.
Genera y ejecuta combinaciones cartesianas a partir de un YAML base y ejes a variar.
"""

import copy
import itertools
import pathlib
from typing import Dict, Any, List, Generator
import yaml

from .config import load_config, _deep_update
from .trainer import Trainer


def set_dotted_key(target: Dict[str, Any], dotted_key: str, value: Any) -> None:
    """Asigna un valor en un diccionario anidado utilizando una clave con puntos."""
    keys = dotted_key.split(".")
    curr = target
    for k in keys[:-1]:
        if k not in curr or not isinstance(curr[k], dict):
            curr[k] = {}
        curr = curr[k]
    curr[keys[-1]] = value


def expand_sweep_grid(sweep_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Expande el producto cartesiano de hiperparámetros definido en la sección 'grid',
    resolviendo variantes y semillas.
    """
    base_path = sweep_cfg.get("base")
    if not base_path:
        raise ValueError("El archivo de sweep debe especificar 'base: <ruta_al_yaml_base>'.")

    base_cfg = load_config(base_path)
    grid = sweep_cfg.get("grid", {})
    seeds = sweep_cfg.get("seeds", [base_cfg.get("seed", 42)])
    variants = sweep_cfg.get("variants", {})

    grid_keys = list(grid.keys())
    grid_values = [grid[k] for k in grid_keys]

    all_configs: List[Dict[str, Any]] = []

    # Producto cartesiano de los ejes
    combos = list(itertools.product(*grid_values)) if grid_values else [()]

    for combo in combos:
        for seed in seeds:
            cfg = copy.deepcopy(base_cfg)
            cfg["seed"] = int(seed)

            for key, val in zip(grid_keys, combo):
                # Resolver si el valor es una referencia a una variante
                if isinstance(val, str) and val in variants:
                    resolved_val = copy.deepcopy(variants[val])
                else:
                    resolved_val = val

                set_dotted_key(cfg, key, resolved_val)

            all_configs.append(cfg)

    return all_configs


def run_sweep(sweep_path: str, verbose: bool = True) -> List[Dict[str, Any]]:
    """Carga una configuración de sweep y ejecuta secuencialmente todas las corridas."""
    path = pathlib.Path(sweep_path)
    if not path.exists():
        raise FileNotFoundError(f"Archivo de sweep no encontrado: {path}")

    with open(path, "r", encoding="utf-8") as f:
        sweep_data = yaml.safe_load(f)

    configs = expand_sweep_grid(sweep_data)
    total = len(configs)
    if verbose:
        print(f"[SWEEP] Iniciando sweep '{path.stem}': {total} corridas en total.")

    results = []
    for idx, cfg in enumerate(configs, 1):
        if verbose:
            print(f"\n[SWEEP] Corriendo {idx}/{total}: {cfg.get('run_name', 'run')} (seed={cfg['seed']})")
        trainer = Trainer(cfg)
        res = trainer.train()
        results.append(res)
        if verbose:
            last = res["last_metric"]
            print(f"  → Estado: {res['summary']['status']}, Pérdida: {last.get('loss'):.4f}, "
                  f"Patrones OK: {last.get('n_patterns_ok')}/{last.get('n_patterns_perfect')}")

    return results
