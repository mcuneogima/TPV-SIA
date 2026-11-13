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

    mode = sweep_cfg.get("mode", "cartesian")
    if mode == "cartesian":
        combinations = [
            (dict(zip(grid_keys, combo)), seed)
            for combo in (itertools.product(*grid_values) if grid_values else [()])
            for seed in seeds
        ]
    elif mode == "ofat":
        combinations = [
            ({key: value}, seed)
            for key, values in grid.items()
            for value in values
            for seed in seeds
        ]
    else:
        raise ValueError(f"Modo de sweep desconocido: '{mode}'. Opciones: cartesian, ofat")

    seen = set()
    for overrides, seed in combinations:
        cfg = copy.deepcopy(base_cfg)
        cfg["seed"] = int(seed)

        for key, val in overrides.items():
            resolved = copy.deepcopy(variants[val]) if isinstance(val, str) and val in variants else val
            if isinstance(resolved, dict) and "layers" in resolved and key.endswith(".layers"):
                resolved = resolved["layers"]
            set_dotted_key(cfg, key, resolved)

        fingerprint = yaml.safe_dump(cfg, sort_keys=True)
        if fingerprint not in seen:
            all_configs.append(cfg)
            seen.add(fingerprint)

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
