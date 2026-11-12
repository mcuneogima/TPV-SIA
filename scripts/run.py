"""
scripts/run.py
==============
Script principal para ejecutar una corrida individual de entrenamiento a partir de un YAML.

Uso:
  python scripts/run.py --config configs/ae_basic.yaml
"""

import argparse
import pathlib
import sys
import time

# Configurar path del repositorio
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tp5.train.config import load_config
from tp5.train.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Ejecuta una corrida de entrenamiento de TP5.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Ruta al archivo YAML de configuración.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Semilla aleatoria (sobreescribe la del YAML si se especifica).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed

    print("=" * 70)
    print(f"  TP5 Deep Learning — Corrida: {cfg.get('run_name', 'run')}")
    print(f"  Configuración : {args.config}")
    print(f"  Modelo        : {cfg['model']['type']}")
    print(f"  Dataset       : {cfg['data']['dataset']}")
    print(f"  Semilla       : {cfg['seed']}")
    print("=" * 70)

    trainer = Trainer(cfg)
    print(f"  Arquitectura  : {trainer.model.arch_str()}")
    print(f"  Parámetros    : {trainer.model.count_params():,}")
    print(f"  Optimizador   : {cfg['optimizer']['name']} (lr={cfg['optimizer']['lr']})")
    print(f"  Pérdida       : {cfg['loss']['recon']}")
    print("-" * 70)

    res = trainer.train()

    print("\n" + "=" * 70)
    print("  Entrenamiento finalizado exitosamente.")
    print(f"  Run ID        : {res['run_id']}")
    print(f"  Directorio    : {res['run_dir']}")
    last = res["last_metric"]
    print(f"  Épocas totales: {last.get('epoch')}")
    print(f"  Pérdida final : {last.get('loss'):.6f}")
    if last.get("n_patterns_ok") is not None:
        print(f"  Patrones OK   : {last.get('n_patterns_ok')}/32 (<= 1 px error)")
        print(f"  Perfectos     : {last.get('n_patterns_perfect')}/32 (0 px error)")
        print(f"  Máx px error  : {last.get('max_pixel_errors')}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
