"""
scripts/sweep.py
================
Script para ejecutar barridos sistemáticos de hiperparámetros (Sweeps).

Uso:
  python scripts/sweep.py --sweep configs/sweeps/a2_arquitecturas.yaml
"""

import argparse
import pathlib
import sys

# Configurar path del repositorio
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tp5.train.sweep import run_sweep


def main():
    parser = argparse.ArgumentParser(description="Ejecuta un sweep de hiperparámetros.")
    parser.add_argument(
        "--sweep",
        type=str,
        required=True,
        help="Ruta al archivo YAML de sweep.",
    )
    args = parser.parse_args()

    print("=" * 70)
    print(f"  TP5 Deep Learning — Barrido de Hiperparámetros")
    print(f"  Sweep YAML : {args.sweep}")
    print("=" * 70)

    results = run_sweep(args.sweep, verbose=True)

    print("\n" + "=" * 70)
    print(f"  Sweep completado: {len(results)} corridas ejecutadas.")
    print(f"  Resultados consolidados en: results/summary/runs.csv")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
