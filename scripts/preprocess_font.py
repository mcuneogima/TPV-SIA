"""
preprocess_font.py
==================
Script standalone para parsear data/raw/font.h y generar data/processed/font.npz.

Uso:
  python scripts/preprocess_font.py [--font-path data/raw/font.h] [--out data/processed/font.npz]
"""

import argparse
import pathlib
import sys

# Ajustar path para importar desde src/
repo_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from tp5.data.fontparser import parse_font, save_font, render_ascii


def main():
    parser = argparse.ArgumentParser(description="Preprocesa font.h → font.npz")
    parser.add_argument("--font-path", default="data/raw/font.h",
                        help="Ruta al archivo font.h (default: data/raw/font.h)")
    parser.add_argument("--out", default="data/processed/font.npz",
                        help="Ruta de salida del .npz (default: data/processed/font.npz)")
    args = parser.parse_args()

    font_path = repo_root / args.font_path
    out_path  = repo_root / args.out

    if not font_path.exists():
        print(f"[ERROR] No se encontró el archivo: {font_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Parseando {font_path} ...")
    data = parse_font(str(font_path))

    print("\n=== Render ASCII de los 32 caracteres ===\n")
    print(render_ascii(data["X"], data["labels"], data["shape"]))

    save_font(data, str(out_path))

    # Validaciones finales
    n, f = data["X"].shape
    print(f"\n[OK] Validaciones:")
    print(f"     Patrones  : {n} (esperado: 32)")
    print(f"     Features  : {f} (esperado: 35)")
    print(f"     Valores   : {{0, 1}} ✓")
    print(f"     Archivo   : {out_path}")


if __name__ == "__main__":
    main()
