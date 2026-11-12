"""
preprocess_sprites.py
=====================
Script standalone para procesar el tileset de sprites y generar data/processed/sprites.npz.

Uso:
  python scripts/preprocess_sprites.py [--config configs/dataset_sprites.yaml]

Si no se pasa config, usa valores por defecto (todos los tiles no-fondo).
"""

import argparse
import pathlib
import sys

repo_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from tp5.data.sprites import load_sprites, save_sprites


def main():
    parser = argparse.ArgumentParser(description="Preprocesa tileset → sprites.npz")
    parser.add_argument("--tsx",      default="data/raw/sprites/tileset_colored.tsx")
    parser.add_argument("--png",      default="data/raw/sprites/tileset_legacy.png")
    parser.add_argument("--channels", default="rgb", choices=["rgb", "gray"])
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--out-npz",  default="data/processed/sprites.npz")
    parser.add_argument("--out-meta", default="data/processed/sprites_meta.json")
    parser.add_argument("--dedup",    action="store_true", default=True)
    args = parser.parse_args()

    tsx_path = repo_root / args.tsx
    png_path = repo_root / args.png
    out_npz  = repo_root / args.out_npz
    out_meta = repo_root / args.out_meta

    for p in [tsx_path, png_path]:
        if not p.exists():
            print(f"[ERROR] No se encontró: {p}", file=sys.stderr)
            sys.exit(1)

    print(f"[INFO] Cargando tileset desde {png_path.name} ...")
    data = load_sprites(
        str(tsx_path), str(png_path),
        channels=args.channels,
        dedup=args.dedup,
        val_split=args.val_split,
        seed=args.seed,
    )

    n_train = len(data["X_train"])
    n_val   = len(data["X_val"]) if data["X_val"] is not None else 0
    shape   = data["shape"]
    print(f"[OK] Train: {n_train} tiles | Val: {n_val} tiles | Shape: {shape}")

    save_sprites(data, str(out_npz), str(out_meta))


if __name__ == "__main__":
    main()
