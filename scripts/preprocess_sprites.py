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

import yaml

repo_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from tp5.data.sprites import load_sprites, save_sprites


def main():
    parser = argparse.ArgumentParser(description="Preprocesa tileset → sprites.npz")
    parser.add_argument("--config", default=None, help="YAML con opciones del dataset (sección data).")
    parser.add_argument("--tsx")
    parser.add_argument("--png")
    parser.add_argument("--channels", choices=["rgb", "gray"])
    parser.add_argument("--val-split", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--out-npz")
    parser.add_argument("--out-meta")
    parser.add_argument("--dedup", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()

    data_cfg = {}
    if args.config:
        config_path = repo_root / args.config
        if not config_path.exists():
            parser.error(f"No existe el archivo de configuración: {config_path}")
        with config_path.open("r", encoding="utf-8") as file:
            data_cfg = yaml.safe_load(file).get("data", {})

    def option(value, key, default):
        return value if value is not None else data_cfg.get(key, default)

    tsx = option(args.tsx, "tsx", "data/raw/sprites/tileset_colored.tsx")
    png = option(args.png, "png", "data/raw/sprites/tileset_legacy.png")
    channels = option(args.channels, "channels", "rgb")
    val_split = option(args.val_split, "val_split", 0.2)
    seed = option(args.seed, "seed", 42)
    dedup = option(args.dedup, "dedup", True)
    out_npz_arg = option(args.out_npz, "path", "data/processed/sprites.npz")
    default_meta = (
        "data/processed/sprites_meta.json"
        if args.config is None and args.out_npz is None
        else str(pathlib.Path(out_npz_arg).with_name(f"{pathlib.Path(out_npz_arg).stem}_meta.json"))
    )
    out_meta_arg = option(args.out_meta, "meta_path", default_meta)
    regions = data_cfg.get("regions")

    tsx_path = repo_root / tsx
    png_path = repo_root / png
    out_npz = repo_root / out_npz_arg
    out_meta = repo_root / out_meta_arg

    for p in [tsx_path, png_path]:
        if not p.exists():
            print(f"[ERROR] No se encontró: {p}", file=sys.stderr)
            sys.exit(1)

    print(f"[INFO] Cargando tileset desde {png_path.name} ...")
    data = load_sprites(
        str(tsx_path), str(png_path),
        channels=channels,
        regions=regions,
        dedup=dedup,
        val_split=val_split,
        seed=seed,
    )

    n_train = len(data["X_train"])
    n_val   = len(data["X_val"]) if data["X_val"] is not None else 0
    shape   = data["shape"]
    print(f"[OK] Train: {n_train} tiles | Val: {n_val} tiles | Shape: {shape}")

    save_sprites(data, str(out_npz), str(out_meta))


if __name__ == "__main__":
    main()
