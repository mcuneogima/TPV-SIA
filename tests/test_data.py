"""
tests/test_data.py
==================
Test visual del parseo de font.h y del tileset de sprites.

Salidas en tests/output/:
  font_render.txt   — render ASCII de los 32 caracteres con metadatos
  sprites_grid.png  — grilla visual de una muestra de tiles del dataset
  sprites_meta.txt  — resumen del dataset de sprites

Uso:
  python tests/test_data.py
  python tests/test_data.py --no-sprites   (si no tiene Pillow/matplotlib)
"""

import argparse
import pathlib
import sys

# Ajustar path
REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

OUT_DIR = REPO / "tests" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PASS = "✓"
FAIL = "✗"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ok(msg):
    print(f"  {PASS}  {msg}")

def _fail(msg):
    print(f"  {FAIL}  {msg}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Test 1: Font
# ---------------------------------------------------------------------------

def test_font():
    print("\n── TEST FONT ─────────────────────────────────────────────────")

    from tp5.data.fontparser import parse_font, render_ascii

    font_h = REPO / "data" / "raw" / "font.h"
    if not font_h.exists():
        _fail(f"No se encontró {font_h}")

    data = parse_font(str(font_h))
    X      = data["X"]
    labels = data["labels"]
    shape  = data["shape"]

    # Validaciones
    assert X.shape == (32, 35), f"Shape incorrecto: {X.shape}"
    _ok(f"32 patrones × 35 bits  — shape: {X.shape}")

    import numpy as np
    assert X.dtype in (np.uint8,), f"dtype inesperado: {X.dtype}"
    assert set(X.flatten().tolist()) <= {0, 1}, "Valores fuera de {0, 1}"
    _ok("Todos los valores en {0, 1}")

    unique = len(np.unique(X, axis=0))
    assert unique == 32, f"Solo {unique} patrones únicos (se esperaban 32)"
    _ok(f"32 patrones distintos")

    # Distancia de Hamming mínima
    n = X.shape[0]
    min_d = X.shape[1] + 1
    min_pair = None
    for i in range(n):
        for j in range(i + 1, n):
            d = int(np.sum(X[i] != X[j]))
            if d < min_d:
                min_d = d
                min_pair = (labels[i], labels[j])
    _ok(f"Distancia de Hamming mínima: {min_d} (par: '{min_pair[0]}' vs '{min_pair[1]}')")

    # Render ASCII → archivo
    render = render_ascii(X, labels, shape)
    out_txt = OUT_DIR / "font_render.txt"

    header = (
        "=== RENDER ASCII DE LOS 32 CARACTERES (font.h) ===\n"
        f"Patrones : {X.shape[0]}\n"
        f"Features : {X.shape[1]} bits  ({shape[0]}×{shape[1]})\n"
        f"Hamming mínimo entre pares: {min_d}  ('{min_pair[0]}' vs '{min_pair[1]}')\n"
        "Convención: '#' = 1, '.' = 0\n"
        "=" * 42 + "\n\n"
    )
    out_txt.write_text(header + render, encoding="utf-8")
    _ok(f"Render guardado → {out_txt.relative_to(REPO)}")

    # Imprimir también en consola (compacto: 4 chars por fila)
    print()
    PER_ROW = 4
    rows_h, cols_w = shape
    for start in range(0, 32, PER_ROW):
        block = list(range(start, min(start + PER_ROW, 32)))
        # cabecera
        header_line = "  ".join(f" [{i:02d}] {labels[i]:<4}" for i in block)
        print("  " + header_line)
        for r in range(rows_h):
            row_parts = []
            for i in block:
                bitmap = X[i].reshape(rows_h, cols_w)
                row_parts.append("".join("#" if b else "." for b in bitmap[r]))
            print("  " + "    ".join(row_parts))
        print()


# ---------------------------------------------------------------------------
# Test 2: Sprites
# ---------------------------------------------------------------------------

def test_sprites():
    print("\n── TEST SPRITES ──────────────────────────────────────────────")

    from tp5.data.sprites import load_sprites

    tsx = REPO / "data" / "raw" / "sprites" / "tileset_colored.tsx"
    png = REPO / "data" / "raw" / "sprites" / "tileset_legacy.png"

    for p in [tsx, png]:
        if not p.exists():
            _fail(f"No se encontró: {p}")
    # se puede cambiar la region a gusto
    data = load_sprites(str(tsx), str(png), channels="rgb", val_split=0.2, seed=412, dedup=True, regions=[{"name": "animales", "rows": [5, 8], "cols": [24, 31]}])
    object_data = load_sprites(
        str(tsx),
        str(png),
        channels="rgb",
        val_split=0.2,
        seed=42,
        dedup=True,
        regions=[{"name": "objetos", "rows": [0, 5], "cols": [0, 10]}],
    )
    assert object_data["meta"]["n_discarded_bg"] == 2
    assert object_data["meta"]["n_after_dedup"] == 64

    X_train = data["X_train"]
    X_val   = data["X_val"]
    shape   = data["shape"]   # (H, W, C)
    meta    = data["meta"]

    import numpy as np

    # Validaciones básicas
    assert X_train.ndim == 2, "X_train debería ser 2D (n, features)"
    assert X_train.dtype == np.float32
    _ok(f"X_train: {X_train.shape}  (n_tiles × features)")

    assert X_val is not None and X_val.shape[1] == X_train.shape[1]
    _ok(f"X_val:   {X_val.shape}")

    H, W, C = shape
    expected_features = H * W * C
    assert X_train.shape[1] == expected_features, \
        f"Features esperadas: {expected_features}, obtenidas: {X_train.shape[1]}"
    _ok(f"Tile shape: {shape}  → {expected_features} features por tile")

    assert X_train.min() >= 0.0 and X_train.max() <= 1.0, "Valores fuera de [0,1]"
    _ok(f"Valores en [0, 1]  (min={X_train.min():.3f}, max={X_train.max():.3f})")

    # Resumen en texto
    out_meta = OUT_DIR / "sprites_meta.txt"
    lines = [
        "=== DATASET DE SPRITES ===",
        f"PNG fuente  : {meta['png']}",
        f"TSX fuente  : {meta['tsx']}",
        f"Canales     : {meta['channels']}",
        f"Tile shape  : {shape}  ({expected_features} features)",
        f"n_train     : {meta['n_train']}",
        f"n_val       : {meta['n_val']}",
        f"Dedup       : {meta['dedup']}  (antes={meta['n_before_dedup']}, después={meta['n_after_dedup']})",
        f"Val split   : {meta['val_split']}  (seed={meta['seed']})",
        f"Color fondo : {meta['bg_color']}",
    ]
    out_meta.write_text("\n".join(lines), encoding="utf-8")
    _ok(f"Resumen guardado → {out_meta.relative_to(REPO)}")

    # Grilla visual
    _make_sprite_grid(X_train, shape, n_cols=16, n_rows=8)


def _make_sprite_grid(X: "np.ndarray", shape: tuple, n_cols: int = 16, n_rows: int = 8):
    """Guarda una grilla de n_cols × n_rows sprites como PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  [SKIP] matplotlib no disponible — no se genera sprites_grid.png")
        return

    H, W, C = shape
    n_show = min(n_cols * n_rows, len(X))
    idx = np.random.default_rng(0).choice(len(X), size=n_show, replace=False)
    idx = np.sort(idx)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 0.8, n_rows * 0.8))
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle(f"Sprites dataset — {len(X)} tiles (mostrando {n_show})\nShape: {shape}",
                 color="white", fontsize=10, y=1.01)

    for ax_i, tile_i in enumerate(idx):
        r, c = divmod(ax_i, n_cols)
        ax = axes[r, c]
        tile = X[tile_i].reshape(shape).clip(0, 1)
        ax.imshow(tile, interpolation="nearest")
        ax.axis("off")

    # apagar ejes sobrantes
    for ax_i in range(n_show, n_cols * n_rows):
        r, c = divmod(ax_i, n_cols)
        axes[r, c].axis("off")

    plt.tight_layout(pad=0.1)
    out_png = OUT_DIR / "sprites_grid.png"
    fig.savefig(out_png, dpi=100, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  {PASS}  Grilla visual guardada → {out_png.relative_to(REPO)}")


# ---------------------------------------------------------------------------
# Test 3: Loader unificado
# ---------------------------------------------------------------------------

def test_loader():
    print("\n── TEST LOADER ───────────────────────────────────────────────")

    from tp5.data.loader import load_dataset

    # font sin split
    cfg = {"data": {"dataset": "font", "path": "data/processed/font.npz", "val_split": 0.0}}
    ds = load_dataset(cfg)
    assert ds["X_train"].shape == (32, 35)
    assert ds["X_val"] is None
    _ok("Font sin val_split: X_train (32, 35), X_val None")

    # font con split
    cfg2 = {"data": {"dataset": "font", "path": "data/processed/font.npz", "val_split": 0.2, "seed": 71}}
    ds2 = load_dataset(cfg2)
    assert ds2["X_train"].shape[0] + ds2["X_val"].shape[0] == 32
    _ok(f"Font con val_split 0.2: train={ds2['X_train'].shape[0]}, val={ds2['X_val'].shape[0]}")

    # sprites
    cfg3 = {"data": {"dataset": "sprites", "path": "data/processed/sprites.npz", "regions": [{"name": "animales", "rows": [5, 8], "cols": [24, 31]}]}}
    ds3 = load_dataset(cfg3)
    assert ds3["X_train"].ndim == 2
    _ok(f"Sprites loader: X_train {ds3['X_train'].shape}, X_val {ds3['X_val'].shape}")


# ---------------------------------------------------------------------------
# Test 4: Noise
# ---------------------------------------------------------------------------

def test_noise():
    print("\n── TEST NOISE ────────────────────────────────────────────────")

    import numpy as np
    from tp5.data.noise import corrupt

    rng = np.random.default_rng(99)
    X = np.ones((10, 35), dtype=np.uint8)   # todos unos

    # flip con level=1.0 → todos deberían invertirse
    Xf = corrupt(X, "flip", 1.0, rng)
    assert (Xf == 0).all(), "flip 1.0 debería invertir todos los bits"
    _ok("flip level=1.0 → todos los bits invertidos")

    # mask con level=1.0 → todos a 0
    Xm = corrupt(X, "mask", 1.0, rng)
    assert (Xm == 0).all(), "mask 1.0 debería poner todo a 0"
    _ok("mask level=1.0 → todos los bits a 0")

    # flip con level=0.0 → sin cambios
    Xn = corrupt(X, "flip", 0.0, rng)
    assert (Xn == X).all(), "flip 0.0 no debería cambiar nada"
    _ok("flip level=0.0 → sin cambios")

    # Reproducibilidad: misma semilla → mismo resultado
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    Xa = corrupt(X, "flip", 0.3, rng1)
    Xb = corrupt(X, "flip", 0.3, rng2)
    assert (Xa == Xb).all(), "Misma semilla debe dar mismo resultado"
    _ok("Reproducibilidad: misma semilla → misma corrupción")

    # Tasa aproximada (flip 0.1 sobre muestra grande)
    big = np.ones((1000, 35), dtype=np.uint8)
    Xbig = corrupt(big, "flip", 0.1, np.random.default_rng(0))
    rate = (big != Xbig).mean()
    assert 0.07 < rate < 0.13, f"Tasa de flip esperada ~0.10, obtenida {rate:.3f}"
    _ok(f"Tasa de flip 0.1 sobre muestra grande: {rate:.4f}  (esperado ~0.10)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-sprites", action="store_true",
                        help="Omite el test de sprites (si no hay Pillow)")
    args = parser.parse_args()

    print("=" * 60)
    print("  TP5 — Test de la capa de datos")
    print(f"  Repo: {REPO}")
    print(f"  Salidas: {OUT_DIR.relative_to(REPO)}/")
    print("=" * 60)

    test_font()
    test_noise()
    test_loader()

    if not args.no_sprites:
        test_sprites()
    else:
        print("\n── TEST SPRITES  [omitido por --no-sprites] ──────────────")

    print("\n" + "=" * 60)
    print("  Todos los tests pasaron.")
    print(f"  Salidas visuales en: {OUT_DIR.relative_to(REPO)}/")
    print("=" * 60 + "\n")
