"""
sprites.py
==========
Pipeline de carga y preprocesamiento del dataset de sprites (tileset 16x16).

El tileset se especifica como un .png + un .tsx (XML de Tiled Map Editor).
El .tsx contiene la geometría: tile size, spacing, columns, etc.

Salida: data/processed/sprites.npz con:
  X_train, X_val            : float32 (n, features)
  tile_index_train/_val     : int (n, 2) — (fila, columna) de origen en el tileset
  shape                     : (H, W, C) o (H, W) si gray
  meta.json                 : parámetros del procesamiento
"""

import json
import pathlib
import xml.etree.ElementTree as ET
import numpy as np

try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


def _parse_tsx(tsx_path: str) -> dict:
    """Extrae geometría del tileset desde el archivo .tsx (XML de Tiled)."""
    tree = ET.parse(tsx_path)
    root = tree.getroot()

    ts = root.attrib
    image_el = root.find("image")

    return {
        "tile_width":  int(ts.get("tilewidth",  16)),
        "tile_height": int(ts.get("tileheight", 16)),
        "spacing":     int(ts.get("spacing",     1)),
        "margin":      int(ts.get("margin",      0)),
        "columns":     int(ts.get("columns",     32)),
        "tile_count":  int(ts.get("tilecount",   0)),
        "img_source":  image_el.attrib.get("source", "") if image_el is not None else "",
        "img_width":   int(image_el.attrib.get("width",  0)) if image_el is not None else 0,
        "img_height":  int(image_el.attrib.get("height", 0)) if image_el is not None else 0,
    }


def _cut_tiles(png_path: str, geom: dict, channels: str = "rgb") -> tuple:
    """
    Corta el tileset en tiles individuales.

    Retorna:
      tiles       : list of np.ndarray (H, W, C) float32 en [0,1]
      tile_coords : list of (row, col) int — posición en la grilla
    """
    if not _PIL_AVAILABLE:
        raise ImportError("Pillow es requerido para leer imágenes. pip install Pillow")

    img = Image.open(png_path).convert("RGBA")
    img_arr = np.array(img, dtype=np.uint8)

    tw = geom["tile_width"]
    th = geom["tile_height"]
    sp = geom["spacing"]
    mg = geom["margin"]
    cols = geom["columns"]

    img_h, img_w = img_arr.shape[:2]
    # calcular filas disponibles
    rows = (img_h - mg) // (th + sp)

    tiles = []
    coords = []

    for r in range(rows):
        for c in range(cols):
            x0 = mg + c * (tw + sp)
            y0 = mg + r * (th + sp)
            x1 = x0 + tw
            y1 = y0 + th
            if x1 > img_w or y1 > img_h:
                continue
            tile = img_arr[y0:y1, x0:x1]  # (H, W, 4) RGBA

            if channels == "gray":
                # promedio RGB
                tile_out = tile[:, :, :3].mean(axis=2).astype(np.float32) / 255.0
            else:
                # RGB, descartar canal alpha
                tile_out = (tile[:, :, :3].astype(np.float32) / 255.0)

            tiles.append(tile_out)
            coords.append((r, c))

    return tiles, coords


def _detect_background_color(tiles: list) -> np.ndarray:
    """Detecta el color de fondo como el color más frecuente en las esquinas."""
    corners = []
    for t in tiles:
        h, w = t.shape[:2]
        for y, x in [(0,0),(0,w-1),(h-1,0),(h-1,w-1)]:
            if t.ndim == 3:
                corners.append(t[y, x])
            else:
                corners.append(t[y, x])
    corners = np.array(corners)
    # redondear a 2 decimales para agrupar colores similares
    rounded = np.round(corners, 2)
    unique, counts = np.unique(rounded.reshape(-1, rounded.shape[-1]) if rounded.ndim > 1 else rounded.reshape(-1,1), axis=0, return_counts=True)
    bg = unique[np.argmax(counts)]
    return bg


def _is_background(tile: np.ndarray, bg_color: np.ndarray, tol: float = 0.05) -> bool:
    """Devuelve True si el tile es casi todo fondo."""
    if tile.ndim == 3:
        diff = np.abs(tile - bg_color).max(axis=2)
    else:
        diff = np.abs(tile - bg_color[0])
    frac_bg = (diff < tol).mean()
    return frac_bg > 0.95


def load_sprites(
    tsx_path: str,
    png_path: str,
    channels: str = "rgb",
    regions: list = None,
    dedup: bool = True,
    hflip_augment: bool = False,
    val_split: float = 0.2,
    seed: int = 42,
) -> dict:
    """
    Carga y preprocesa el tileset de sprites.

    Parámetros
    ----------
    tsx_path  : ruta al .tsx
    png_path  : ruta al .png
    channels  : 'rgb' | 'gray'
    regions   : lista de dicts {name, rows:[r0,r1], cols:[c0,c1]}
                Si None, usa todos los tiles no-fondo.
    dedup     : eliminar tiles duplicados exactos
    hflip_augment : agregar versión espejada horizontal (solo en train)
    val_split : fracción para validación
    seed      : semilla para el split

    Retorna
    -------
    dict con X_train, X_val, tile_index_train, tile_index_val, shape, meta
    """
    geom = _parse_tsx(tsx_path)
    tiles, coords = _cut_tiles(png_path, geom, channels)

    # detectar fondo
    bg_color = _detect_background_color(tiles)

    # filtrar por regiones si se especificaron
    if regions:
        selected = []
        for tile, (r, c) in zip(tiles, coords):
            for reg in regions:
                r0, r1 = reg["rows"]
                c0, c1 = reg["cols"]
                if r0 <= r <= r1 and c0 <= c <= c1:
                    selected.append((tile, (r, c)))
                    break
        tiles = [t for t, _ in selected]
        coords = [rc for _, rc in selected]
    else:
        # filtrar tiles vacíos (solo fondo)
        kept = [(t, rc) for t, rc in zip(tiles, coords) if not _is_background(t, bg_color)]
        tiles = [t for t, _ in kept]
        coords = [rc for _, rc in kept]

    if not tiles:
        raise ValueError("No se encontraron tiles tras el filtrado. Revisá las regiones o el PNG.")

    # aplanar tiles
    flat = np.array([t.flatten() for t in tiles], dtype=np.float32)
    coords_arr = np.array(coords, dtype=np.int32)

    # dedup
    n_before_dedup = len(flat)
    if dedup:
        _, unique_idx = np.unique(flat, axis=0, return_index=True)
        unique_idx = np.sort(unique_idx)
        flat = flat[unique_idx]
        coords_arr = coords_arr[unique_idx]
    n_after_dedup = len(flat)

    # shape de cada tile
    tw = geom["tile_width"]
    th = geom["tile_height"]
    if channels == "gray":
        tile_shape = (th, tw)
    else:
        tile_shape = (th, tw, 3)

    # split train/val
    rng = np.random.default_rng(seed)
    idx = np.arange(len(flat))
    rng.shuffle(idx)
    n_val = int(len(flat) * val_split)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    X_train = flat[train_idx]
    X_val   = flat[val_idx] if n_val > 0 else None
    idx_train = coords_arr[train_idx]
    idx_val   = coords_arr[val_idx] if n_val > 0 else None

    # augmentación horizontal (solo train)
    if hflip_augment and len(X_train) > 0:
        tw_c = tw * (3 if channels == "rgb" else 1)
        flipped = []
        for row in X_train:
            tile_arr = row.reshape(tile_shape)
            flipped.append(np.fliplr(tile_arr).flatten())
        X_train = np.vstack([X_train, np.array(flipped, dtype=np.float32)])
        idx_train = np.vstack([idx_train, idx_train])

    meta = {
        "tsx": tsx_path,
        "png": png_path,
        "channels": channels,
        "regions": regions,
        "dedup": dedup,
        "n_before_dedup": n_before_dedup,
        "n_after_dedup": n_after_dedup,
        "n_discarded_bg": len(tiles),
        "val_split": val_split,
        "seed": seed,
        "n_train": len(X_train),
        "n_val": len(X_val) if X_val is not None else 0,
        "tile_shape": list(tile_shape),
        "bg_color": bg_color.tolist(),
    }

    return {
        "X_train": X_train,
        "X_val":   X_val,
        "tile_index_train": idx_train,
        "tile_index_val":   idx_val,
        "shape":   tile_shape,
        "meta":    meta,
    }


def save_sprites(data: dict, out_npz: str, out_meta: str = None) -> None:
    """Guarda sprites.npz y meta.json."""
    pathlib.Path(out_npz).parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        "X_train": data["X_train"],
        "tile_index_train": data["tile_index_train"],
        "shape": np.array(data["shape"]),
    }
    if data["X_val"] is not None:
        save_dict["X_val"] = data["X_val"]
        save_dict["tile_index_val"] = data["tile_index_val"]

    np.savez_compressed(out_npz, **save_dict)
    print(f"[INFO] sprites.npz guardado en {out_npz}")

    if out_meta:
        pathlib.Path(out_meta).parent.mkdir(parents=True, exist_ok=True)
        with open(out_meta, "w", encoding="utf-8") as f:
            json.dump(data["meta"], f, indent=2)
        print(f"[INFO] meta.json guardado en {out_meta}")


def load_sprites_npz(path: str) -> dict:
    """Carga sprites.npz y retorna dict."""
    npz = np.load(path, allow_pickle=True)
    out = {
        "X_train": npz["X_train"],
        "tile_index_train": npz["tile_index_train"],
        "shape": tuple(npz["shape"]),
    }
    if "X_val" in npz:
        out["X_val"] = npz["X_val"]
        out["tile_index_val"] = npz["tile_index_val"]
    else:
        out["X_val"] = None
        out["tile_index_val"] = None
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    root = pathlib.Path(__file__).resolve().parents[3]
    tsx  = root / "data" / "raw" / "sprites" / "tileset_colored.tsx"
    png  = root / "data" / "raw" / "sprites" / "tileset_legacy.png"
    out_npz  = root / "data" / "processed" / "sprites.npz"
    out_meta = root / "data" / "processed" / "sprites_meta.json"

    if not tsx.exists() or not png.exists():
        print(f"[ERROR] No se encontraron los archivos del tileset en {tsx.parent}", file=sys.stderr)
        sys.exit(1)

    data = load_sprites(str(tsx), str(png), channels="rgb", val_split=0.2, seed=42)
    print(f"[INFO] Train: {data['X_train'].shape}, Val: {data['X_val'].shape if data['X_val'] is not None else 'None'}")
    save_sprites(data, str(out_npz), str(out_meta))
