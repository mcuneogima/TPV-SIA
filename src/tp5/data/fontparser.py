"""
fontparser.py
=============
Parsea data/raw/font.h y produce data/processed/font.npz.

Convención de bits:
  - Cada fila del caracter es un byte (uint8) con 5 bits útiles.
  - Bit 4 = columna izquierda, bit 0 = columna derecha.
  - Ejemplo: 0x0e = 0b01110 → columnas [0,1,1,1,0]

Uso como script:
  python -m tp5.data.fontparser
  python scripts/preprocess_font.py
"""

import re
import hashlib
import pathlib
import numpy as np


COLS = 5
ROWS = 7
N_CHARS = 32
FLAT = ROWS * COLS   # 35


def _extract_rows_and_labels(path: str) -> tuple:
    """Lee font.h y extrae los 32x7 bytes y los labels."""
    text = pathlib.Path(path).read_text(encoding="ascii", errors="replace")

    # Extraer cada bloque { byte, byte, ... } con su comentario opcional
    # Patrón: { hex, hex, ... }  // opcional_label
    block_pattern = re.compile(
        r'\{([^}]+)\}[^/\n]*(?://\s*0x[0-9a-fA-F]+,?\s*(.*))?',
    )

    rows_all = []   # lista de listas de 7 ints
    labels = []

    for m in block_pattern.finditer(text):
        hex_part = m.group(1)
        label_part = (m.group(2) or "").strip()

        # extraer todos los literales hex del bloque
        hex_vals = re.findall(r'0x[0-9a-fA-F]{2}', hex_part)
        if len(hex_vals) != ROWS:
            continue   # ignorar la declaración del array exterior

        byte_vals = [int(h, 16) for h in hex_vals]
        rows_all.append(byte_vals)

        # label: usar el caracter imprimible si está en el comentario
        char_label = label_part if label_part else f"0x{0x60 + len(labels):02x}"
        labels.append(char_label)

    return rows_all, labels


def _bytes_to_bitmap(byte_row: list) -> np.ndarray:
    """Convierte 7 bytes a matriz (7, 5) de bits {0, 1}."""
    bitmap = np.zeros((ROWS, COLS), dtype=np.uint8)
    for r, byte in enumerate(byte_row):
        for c in range(COLS):
            # bit 4-c → columna c (bit 4 = izq, bit 0 = der)
            bitmap[r, c] = (byte >> (COLS - 1 - c)) & 1
    return bitmap


def parse_font(path: str) -> dict:
    """
    Lee font.h y devuelve:
      X      : np.uint8 (32, 35)  — patrones aplanados por filas
      labels : list[str] (32)
      shape  : (7, 5)
    """
    rows_all, labels = _extract_rows_and_labels(path)

    if len(rows_all) != N_CHARS:
        raise ValueError(
            f"Se esperaban {N_CHARS} caracteres, se encontraron {len(rows_all)}. "
            f"Revisá el formato de font.h."
        )

    bitmaps = [_bytes_to_bitmap(r) for r in rows_all]
    X = np.array([b.flatten() for b in bitmaps], dtype=np.uint8)

    # --- validaciones ---
    assert X.shape == (N_CHARS, FLAT), f"Shape incorrecto: {X.shape}"
    assert np.all((X == 0) | (X == 1)), "Hay valores fuera de {0, 1}"

    unique_rows = np.unique(X, axis=0)
    if unique_rows.shape[0] != N_CHARS:
        duplicates = N_CHARS - unique_rows.shape[0]
        print(f"[WARN] {duplicates} patrón(es) duplicado(s) detectado(s).")

    # Distancia de Hamming mínima entre pares
    min_hamming = _min_hamming(X)
    print(f"[INFO] Distancia de Hamming mínima entre pares: {min_hamming}")

    return {"X": X, "labels": labels, "shape": (ROWS, COLS)}


def _min_hamming(X: np.ndarray) -> int:
    """Calcula la distancia de Hamming mínima entre todos los pares de patrones."""
    n = X.shape[0]
    min_d = FLAT + 1
    for i in range(n):
        for j in range(i + 1, n):
            d = int(np.sum(X[i] != X[j]))
            if d < min_d:
                min_d = d
    return min_d


def save_font(data: dict, out_path: str) -> None:
    """Guarda el dataset de font en un .npz."""
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X=data["X"],
        labels=np.array(data["labels"]),
        shape=np.array(data["shape"]),
    )
    print(f"[INFO] Guardado en {out_path}")


def load_font(path: str) -> dict:
    """Carga font.npz y devuelve dict con X, labels, shape."""
    npz = np.load(path, allow_pickle=True)
    return {
        "X": npz["X"],
        "labels": list(npz["labels"]),
        "shape": tuple(npz["shape"]),
    }


def render_ascii(X: np.ndarray, labels: list, shape: tuple = (ROWS, COLS)) -> str:
    """Genera un string con el render ASCII de todos los patrones."""
    rows, cols = shape
    lines = []
    for idx, (flat, label) in enumerate(zip(X, labels)):
        bitmap = flat.reshape(rows, cols)
        lines.append(f"--- [{idx:02d}] {label} ---")
        for row in bitmap:
            lines.append("".join("#" if b else "." for b in row))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    root = pathlib.Path(__file__).resolve().parents[3]  # raíz del repo
    font_h = root / "data" / "raw" / "font.h"
    out_npz = root / "data" / "processed" / "font.npz"

    if not font_h.exists():
        print(f"[ERROR] No se encontró {font_h}", file=sys.stderr)
        sys.exit(1)

    data = parse_font(str(font_h))
    print(render_ascii(data["X"], data["labels"], data["shape"]))
    save_font(data, str(out_npz))
    print(f"\n[OK] {data['X'].shape[0]} patrones de {data['X'].shape[1]} bits guardados en {out_npz}")
