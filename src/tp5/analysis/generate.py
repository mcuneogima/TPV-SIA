"""
tp5.analysis.generate
=====================
Generación de caracteres a partir del espacio latente y decodificador.
Soporta tres modos (sección 1.a.4 del PLAN.md):
  (a) Punto específico en zona arbitraria/vacía (z1, z2).
  (b) Punto medio entre dos letras conocidas.
  (c) Interpolación lineal en N pasos entre dos letras.
"""

from typing import Tuple, List, Optional
import numpy as np
import pandas as pd


def generate_from_point(
    model,
    z: np.ndarray,
    binarize: bool = False,
    threshold: float = 0.5,
    shape: Tuple[int, int] = (7, 5),
) -> np.ndarray:
    """
    Decodifica una coordenada latente z de 2 dimensiones.
    Retorna la imagen decodificada con forma (7, 5).
    """
    z_arr = np.asarray(z, dtype=np.float64).reshape(1, -1)
    x_hat = model.decode(z_arr)[0]
    if binarize:
        x_hat = (x_hat >= threshold).astype(np.float64)
    return x_hat.reshape(shape)


def generate_midpoint(
    model,
    df_latent: pd.DataFrame,
    label_a: str,
    label_b: str,
    binarize: bool = False,
    threshold: float = 0.5,
    shape: Tuple[int, int] = (7, 5),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcula el punto medio entre dos letras del espacio latente y lo decodifica.
    Retorna: (z_mid, img_mid, z_a, z_b).
    """
    row_a = df_latent[df_latent["label"] == label_a]
    row_b = df_latent[df_latent["label"] == label_b]

    if row_a.empty:
        raise ValueError(f"Caracter '{label_a}' no encontrado en latent.csv")
    if row_b.empty:
        raise ValueError(f"Caracter '{label_b}' no encontrado en latent.csv")

    z_a = row_a[["z1", "z2"]].values[0]
    z_b = row_b[["z1", "z2"]].values[0]

    z_mid = 0.5 * (z_a + z_b)
    img_mid = generate_from_point(model, z_mid, binarize=binarize, threshold=threshold, shape=shape)

    return z_mid, img_mid, z_a, z_b


def interpolate_linear(
    model,
    df_latent: pd.DataFrame,
    label_a: str,
    label_b: str,
    steps: int = 7,
    binarize: bool = False,
    threshold: float = 0.5,
    shape: Tuple[int, int] = (7, 5),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Realiza una interpolación lineal z(t) = (1-t)*zA + t*zB con t en [0, 1].
    Retorna: (t_values, z_trajectory, decoded_images)
      - t_values: array de tamaño (steps,)
      - z_trajectory: array de tamaño (steps, 2)
      - decoded_images: array de tamaño (steps, 7, 5)
    """
    row_a = df_latent[df_latent["label"] == label_a]
    row_b = df_latent[df_latent["label"] == label_b]

    if row_a.empty:
        raise ValueError(f"Caracter '{label_a}' no encontrado en latent.csv")
    if row_b.empty:
        raise ValueError(f"Caracter '{label_b}' no encontrado en latent.csv")

    z_a = row_a[["z1", "z2"]].values[0]
    z_b = row_b[["z1", "z2"]].values[0]

    t_values = np.linspace(0.0, 1.0, steps)
    z_trajectory = np.outer(1.0 - t_values, z_a) + np.outer(t_values, z_b)

    decoded = model.decode(z_trajectory)
    if binarize:
        decoded = (decoded >= threshold).astype(np.float64)

    images = decoded.reshape(steps, shape[0], shape[1])

    return t_values, z_trajectory, images
