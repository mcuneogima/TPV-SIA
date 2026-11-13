"""
tests/test_analysis.py
======================
Pruebas unitarias para los módulos de análisis y generación de la Fase 3 (Parte 1.a).
"""

import pathlib
import pytest
import numpy as np
import pandas as pd

from tp5.analysis.latent import load_model_from_run, load_latent_data, evaluate_decoder_grid, find_nearest_character
from tp5.analysis.generate import generate_from_point, generate_midpoint, interpolate_linear


@pytest.fixture
def sample_run_dir():
    # Usar una corrida existente en results/runs/
    runs_dir = pathlib.Path("results/runs")
    all_runs = list(runs_dir.iterdir())
    assert len(all_runs) > 0, "Debe haber al menos una corrida entrenada en results/runs/"
    return all_runs[0]


def test_load_model_and_latent(sample_run_dir):
    model, cfg = load_model_from_run(sample_run_dir)
    assert model is not None
    assert "model" in cfg

    df_latent = load_latent_data(sample_run_dir)
    assert isinstance(df_latent, pd.DataFrame)
    assert "z1" in df_latent.columns
    assert "z2" in df_latent.columns
    assert "label" in df_latent.columns
    assert len(df_latent) == 32


def test_evaluate_decoder_grid(sample_run_dir):
    model, _ = load_model_from_run(sample_run_dir)
    gx, gy, decoded = evaluate_decoder_grid(model, -2.0, 2.0, -2.0, 2.0, steps=4, shape=(7, 5))
    assert len(gx) == 4
    assert len(gy) == 4
    assert decoded.shape == (4, 4, 7, 5)
    assert np.all(decoded >= 0.0) and np.all(decoded <= 1.0)


def test_generation_point(sample_run_dir):
    model, _ = load_model_from_run(sample_run_dir)
    img_cont = generate_from_point(model, [0.0, 0.0], binarize=False)
    assert img_cont.shape == (7, 5)
    assert np.all(img_cont >= 0.0) and np.all(img_cont <= 1.0)

    img_bin = generate_from_point(model, [0.0, 0.0], binarize=True)
    assert set(np.unique(img_bin)).issubset({0.0, 1.0})


def test_generation_midpoint_and_interpolate(sample_run_dir):
    model, _ = load_model_from_run(sample_run_dir)
    df_latent = load_latent_data(sample_run_dir)

    # Punto medio
    z_mid, img_mid, z_a, z_b = generate_midpoint(model, df_latent, "a", "b")
    assert z_mid.shape == (2,)
    assert img_mid.shape == (7, 5)

    # Interpolación
    t_vals, z_traj, imgs = interpolate_linear(model, df_latent, "a", "b", steps=5)
    assert len(t_vals) == 5
    assert z_traj.shape == (5, 2)
    assert imgs.shape == (5, 7, 5)

    # Búsqueda de vecino cercano
    lbl, dist, idx = find_nearest_character(df_latent, z_a)
    assert lbl == "a"
    assert np.isclose(dist, 0.0)
