"""
tests/test_analysis.py
======================
Pruebas unitarias para los módulos de análisis y generación de la Fase 3 (Parte 1.a).
"""

import pathlib
import pytest
import numpy as np
import pandas as pd
import yaml

from tp5.analysis.latent import load_model_from_run, load_latent_data, evaluate_decoder_grid, find_nearest_character
from tp5.analysis.generate import generate_from_point, generate_midpoint, interpolate_linear
from tp5.models.registry import build_model
from scripts.plot_vae import _plot_latent_manifold
from scripts.generate import plot_new_letter


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


def test_load_grayscale_sprite_model_without_channels_config(tmp_path):
    cfg = {
        "run_name": "gray_sprite_vae",
        "data": {"dataset": "sprites"},
        "model": {
            "type": "vae",
            "latent_dim": 2,
            "encoder": {"layers": [{"units": 8, "activation": "relu"}]},
            "decoder": {"layers": [{"units": 8, "activation": "relu"}, {"units": 256, "activation": "sigmoid"}]},
            "init": "he",
        },
        "loss": {"recon": "mse", "beta": 1.0},
    }
    model = build_model(cfg, input_dim=256, rng=np.random.default_rng(42))
    run_dir = tmp_path / "gray_sprite_vae"
    run_dir.mkdir()
    (run_dir / "config_resolved.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    np.savez_compressed(run_dir / "weights.npz", **model.get_weights())

    loaded, _ = load_model_from_run(run_dir)

    assert loaded.input_dim == 256
    assert loaded.decode(np.zeros((1, 2))).shape == (1, 256)


def test_grayscale_sprite_manifold(tmp_path):
    class GrayDecoder:
        def decode(self, z):
            self.points = z
            return np.tile(np.linspace(0, 1, 256), (len(z), 1))

    model = GrayDecoder()
    path = tmp_path / "gray_manifold.png"
    _plot_latent_manifold(model, path, steps=3, shape=(16, 16))

    from PIL import Image

    image = np.asarray(Image.open(path).convert("RGB"))
    assert path.is_file()
    assert np.array_equal(image[:, :, 0], image[:, :, 1])
    assert np.array_equal(image[:, :, 1], image[:, :, 2])
    np.testing.assert_array_equal(model.points[[0, -1]], [[-2.5, -2.5], [2.5, 2.5]])


def test_rgb_sprite_manifold(tmp_path):
    class RGBDecoder:
        def decode(self, z):
            image = np.zeros((16, 16, 3))
            image[:, :, 0] = 1
            return np.tile(image.ravel(), (len(z), 1))

    path = tmp_path / "rgb_manifold.png"
    _plot_latent_manifold(RGBDecoder(), path, steps=2, shape=(16, 16, 3))
    from PIL import Image

    image = np.asarray(Image.open(path).convert("RGB"))
    assert np.any(image[:, :, 0] != image[:, :, 1])


def test_new_letter_rejects_training_bitmap(tmp_path, monkeypatch):
    class Decoder:
        def decode(self, z):
            return np.zeros((len(z), 35))

    monkeypatch.setattr(
        "scripts.generate.load_dataset",
        lambda cfg: {"image_shape": (7, 5), "labels": ["a"], "X_train": np.zeros((1, 35))},
    )
    with pytest.raises(ValueError, match="coincide"):
        plot_new_letter(Decoder(), {}, pd.DataFrame(), np.zeros(2), tmp_path / "letter.png", "H")


def test_new_letter_reports_nearest_training_pattern(tmp_path, monkeypatch, capsys):
    class Decoder:
        def decode(self, z):
            output = np.zeros((len(z), 35))
            output[:, :2] = 1
            return output

    monkeypatch.setattr(
        "scripts.generate.load_dataset",
        lambda cfg: {"image_shape": (7, 5), "labels": ["a"], "X_train": np.zeros((1, 35))},
    )
    path = tmp_path / "letter.png"
    plot_new_letter(Decoder(), {}, pd.DataFrame({"z1": [0.0], "z2": [0.0]}), np.ones(2), path, "H")
    assert path.is_file()
    assert "vecino 'a', distancia 2 píxeles" in capsys.readouterr().out
