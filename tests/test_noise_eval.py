import pathlib
import sys

import numpy as np
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tp5.analysis.noise_eval import evaluate_noise_levels, reconstruction_metrics
from tp5.train.sweep import expand_sweep_grid


def test_reconstruction_metrics_count_pattern_error_threshold():
    clean = np.array([[0, 0, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
    prediction = np.array([[0.1, 0.9, 0.9], [0.1, 0.1, 0.1], [0.9, 0.9, 0.1]])

    metrics = reconstruction_metrics(clean, prediction)

    assert metrics["mean_pixel_errors"] == pytest.approx(1.0)
    assert metrics["max_pixel_errors"] == 2.0
    assert metrics["frac_perfect"] == pytest.approx(1 / 3)
    assert metrics["frac_ok"] == pytest.approx(2 / 3)


def test_noise_evaluation_is_reproducible_and_tests_clean_targets():
    class IdentityModel:
        def forward(self, values):
            return values

    clean = np.array([[0.0, 1.0], [1.0, 0.0]])
    first = evaluate_noise_levels(IdentityModel(), clean, "flip", [0.2], 3, seed=7)
    second = evaluate_noise_levels(IdentityModel(), clean, "flip", [0.2], 3, seed=7)

    assert first == second
    assert len(first) == 3
    assert all("frac_ok" in result for result in first)
    noisy_baseline = evaluate_noise_levels(None, clean, "flip", [0.2], 1, seed=7)
    assert noisy_baseline[0]["mean_pixel_errors"] >= 0.0


def test_sweep_cartesian_and_ofat_modes_expand_expected_variants(tmp_path):
    base_path = tmp_path / "base.yaml"
    base_path.write_text(
        "seed: 9\nmodel:\n  init: xavier\n  encoder:\n    layers:\n      - {units: 2, activation: tanh}\n",
        encoding="utf-8",
    )
    variants = {"wide": {"layers": [{"units": 4, "activation": "relu"}]}}

    cartesian = expand_sweep_grid({
        "base": str(base_path),
        "seeds": [1, 2],
        "grid": {"model.init": ["xavier", "he"], "model.encoder.layers": ["wide"]},
        "variants": variants,
    })
    assert len(cartesian) == 4
    assert {cfg["model"]["init"] for cfg in cartesian} == {"xavier", "he"}
    assert all(cfg["model"]["encoder"]["layers"][0]["units"] == 4 for cfg in cartesian)

    ofat = expand_sweep_grid({
        "base": str(base_path),
        "mode": "ofat",
        "seeds": [1],
        "grid": {"model.init": ["xavier", "he"]},
    })
    assert len(ofat) == 2
    assert {cfg["model"]["init"] for cfg in ofat} == {"xavier", "he"}
