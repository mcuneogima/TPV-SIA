"""Compare matched 500/2000-epoch sprite VAEs and export training reconstructions."""

import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tp5.analysis.latent import load_model_from_run
from tp5.data.loader import load_dataset

SEEDS = (42, 101, 202)
OUTPUT = ROOT / "figures" / "vae_long"
CONDITIONS = {
    "full": {
        500: (ROOT / "results/summary/runs.csv", ROOT / "results/runs", "vae_sprites_b1"),
        2000: (ROOT / "results/sprites_long/full/summary/runs.csv", ROOT / "results/sprites_long/full/runs", "vae_sprites_full_long"),
    },
    "animals_gray": {
        500: (ROOT / "results/sprites_long/animals_gray/summary/runs.csv", ROOT / "results/sprites_long/animals_gray/runs", "vae_sprites_animals_gray"),
        2000: (ROOT / "results/sprites_long/animals_gray/summary/runs.csv", ROOT / "results/sprites_long/animals_gray/runs", "vae_sprites_animals_gray_long_matched"),
    },
}


def find_run(spec, epochs, seed):
    summary, runs_dir, run_name = spec
    runs = pd.read_csv(summary)
    selected = runs[
        (runs["run_name"] == run_name)
        & (runs["model_type"] == "vae")
        & (runs["dataset"] == "sprites")
        & (runs["seed"] == seed)
        & (runs["epochs_run"] == epochs)
        & np.isclose(runs["beta"], 0.5)
        & runs["arch"].str.contains("[2x16]", regex=False)
        & (runs["status"] == "ok")
    ]["run_id"].drop_duplicates()
    if len(selected) != 1:
        raise ValueError(f"Se esperaba una corrida para {run_name}, {epochs}, seed={seed}: {selected.tolist()}")
    path = runs_dir / selected.iloc[0]
    if not (path / "weights.npz").exists():
        raise FileNotFoundError(f"Faltan pesos de {path}")
    return path


def show_image(axis, image, shape):
    image = image.reshape(shape)
    if len(shape) == 2:
        axis.imshow(image, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    else:
        axis.imshow(np.clip(image, 0, 1), interpolation="nearest")
    axis.set_xticks([])
    axis.set_yticks([])


def plot_reconstructions(models, data, split, output):
    originals = data[f"X_{split}"]
    rng = np.random.default_rng(42)
    indices = rng.choice(len(originals), size=min(16, len(originals)), replace=False)
    selected = originals[indices]
    images = [selected] + [model.decode(model.encode_mean(selected)) for model in models]
    fig, axes = plt.subplots(3, len(indices), figsize=(1.35 * len(indices), 4.5), dpi=140, squeeze=False)
    for row, (label, batch) in enumerate(zip(("Original", "500 épocas", "2000 épocas"), images)):
        for col, image in enumerate(batch):
            show_image(axes[row, col], image, data["image_shape"])
            if row == 0:
                axes[row, col].set_title(f"#{indices[col]}", fontsize=8)
        axes[row, 0].set_ylabel(label)
    fig.suptitle(f"{split}: originales y decodificación de μ · semilla 42")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_all_training(model, data, output_dir):
    originals = data["X_train"]
    reconstructed = model.decode(model.encode_mean(originals))
    shape = data["image_shape"]
    gray = len(shape) == 2
    image_shape = (16, 16, 3)
    per_page = 80
    output_dir.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(originals), per_page):
        count = min(per_page, len(originals) - start)
        rows = (count + 9) // 10
        canvas = np.ones((rows * 37, 10 * 34, 3), dtype=np.float64)
        for index in range(count):
            row, col = divmod(index, 10)
            for side, batch in enumerate((originals, reconstructed)):
                image = batch[start + index].reshape(shape)
                if gray:
                    image = np.repeat(image[:, :, None], 3, axis=2)
                canvas[row * 37:row * 37 + 16, col * 34 + side * 17:col * 34 + side * 17 + 16] = np.clip(image, 0, 1).reshape(image_shape)
        fig, axis = plt.subplots(figsize=(17, rows * 1.7), dpi=140)
        axis.imshow(canvas, interpolation="nearest")
        axis.set_axis_off()
        for index in range(count):
            row, col = divmod(index, 10)
            axis.text(col * 34 + 16, row * 37 + 19, str(start + index), ha="center", va="top", fontsize=7)
        axis.set_title(f"Train: original | reconstrucción con μ, 2000 épocas, seed 42 · índices {start}–{start + count - 1}")
        fig.tight_layout()
        fig.savefig(output_dir / f"train_{start:04d}_{start + count - 1:04d}.png", bbox_inches="tight")
        plt.close(fig)


def plot_prior(models, data, output):
    rng = np.random.default_rng(42)
    z = rng.normal(size=(32, 16))
    samples = [model.decode(z) for model in models]
    fig, axes = plt.subplots(8, 8, figsize=(12, 12), dpi=140, squeeze=False)
    for index in range(32):
        group, col = divmod(index, 8)
        for version in range(2):
            axis = axes[group * 2 + version, col]
            show_image(axis, samples[version][index], data["image_shape"])
            if version == 0:
                axis.set_title(f"z #{index}", fontsize=8)
        axes[group * 2, 0].set_ylabel("500")
        axes[group * 2 + 1, 0].set_ylabel("2000")
    fig.suptitle("Mismos 32 vectores z ~ N(0,I), modelos seed 42")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_prior_neighbors(model, data, output):
    rng = np.random.default_rng(42)
    samples = model.decode(rng.normal(size=(32, 16)))
    train = data["X_train"]
    distances = np.mean((samples[:, None, :] - train[None, :, :]) ** 2, axis=2)
    nearest = np.argmin(distances, axis=1)
    fig, axes = plt.subplots(8, 8, figsize=(12, 12), dpi=140, squeeze=False)
    for index in range(32):
        group, col = divmod(index, 8)
        show_image(axes[group * 2, col], samples[index], data["image_shape"])
        show_image(axes[group * 2 + 1, col], train[nearest[index]], data["image_shape"])
        axes[group * 2, col].set_title(f"#{index} · {distances[index, nearest[index]]:.3f}", fontsize=8)
        axes[group * 2 + 1, col].set_xlabel(f"train #{nearest[index]}", fontsize=8)
        axes[group * 2, 0].set_ylabel("Prior")
        axes[group * 2 + 1, 0].set_ylabel("Vecino")
    fig.suptitle("2000 épocas: muestras del prior y vecino train por MSE · seed 42")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def nearest_centroid_distance(points, means):
    distances = np.sum((points[:, None, :] - means[None, :, :]) ** 2, axis=2)
    return np.sqrt(np.min(distances, axis=1))


def summarize(model, data, history, condition, epochs, seed, run_id):
    train, val = data["X_train"], data["X_val"]
    mu, logvar = model.encode(train)
    rng = np.random.default_rng(12345)
    chosen = rng.integers(len(train), size=1000)
    posterior = mu[chosen] + np.exp(0.5 * logvar[chosen]) * rng.normal(size=(1000, mu.shape[1]))
    prior = rng.normal(size=(1000, mu.shape[1]))
    posterior_distance = nearest_centroid_distance(posterior, mu)
    prior_distance = nearest_centroid_distance(prior, mu)
    sample_rng = np.random.default_rng(42)
    samples = model.decode(sample_rng.normal(size=(32, mu.shape[1])))
    pixel_distances = np.mean((samples[:, None, :] - train[None, :, :]) ** 2, axis=2)
    validation = history.dropna(subset=["val_loss"])
    best = validation.loc[validation["val_loss"].idxmin()]
    final = history.iloc[-1]
    return {
        "condition": condition, "epochs": epochs, "seed": seed, "run_id": run_id,
        "train_mse_per_feature": np.mean((model.decode(model.encode_mean(train)) - train) ** 2),
        "val_mse_per_feature": np.mean((model.decode(model.encode_mean(val)) - val) ** 2),
        "kl_train": final["kl_loss"],
        "best_val_epoch": int(best["epoch"]),
        "best_val_mse_per_feature": best["val_loss"] / train.shape[1],
        "prior_nearest_mu_median": np.median(prior_distance),
        "posterior_nearest_mu_median": np.median(posterior_distance),
        "prior_nearest_train_mse_median": np.median(np.min(pixel_distances, axis=1)),
    }


def plot_curves(histories, features, output):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=150)
    for seed, history in histories.items():
        axes[0].plot(history["epoch"], history["recon_loss"] / features, label=str(seed))
        axes[1].plot(history["epoch"], history["val_loss"] / features, label=str(seed))
        axes[2].plot(history["epoch"], history["kl_loss"], label=str(seed))
    for axis, title in zip(axes, ("MSE train por feature", "MSE validación por feature", "KL train")):
        axis.axvline(500, color="black", linestyle="--", linewidth=1)
        axis.set_title(title)
        axis.set_xlabel("Época")
        axis.grid(alpha=0.25)
        axis.legend(title="Semilla")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results = []
    for condition, specs in CONDITIONS.items():
        out = OUTPUT / condition
        out.mkdir(parents=True, exist_ok=True)
        histories = {}
        for seed in SEEDS:
            models = []
            datasets = []
            configs = []
            for epochs in (500, 2000):
                run = find_run(specs[epochs], epochs, seed)
                model, cfg = load_model_from_run(run)
                data = load_dataset(cfg)
                history = pd.read_csv(run / "epochs.csv")
                results.append(summarize(model, data, history, condition, epochs, seed, run.name))
                models.append(model)
                datasets.append(data)
                configs.append(cfg)
                if epochs == 2000:
                    histories[seed] = history
            for section in ("data", "model", "loss", "noise", "optimizer"):
                if configs[0].get(section) != configs[1].get(section):
                    raise ValueError(f"Configuración {section} distinta: {condition}, seed {seed}")
            train_cfgs = [{key: value for key, value in cfg["train"].items() if key != "epochs"} for cfg in configs]
            if train_cfgs[0] != train_cfgs[1]:
                raise ValueError(f"Parámetros de entrenamiento distintos: {condition}, seed {seed}")
            for split in ("X_train", "X_val"):
                if not np.array_equal(datasets[0][split], datasets[1][split]):
                    raise ValueError(f"Datasets distintos: {condition}, seed {seed}, {split}")
            if seed == 42:
                data = datasets[1]
                for split in ("train", "val"):
                    plot_reconstructions(models, data, split, out / f"recon_{split}.png")
                plot_all_training(models[1], data, out / "train_completo")
                plot_prior(models, data, out / "prior_comparacion.png")
                plot_prior_neighbors(models[1], data, out / "prior_vecinos.png")
        plot_curves(histories, datasets[1]["X_train"].shape[1], out / "curvas.png")
    pd.DataFrame(results).to_csv(OUTPUT / "metrics.csv", index=False)
    print(f"Análisis guardado en {OUTPUT}")


if __name__ == "__main__":
    main()
