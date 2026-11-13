"""Generate analysis figures for the sprite VAE experiments."""

import argparse
import pathlib
import sys
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tp5.analysis.latent import load_latent_data, load_model_from_run


def _run_path(runs_dir: pathlib.Path, run_id: str) -> pathlib.Path:
    path = runs_dir / str(run_id)
    if not (path / "weights.npz").exists():
        raise FileNotFoundError(f"No se encuentran pesos para la corrida {run_id}: {path}")
    return path


def _load_summary(path: pathlib.Path, runs_dir: pathlib.Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}. Ejecutá primero el sweep VAE.")
    df = pd.read_csv(path)
    required = {"run_id", "dataset", "model_type", "beta", "status"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"El resumen no contiene las columnas: {', '.join(sorted(missing))}")
    runs = df[
        (df["dataset"] == "sprites")
        & (df["model_type"] == "vae")
        & df["status"].isin(["ok", "early_stopped"])
    ].copy()
    runs = runs[runs["run_id"].map(lambda run_id: (runs_dir / str(run_id) / "weights.npz").exists())]
    if runs.empty:
        raise ValueError("No hay corridas VAE de sprites válidas con pesos guardados.")
    return runs


def _epoch_frame(row: pd.Series, runs_dir: pathlib.Path) -> pd.DataFrame:
    path = _run_path(runs_dir, row["run_id"]) / "epochs.csv"
    if not path.exists():
        raise FileNotFoundError(f"No existe el historial de épocas: {path}")
    return pd.read_csv(path)


def _lowest_validation_row(runs: pd.DataFrame, runs_dir: pathlib.Path) -> pd.Series:
    scored = []
    for _, row in runs.iterrows():
        history = _epoch_frame(row, runs_dir)
        values = history["val_loss"].dropna()
        if not values.empty:
            scored.append((float(values.iloc[-1]), row))
    if not scored:
        raise ValueError("No hay métricas de validación para seleccionar el VAE.")
    return min(scored, key=lambda item: item[0])[1]


def _config_value(config_path: str, section: str, key: str) -> object:
    import yaml

    path = pathlib.Path(config_path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file).get(section, {}).get(key)


def _plot_metric_curves(
    runs: pd.DataFrame,
    runs_dir: pathlib.Path,
    groups: Iterable[Tuple[str, pd.DataFrame]],
    output_path: pathlib.Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=180)
    metrics = [
        ("recon_loss", "Reconstrucción (MSE)"),
        ("kl_loss", "Divergencia KL"),
        ("loss", "Pérdida total"),
    ]
    for label, group in groups:
        histories = [_epoch_frame(row, runs_dir) for _, row in group.iterrows()]
        if not histories:
            continue
        epochs = np.sort(np.unique(np.concatenate([history["epoch"].values for history in histories])))
        matrices = []
        for history in histories:
            aligned = history.set_index("epoch").reindex(epochs).ffill().bfill()
            matrices.append(aligned)
        for axis, (metric, ylabel) in zip(axes, metrics):
            values = np.stack([history[metric].to_numpy(dtype=float) for history in matrices])
            mean = np.mean(values, axis=0)
            std = np.std(values, axis=0)
            axis.plot(epochs, mean, linewidth=1.8, label=f"{label} (n={len(histories)})")
            axis.fill_between(epochs, np.maximum(0, mean - std), mean + std, alpha=0.15)
            axis.set_xlabel("Época")
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.25)
    for axis in axes:
        axis.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_beta_tradeoff(runs: pd.DataFrame, runs_dir: pathlib.Path, path: pathlib.Path) -> None:
    rows = []
    for beta in sorted(runs["beta"].dropna().astype(float).unique()):
        candidates = runs[
            np.isclose(runs["beta"].astype(float), beta)
            & runs["arch"].astype(str).str.contains(r"\[2x16\]", regex=True)
            & (runs["_warmup"] == 50)
        ]
        for _, run in candidates.iterrows():
            history = _epoch_frame(run, runs_dir)
            final = history.iloc[-1]
            rows.append({
                "beta": beta,
                "recon": float(final.get("val_loss", np.nan)),
                "kl": float(final["kl_loss"]),
                "seed": int(run["seed"]),
            })
    tradeoff = pd.DataFrame(rows).drop_duplicates(["beta", "seed"])
    fig, ax = plt.subplots(figsize=(8, 6), dpi=180)
    for beta, group in tradeoff.groupby("beta"):
        ax.scatter(group["recon"], group["kl"], s=65, label=f"β={beta:g}")
        for _, row in group.iterrows():
            ax.annotate(str(int(row["seed"])), (row["recon"], row["kl"]), xytext=(4, 3), textcoords="offset points", fontsize=7)
    ax.set_xlabel("MSE de validación (menor es mejor)")
    ax.set_ylabel("KL final promedio")
    ax.set_title("Trade-off de β (dimensión latente 16)")
    ax.grid(alpha=0.3)
    ax.legend(title="Regularización")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_latent_posterior(run_dir: pathlib.Path, npz_path: pathlib.Path, path: pathlib.Path) -> None:
    latent = load_latent_data(run_dir)
    if not {"z1", "z2", "logvar1", "logvar2"}.issubset(latent.columns):
        raise ValueError("La figura de posterior requiere una corrida VAE con al menos dos dimensiones latentes.")
    archive = np.load(npz_path, allow_pickle=True)
    tile_indices = archive["tile_index_train"]
    n = min(len(latent), len(tile_indices))
    latent = latent.iloc[:n]
    rows = tile_indices[:n, 0]
    fig, ax = plt.subplots(figsize=(9, 7), dpi=180)
    points = ax.scatter(latent["z1"], latent["z2"], c=rows, cmap="viridis", s=18, alpha=0.75)
    fig.colorbar(points, ax=ax, label="Fila de origen en el tileset (no es una clase semántica)")
    selected = np.linspace(0, n - 1, min(12, n), dtype=int)
    for index in selected:
        sigma_1 = float(np.exp(0.5 * np.clip(latent.iloc[index]["logvar1"], -20, 20)))
        sigma_2 = float(np.exp(0.5 * np.clip(latent.iloc[index]["logvar2"], -20, 20)))
        ax.add_patch(Ellipse(
            (latent.iloc[index]["z1"], latent.iloc[index]["z2"]),
            width=2 * sigma_1,
            height=2 * sigma_2,
            fill=False,
            edgecolor="#DC2626",
            linewidth=0.9,
            alpha=0.7,
        ))
    ax.set_xlabel("$\\mu_1$")
    ax.set_ylabel("$\\mu_2$")
    ax.set_title("Medias del posterior y elipses de ±1σ (primeras 2 dimensiones)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_latent_manifold(model: object, path: pathlib.Path, steps: int = 13) -> None:
    axis_values = np.linspace(-2.5, 2.5, steps)
    grid_1, grid_2 = np.meshgrid(axis_values, axis_values)
    latent = np.stack([grid_1.ravel(), grid_2.ravel()], axis=1)
    decoded = model.decode(latent)
    fig, axes = plt.subplots(steps, steps, figsize=(12, 12), dpi=140)
    for index, axis in enumerate(axes.flat):
        axis.imshow(decoded[index].reshape(16, 16, 3), interpolation="nearest")
        axis.set_axis_off()
    fig.suptitle("Decoder sobre una grilla del prior N(0,I), espacio latente 2D", y=0.995)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_reconstructions(
    model: object,
    archive: np.lib.npyio.NpzFile,
    path: pathlib.Path,
    seed: int,
    n_samples: int = 8,
) -> None:
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(4, n_samples, figsize=(2 * n_samples, 8), dpi=160)
    for split_index, (split, originals) in enumerate((
        ("train", archive["X_train"]),
        ("val", archive["X_val"]),
    )):
        count = min(n_samples, len(originals))
        indices = rng.choice(len(originals), size=count, replace=False)
        chosen = originals[indices].astype(np.float64)
        recon = model.decode(model.encode_mean(chosen))
        for col in range(n_samples):
            axes[split_index * 2, col].set_axis_off()
            axes[split_index * 2 + 1, col].set_axis_off()
            if col < count:
                axes[split_index * 2, col].imshow(chosen[col].reshape(16, 16, 3))
                axes[split_index * 2 + 1, col].imshow(recon[col].reshape(16, 16, 3))
        axes[split_index * 2, 0].set_ylabel(f"{split}\noriginal")
        axes[split_index * 2 + 1, 0].set_ylabel(f"{split}\nreconstrucción")
    fig.suptitle("Reconstrucciones deterministas usando z=μ")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_dataset_samples(archive: np.lib.npyio.NpzFile, path: pathlib.Path, seed: int) -> None:
    train = archive["X_train"]
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(train), size=min(64, len(train)), replace=False)
    columns = 8
    rows = int(np.ceil(len(indices) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 1.5, rows * 1.5), dpi=150, squeeze=False)
    for cell, axis in enumerate(axes.flat):
        axis.set_axis_off()
        if cell < len(indices):
            axis.imshow(train[indices[cell]].reshape(16, 16, 3))
    fig.suptitle("Muestras del dataset procesado de sprites (train)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_interpolation(
    ae_model: object,
    vae_model: object,
    ae_latent: pd.DataFrame,
    vae_latent: pd.DataFrame,
    X_train: np.ndarray,
    path: pathlib.Path,
    steps: int = 9,
) -> None:
    vae_coords = vae_latent[["z1", "z2"]].to_numpy()
    pair = np.unravel_index(np.argmax(np.linalg.norm(vae_coords[:, None] - vae_coords[None, :], axis=2)), (len(vae_coords), len(vae_coords)))
    index_a, index_b = map(int, pair)
    t_values = np.linspace(0.0, 1.0, steps)
    ae_coords = ae_latent[["z1", "z2"]].to_numpy()
    trajectories = [
        ("AE determinista", ae_model, ae_coords),
        ("VAE", vae_model, vae_coords),
    ]
    fig, axes = plt.subplots(3, steps, figsize=(2 * steps, 6), dpi=160)
    axes[0, 0].imshow(X_train[index_a].reshape(16, 16, 3))
    axes[0, 1].imshow(X_train[index_b].reshape(16, 16, 3))
    axes[0, 0].set_ylabel("Extremo A")
    axes[0, 1].set_ylabel("Extremo B")
    for col in range(2, steps):
        axes[0, col].set_axis_off()
    for row, (label, model, coordinates) in enumerate(trajectories, start=1):
        z_a, z_b = coordinates[index_a], coordinates[index_b]
        z_path = (1 - t_values[:, None]) * z_a + t_values[:, None] * z_b
        samples = model.decode(z_path).reshape(steps, 16, 16, 3)
        for col, image in enumerate(samples):
            axes[row, col].imshow(image)
            axes[row, col].set_title(f"t={t_values[col]:.2f}", fontsize=8)
        axes[row, 0].set_ylabel(label)
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle("Interpolación entre las mismas dos muestras reales (latente 2D)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Graficar experimentos y espacio latente del VAE para sprites.")
    parser.add_argument("--summary", default="results/summary/runs.csv")
    parser.add_argument("--runs-dir", default="results/runs")
    parser.add_argument("--data", default="data/processed/sprites.npz")
    parser.add_argument("--output-dir", default="figures/vae")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    summary_path = pathlib.Path(args.summary)
    runs_dir = pathlib.Path(args.runs_dir)
    output_dir = pathlib.Path(args.output_dir)
    runs = _load_summary(summary_path, runs_dir)
    runs = runs[runs["run_id"].map(lambda run_id: (runs_dir / str(run_id) / "weights.npz").exists())]
    if runs.empty:
        parser.error("No se encontraron pesos VAE en --runs-dir.")

    runs["_latent_dim"] = runs["arch"].astype(str).str.extract(r"\[2x(\d+)\]").astype(float)
    runs["_warmup"] = runs["config_path"].map(lambda path: _config_value(path, "loss", "beta_warmup_epochs"))
    warmup_default = 50
    beta_groups = []
    for beta in sorted(runs["beta"].dropna().astype(float).unique()):
        selected = runs[
            np.isclose(runs["beta"].astype(float), beta)
            & (runs["_latent_dim"] == 16)
            & (runs["_warmup"] == warmup_default)
        ]
        if not selected.empty:
            beta_groups.append((f"β={beta:g}", selected))
    dim_groups = []
    for dim in sorted(runs["_latent_dim"].dropna().astype(int).unique()):
        selected = runs[
            np.isclose(runs["beta"].astype(float), 1.0)
            & (runs["_latent_dim"] == dim)
            & (runs["_warmup"] == warmup_default)
        ]
        if not selected.empty:
            dim_groups.append((f"z={dim}", selected))
    warmup_groups = []
    for warmup in (0, 50):
        selected = runs[
            np.isclose(runs["beta"].astype(float), 1.0)
            & (runs["_latent_dim"] == 16)
            & (runs["_warmup"] == warmup)
        ]
        if not selected.empty:
            warmup_groups.append((f"warmup={warmup}", selected))
    _plot_metric_curves(runs, runs_dir, beta_groups, output_dir / "beta_curves.png", "VAE: evolución por β (z=16, warmup=50)")
    _plot_metric_curves(runs, runs_dir, dim_groups, output_dir / "latent_dim_curves.png", "VAE: evolución por dimensión latente (β=1, warmup=50)")
    _plot_metric_curves(runs, runs_dir, warmup_groups, output_dir / "beta_warmup_curves.png", "VAE: comparación de warmup (β=1, z=16)")
    _plot_beta_tradeoff(runs, runs_dir, output_dir / "beta_tradeoff.png")

    archive_path = pathlib.Path(args.data)
    if not archive_path.exists():
        parser.error(f"No existe el dataset de sprites: {archive_path}")
    archive = np.load(archive_path, allow_pickle=True)
    _plot_dataset_samples(archive, output_dir / "dataset_samples.png", args.seed)
    vae_2d_candidates = runs[
        np.isclose(runs["beta"].astype(float), 1.0)
        & (runs["_latent_dim"] == 2)
        & (runs["_warmup"] == warmup_default)
    ]
    if vae_2d_candidates.empty:
        parser.error("Falta VAE de referencia con z=2, β=1 y warmup=50.")
    vae_row = vae_2d_candidates.sort_values("seed").iloc[0]
    vae_dir = _run_path(runs_dir, vae_row["run_id"])
    vae_model, _ = load_model_from_run(vae_dir)
    _plot_latent_posterior(vae_dir, archive_path, output_dir / "latent_posterior_2d.png")
    _plot_latent_manifold(vae_model, output_dir / "latent_manifold_2d.png")
    best_vae = _lowest_validation_row(
        runs[
            (runs["_warmup"] == warmup_default)
        ],
        runs_dir,
    )
    best_model, _ = load_model_from_run(_run_path(runs_dir, best_vae["run_id"]))
    _plot_reconstructions(best_model, archive, output_dir / "reconstructions.png", args.seed)

    ae_summary = pd.read_csv(summary_path)
    ae_candidates = ae_summary[
        (ae_summary["dataset"] == "sprites")
        & (ae_summary["model_type"] == "autoencoder")
        & ae_summary["config_path"].map(lambda path: _config_value(path, "model", "latent_dim") == 2)
        & (ae_summary["seed"].astype(int) == int(vae_row["seed"]))
        & ae_summary["status"].isin(["ok", "early_stopped"])
    ]
    ae_candidates = ae_candidates[ae_candidates["run_id"].map(lambda run_id: (runs_dir / str(run_id) / "weights.npz").exists())]
    if ae_candidates.empty:
        raise FileNotFoundError("No hay AE determinista de sprites con z=2 y semilla compatible. Ejecutá el sweep AE indicado en el plan.")
    ae_row = ae_candidates.iloc[0]
    ae_dir = _run_path(runs_dir, ae_row["run_id"])
    ae_model, _ = load_model_from_run(ae_dir)
    _plot_interpolation(
        ae_model,
        vae_model,
        load_latent_data(ae_dir),
        load_latent_data(vae_dir),
        archive["X_train"].astype(np.float64),
        output_dir / "ae_vae_interpolation.png",
    )
    print(f"Figuras del VAE guardadas en {output_dir}")


if __name__ == "__main__":
    main()
