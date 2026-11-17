"""
scripts/generate.py
===================
Generación e interpolación de muestras a partir del espacio latente.
Soporta los modos:
  (a) Punto específico manual (z1, z2) en zona vacía.
  (b) Punto medio entre dos caracteres conocidos: z_mid = (zA + zB) / 2.
  (c) Interpolación lineal en N pasos entre dos caracteres: z(t) = (1-t)*zA + t*zB.
  (d) Muestreo de sprites desde el prior normal estándar del VAE.
  (e) Letra candidata comparada con todos los patrones de entrenamiento.

Uso:
  python scripts/generate.py --mode all
  python scripts/generate.py --mode midpoint --char1 c --char2 o
  python scripts/generate.py --mode interpolate --char1 a --char2 z --steps 9
  python scripts/generate.py --mode point --z 0.0 0.0
  python scripts/generate.py --mode sample --n-samples 64 --seed 42
  python scripts/generate.py --mode new-letter --z 7.3 1.5 --letter-name "H mayúscula"
"""

import argparse
import math
import pathlib
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tp5.analysis.latent import load_model_from_run, load_latent_data
from tp5.analysis.generate import generate_from_point, generate_midpoint, interpolate_linear
from tp5.data.loader import load_dataset


def find_best_run(summary_csv: pathlib.Path) -> str:
    """Encuentra la mejor corrida en runs.csv."""
    if not summary_csv.exists():
        raise FileNotFoundError(f"No existe {summary_csv}")
    df = pd.read_csv(summary_csv)
    sorted_df = df.sort_values(by=["final_n_patterns_ok", "final_loss"], ascending=[False, True])
    return str(sorted_df.iloc[0]["run_id"])


def find_best_vae_run(summary_csv: pathlib.Path, runs_dir: pathlib.Path) -> str:
    """Choose the sprite VAE with the lowest recorded validation reconstruction loss."""
    if not summary_csv.exists():
        raise FileNotFoundError(f"No existe {summary_csv}")
    runs = pd.read_csv(summary_csv)
    runs = runs[
        (runs["dataset"] == "sprites")
        & (runs["model_type"] == "vae")
        & runs["status"].isin(["ok", "early_stopped"])
    ].copy()
    runs = runs[runs["run_id"].map(lambda run_id: (runs_dir / str(run_id) / "epochs.csv").exists())]
    if runs.empty:
        raise FileNotFoundError("No hay corridas VAE válidas de sprites en el resumen.")

    def validation_loss(run_id: str) -> float:
        history = pd.read_csv(runs_dir / run_id / "epochs.csv")
        if "val_loss" not in history or history["val_loss"].dropna().empty:
            return float("inf")
        return float(history["val_loss"].dropna().iloc[-1])

    runs["selection_val_loss"] = runs["run_id"].map(validation_loss)
    selected = runs.sort_values(["selection_val_loss", "final_recon"], ascending=True).iloc[0]
    return str(selected["run_id"])


def plot_prior_samples(
    model,
    cfg: dict,
    run_id: str,
    output_dir: pathlib.Path,
    n_samples: int,
    seed: int,
    selected_sample: int,
):
    """Sample from N(0,I), plot an 8x8 grid, and record nearest training examples."""
    data = load_dataset(cfg)
    X_train = data["X_train"]
    shape = tuple(data["image_shape"])
    if not 0 <= selected_sample < n_samples:
        raise ValueError(f"--selected-sample debe estar entre 0 y {n_samples - 1}.")
    rng = np.random.default_rng(seed)
    samples = model.sample(n_samples, rng=rng)
    nearest = np.empty(n_samples, dtype=int)
    nearest_mse = np.empty(n_samples, dtype=np.float64)
    for index, sample in enumerate(samples):
        distances = np.mean((X_train - sample[None, :]) ** 2, axis=1)
        nearest[index] = int(np.argmin(distances))
        nearest_mse[index] = float(distances[nearest[index]])
    rows = pd.DataFrame({
        "sample_index": np.arange(n_samples),
        "nearest_train_index": nearest,
        "nearest_train_mse": nearest_mse,
    })

    columns = min(8, n_samples)
    grid_rows = math.ceil(n_samples / columns)
    fig, axes = plt.subplots(grid_rows, columns, figsize=(columns * 2, grid_rows * 2), dpi=160, squeeze=False)
    for index, axis in enumerate(axes.flat):
        axis.set_axis_off()
        if index >= n_samples:
            continue
        image = samples[index].reshape(shape)
        if len(shape) == 2:
            axis.imshow(image, cmap="gray", vmin=0, vmax=1)
        else:
            axis.imshow(np.clip(image, 0, 1))
        axis.set_title(f"#{index} · NN MSE={rows.iloc[index]['nearest_train_mse']:.3f}", fontsize=8)
    fig.suptitle(f"Muestras del prior N(0,I) · {run_id} · seed={seed}")
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / "prior_samples.png"
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)
    rows.to_csv(output_dir / "prior_samples_nearest.csv", index=False)
    np.savez_compressed(output_dir / "prior_samples.npz", samples=samples, nearest_train_index=nearest)
    selected = np.argsort(nearest_mse)[-min(8, n_samples):]
    comparison_fig, comparison_axes = plt.subplots(2, len(selected), figsize=(len(selected) * 2, 4), dpi=160, squeeze=False)
    for column, sample_index in enumerate(selected):
        source_image = samples[sample_index].reshape(shape)
        nearest_image = X_train[nearest[sample_index]].reshape(shape)
        for axis, image in ((comparison_axes[0, column], source_image), (comparison_axes[1, column], nearest_image)):
            if len(shape) == 2:
                axis.imshow(image, cmap="gray", vmin=0, vmax=1)
            else:
                axis.imshow(np.clip(image, 0, 1))
            axis.set_xticks([])
            axis.set_yticks([])
        comparison_axes[0, column].set_title(f"sample {sample_index}\nMSE={nearest_mse[sample_index]:.3f}", fontsize=8)
    comparison_axes[0, 0].set_ylabel("Generada")
    comparison_axes[1, 0].set_ylabel("Vecino train")
    comparison_fig.suptitle("Muestras con mayor distancia a su vecino de entrenamiento más cercano")
    comparison_fig.tight_layout()
    comparison_fig.savefig(output_dir / "prior_samples_nearest_comparison.png", bbox_inches="tight")
    plt.close(comparison_fig)
    selected_fig, selected_axes = plt.subplots(1, 2, figsize=(6, 3.5), dpi=180)
    for axis, image, title in (
        (selected_axes[0], samples[selected_sample].reshape(shape), f"Muestra generada #{selected_sample}"),
        (selected_axes[1], X_train[nearest[selected_sample]].reshape(shape), f"Vecino train #{nearest[selected_sample]}"),
    ):
        if len(shape) == 2:
            axis.imshow(image, cmap="gray", vmin=0, vmax=1)
        else:
            axis.imshow(np.clip(image, 0, 1))
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
    selected_axes[0].set_ylabel(f"MSE al vecino={nearest_mse[selected_sample]:.4f}")
    selected_fig.suptitle("Inspección cualitativa de una muestra del prior")
    selected_fig.tight_layout()
    selected_fig.savefig(output_dir / "prior_sample_selected.png", bbox_inches="tight")
    plt.close(selected_fig)
    print(f"Muestras generadas: {figure_path}")


def plot_point_generation(model, df_latent: pd.DataFrame, z_pt: np.ndarray, out_path: pathlib.Path, run_title: str):
    """Grafica la generación de un glifo a partir de una coordenada arbitraria y su ubicación en el espacio latente."""
    img_cont = generate_from_point(model, z_pt, binarize=False)
    img_bin = generate_from_point(model, z_pt, binarize=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=300, gridspec_kw={"width_ratios": [1, 1, 1.6]})

    # Glifo continuo
    im0 = axes[0].imshow(img_cont, cmap="binary", vmin=0, vmax=1)
    axes[0].set_title(f"Salida Continua\nz = ({z_pt[0]:.2f}, {z_pt[1]:.2f})", fontsize=11, fontweight="bold")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Glifo binarizado (umbral 0.5)
    axes[1].imshow(img_bin, cmap="binary", vmin=0, vmax=1)
    axes[1].set_title(f"Binarizado (umbral 0.5)\nz = ({z_pt[0]:.2f}, {z_pt[1]:.2f})", fontsize=11, fontweight="bold")
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    # Posición en el plano latente
    z1 = df_latent["z1"].values
    z2 = df_latent["z2"].values
    axes[2].scatter(z1, z2, c="#94A3B8", s=60, alpha=0.7, label="Dataset Font (32 letras)")
    axes[2].scatter([z_pt[0]], [z_pt[1]], c="#DC2626", s=180, marker="*", edgecolors="black", zorder=5, label="Punto Elegido")
    axes[2].set_xlabel("$z_1$")
    axes[2].set_ylabel("$z_2$")
    axes[2].set_title("Ubicación en el Espacio Latente", fontsize=11, fontweight="bold")
    axes[2].legend(loc="best")
    axes[2].grid(True, linestyle="--", alpha=0.4)

    fig.suptitle(f"Generación de Caracter desde Punto Manual\n{run_title}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generación manual guardada en: {out_path}")


def plot_new_letter(model, cfg: dict, df_latent: pd.DataFrame, z_pt: np.ndarray, out_path: pathlib.Path, name: str):
    """Compare a candidate letter with every training bitmap."""
    data = load_dataset(cfg)
    if data["image_shape"] != (7, 5) or data["labels"] is None:
        raise ValueError("La generación de letras requiere el dataset font de 7x5.")
    if len(data["labels"]) != len(data["X_train"]):
        raise ValueError("Las etiquetas deben corresponder al conjunto de entrenamiento.")
    continuous = generate_from_point(model, z_pt, binarize=False)
    if not np.isfinite(continuous).all():
        raise ValueError("El decoder produjo valores no finitos.")
    binary = continuous >= 0.5
    distances = np.count_nonzero(data["X_train"] != binary.ravel(), axis=1)
    nearest = int(np.argmin(distances))
    if distances[nearest] == 0:
        raise ValueError("La salida binarizada coincide con un patrón de entrenamiento.")

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.5), dpi=180, gridspec_kw={"width_ratios": [1, 1, 1, 1.8]})
    for axis, image, title in (
        (axes[0], continuous, "Salida continua"),
        (axes[1], binary, f"Candidata: {name}\nUmbral 0.5"),
        (axes[2], data["X_train"][nearest].reshape(7, 5), f"Train más cercano: {data['labels'][nearest]}\n{distances[nearest]} píxeles distintos"),
    ):
        axis.imshow(image, cmap="binary", vmin=0, vmax=1, interpolation="nearest")
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
    axes[3].scatter(df_latent["z1"], df_latent["z2"], s=35, alpha=0.6, label="Train")
    axes[3].scatter([z_pt[0]], [z_pt[1]], marker="*", s=180, color="red", label="Punto elegido")
    axes[3].set_xlabel("$z_1$")
    axes[3].set_ylabel("$z_2$")
    axes[3].legend()
    axes[3].grid(alpha=0.25)
    fig.suptitle(f"Decodificación desde z = ({z_pt[0]:.4f}, {z_pt[1]:.4f})")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Candidata guardada en {out_path}; vecino '{data['labels'][nearest]}', distancia {distances[nearest]} píxeles.")


def plot_midpoint_generation(model, df_latent: pd.DataFrame, c1: str, c2: str, out_path: pathlib.Path, run_title: str):
    """Grafica el punto medio entre dos caracteres conocidos junto a sus originales y trayectoria latente."""
    z_mid, img_mid_cont, z_a, z_b = generate_midpoint(model, df_latent, c1, c2, binarize=False)
    img_mid_bin = generate_from_point(model, z_mid, binarize=True)
    img_a = generate_from_point(model, z_a, binarize=True)
    img_b = generate_from_point(model, z_b, binarize=True)

    fig, axes = plt.subplots(1, 5, figsize=(18, 4.5), dpi=300, gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.8]})

    axes[0].imshow(img_a, cmap="binary", vmin=0, vmax=1)
    axes[0].set_title(f"Letra A: '{c1}'\nzA=({z_a[0]:.2f}, {z_a[1]:.2f})", fontsize=10, fontweight="bold")
    axes[0].set_xticks([]); axes[0].set_yticks([])

    axes[1].imshow(img_mid_cont, cmap="binary", vmin=0, vmax=1)
    axes[1].set_title(f"Punto Medio (Cont.)\nzMid=({z_mid[0]:.2f}, {z_mid[1]:.2f})", fontsize=10, fontweight="bold")
    axes[1].set_xticks([]); axes[1].set_yticks([])

    axes[2].imshow(img_mid_bin, cmap="binary", vmin=0, vmax=1)
    axes[2].set_title(f"Punto Medio (Bin.)\nUmbral 0.5", fontsize=10, fontweight="bold")
    axes[2].set_xticks([]); axes[2].set_yticks([])

    axes[3].imshow(img_b, cmap="binary", vmin=0, vmax=1)
    axes[3].set_title(f"Letra B: '{c2}'\nzB=({z_b[0]:.2f}, {z_b[1]:.2f})", fontsize=10, fontweight="bold")
    axes[3].set_xticks([]); axes[3].set_yticks([])

    # Plano latente con segmento entre A y B
    z1 = df_latent["z1"].values
    z2 = df_latent["z2"].values
    axes[4].scatter(z1, z2, c="#CBD5E1", s=50, alpha=0.6, label="Font Caracteres")
    axes[4].plot([z_a[0], z_b[0]], [z_a[1], z_b[1]], color="#2563EB", linestyle="--", lw=2.0, zorder=3)
    axes[4].scatter([z_a[0]], [z_a[1]], c="#16A34A", s=130, edgecolors="black", zorder=4, label=f"Letra '{c1}'")
    axes[4].scatter([z_b[0]], [z_b[1]], c="#9333EA", s=130, edgecolors="black", zorder=4, label=f"Letra '{c2}'")
    axes[4].scatter([z_mid[0]], [z_mid[1]], c="#DC2626", s=160, marker="*", edgecolors="black", zorder=5, label="Punto Medio")
    axes[4].set_xlabel("$z_1$")
    axes[4].set_ylabel("$z_2$")
    axes[4].set_title("Trayectoria en el Plano Latente", fontsize=11, fontweight="bold")
    axes[4].legend(loc="best", fontsize=9)
    axes[4].grid(True, linestyle="--", alpha=0.4)

    fig.suptitle(f"Generación por Punto Medio: '{c1}' ↔ '{c2}'\n{run_title}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Punto medio guardado en: {out_path}")


def plot_interpolation_generation(model, df_latent: pd.DataFrame, c1: str, c2: str, steps: int, out_path: pathlib.Path, run_title: str):
    """Grafica la tira de interpolación lineal en N pasos entre dos letras y la trayectoria en el espacio latente."""
    t_vals, z_traj, imgs_cont = interpolate_linear(model, df_latent, c1, c2, steps=steps, binarize=False)
    _, _, imgs_bin = interpolate_linear(model, df_latent, c1, c2, steps=steps, binarize=True)

    fig = plt.figure(figsize=(16, 7), dpi=300)
    gs = fig.add_gridspec(2, steps + 2)

    # Fila 1: Tira continua
    for i in range(steps):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(imgs_cont[i], cmap="binary", vmin=0, vmax=1)
        ax.set_title(f"t={t_vals[i]:.2f}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        if i == 0:
            ax.set_ylabel("Continuo", fontsize=11, fontweight="bold")

    # Fila 2: Tira binarizada
    for i in range(steps):
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(imgs_bin[i], cmap="binary", vmin=0, vmax=1)
        lbl = c1 if i == 0 else (c2 if i == steps - 1 else f"paso {i}")
        ax.set_title(f"{lbl}", fontsize=9, fontweight="bold" if (i == 0 or i == steps - 1) else "normal")
        ax.set_xticks([]); ax.set_yticks([])
        if i == 0:
            ax.set_ylabel("Binarizado", fontsize=11, fontweight="bold")

    # Subplot derecho: Recorrido en el espacio latente ocupando ambas filas
    ax_map = fig.add_subplot(gs[:, steps:])
    z1 = df_latent["z1"].values
    z2 = df_latent["z2"].values
    ax_map.scatter(z1, z2, c="#CBD5E1", s=60, alpha=0.6, label="Font Caracteres")
    
    # Trayectoria de interpolación
    ax_map.plot(z_traj[:, 0], z_traj[:, 1], color="#2563EB", lw=2.2, linestyle="-", zorder=3)
    ax_map.scatter(z_traj[:, 0], z_traj[:, 1], c=t_vals, cmap="coolwarm", s=90, edgecolors="black", zorder=4, label="Pasos de Interpolación")
    
    ax_map.annotate(c1, (z_traj[0, 0], z_traj[0, 1]), xytext=(-8, 8), textcoords="offset points", fontweight="bold", fontsize=11, color="#1D4ED8")
    ax_map.annotate(c2, (z_traj[-1, 0], z_traj[-1, 1]), xytext=(8, -8), textcoords="offset points", fontweight="bold", fontsize=11, color="#DC2626")

    ax_map.set_xlabel("$z_1$")
    ax_map.set_ylabel("$z_2$")
    ax_map.set_title("Recorrido Latente", fontsize=11, fontweight="bold")
    ax_map.legend(loc="best", fontsize=9)
    ax_map.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle(f"Interpolación Lineal en el Espacio Latente: '{c1}' → '{c2}' ({steps} pasos)\n{run_title}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Interpolación guardada en: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generar e interpolar muestras desde el espacio latente.")
    parser.add_argument("--run", type=str, default=None, help="Directorio o ID de corrida. Usa la mejor por defecto.")
    parser.add_argument("--summary", type=str, default="results/summary/runs.csv", help="Ruta a runs.csv.")
    parser.add_argument("--mode", type=str, default="all", choices=["point", "midpoint", "interpolate", "all", "sample", "new-letter"], help="Modo de generación.")
    parser.add_argument("--z", nargs=2, type=float, default=[0.0, 0.0], help="Coordenadas z1 z2 para modo point.")
    parser.add_argument("--char1", type=str, default="c", help="Primer caracter para punto medio / interpolación.")
    parser.add_argument("--char2", type=str, default="o", help="Segundo caracter para punto medio / interpolación.")
    parser.add_argument("--letter-name", default="letra candidata", help="Interpretación visual de la letra generada.")
    parser.add_argument("--steps", type=int, default=7, help="Cantidad de pasos para interpolación.")
    parser.add_argument("--output-dir", type=str, default="figures/a4", help="Directorio de figuras de salida.")
    parser.add_argument("--n-samples", type=int, default=64, help="Cantidad de muestras en modo sample (default: 64).")
    parser.add_argument("--seed", type=int, default=42, help="Semilla del muestreo generativo.")
    parser.add_argument("--selected-sample", type=int, default=23, help="Índice de muestra a inspeccionar (default: 23).")
    args = parser.parse_args()

    out_dir = pathlib.Path(args.output_dir)
    if args.n_samples < 1:
        parser.error("--n-samples debe ser positivo.")
    if not 0 <= args.selected_sample < args.n_samples:
        parser.error("--selected-sample debe estar dentro de --n-samples.")

    # Identificar corrida
    if args.run is None:
        if args.mode == "sample":
            run_id = find_best_vae_run(pathlib.Path(args.summary), pathlib.Path("results/runs"))
            run_dir = pathlib.Path("results/runs") / run_id
            print(f"Seleccionando VAE de sprites con menor val_loss: {run_id}")
        else:
            run_id = find_best_run(pathlib.Path(args.summary))
            run_dir = pathlib.Path("results/runs") / run_id
            print(f"Seleccionando automáticamente la mejor corrida: {run_id}")
    else:
        run_dir = pathlib.Path(args.run)
        if not run_dir.exists():
            run_dir = pathlib.Path("results/runs") / args.run
        if not run_dir.exists():
            print(f"Error: no existe corrida en {args.run}")
            sys.exit(1)
        run_id = run_dir.name

    model, cfg = load_model_from_run(run_dir)
    if args.mode == "sample":
        if cfg.get("model", {}).get("type") != "vae" or cfg.get("data", {}).get("dataset") != "sprites":
            parser.error("El modo sample requiere una corrida VAE entrenada sobre sprites.")
        plot_prior_samples(model, cfg, run_id, out_dir, args.n_samples, args.seed, args.selected_sample)
        return

    df_latent = load_latent_data(run_dir)
    run_title = f"Run: {run_id} | Arch: {model.arch_str()} | Pérdida: {cfg['loss']['recon'].upper()}"

    if args.mode == "new-letter":
        if cfg.get("data", {}).get("dataset") != "font":
            parser.error("El modo new-letter requiere una corrida entrenada con font.")
        plot_new_letter(model, cfg, df_latent, np.array(args.z), out_dir / "new_letter.png", args.letter_name)
        return

    if args.mode in ["point", "all"]:
        plot_point_generation(model, df_latent, np.array(args.z), out_dir / "generate_point.png", run_title)

    if args.mode in ["midpoint", "all"]:
        plot_midpoint_generation(model, df_latent, args.char1, args.char2, out_dir / f"generate_midpoint_{args.char1}_{args.char2}.png", run_title)
        if args.mode == "all":
            # Generar otro par representativo para análisis crítico en el informe
            plot_midpoint_generation(model, df_latent, "e", "f", out_dir / "generate_midpoint_e_f.png", run_title)

    if args.mode in ["interpolate", "all"]:
        plot_interpolation_generation(model, df_latent, args.char1, args.char2, args.steps, out_dir / f"generate_interpolation_{args.char1}_{args.char2}.png", run_title)
        if args.mode == "all":
            # Interpolación entre letras muy distintas
            plot_interpolation_generation(model, df_latent, "a", "z", 9, out_dir / "generate_interpolation_a_z.png", run_title)

    print(f"¡Generación e interpolaciones completadas exitosamente en {out_dir}!")


if __name__ == "__main__":
    main()
