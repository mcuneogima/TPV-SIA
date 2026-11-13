"""
scripts/plot_latent.py
======================
Grafica el espacio latente 2D del Autoencoder (Consigna 1.a.3).
Carga latent.csv, weights.npz y config_resolved.yaml para generar:
  1. Scatter 2D de las 32 letras etiquetadas con sus respectivos caracteres (figures/a3/latent_space.png).
  2. Mapa continuo de reconstrucción de la variedad latente (manifold grid) evaluando
     el decodificador sobre una grilla regular 2D (figures/a3/latent_manifold.png).

Uso:
  python scripts/plot_latent.py [--run results/runs/<run_id>] [--output-dir figures/a3]
"""

import argparse
import pathlib
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tp5.analysis.latent import load_model_from_run, load_latent_data, evaluate_decoder_grid


def find_best_run(summary_csv: pathlib.Path) -> str:
    """Encuentra la mejor corrida en runs.csv (máximos patrones OK y mínima pérdida)."""
    if not summary_csv.exists():
        raise FileNotFoundError(f"No existe {summary_csv}")
    df = pd.read_csv(summary_csv)
    # Filtrar corridas válidas
    sorted_df = df.sort_values(by=["final_n_patterns_ok", "final_loss"], ascending=[False, True])
    best_id = sorted_df.iloc[0]["run_id"]
    return str(best_id)


def plot_latent_scatter(df_latent: pd.DataFrame, out_path: pathlib.Path, run_title: str):
    """Grafica scatter 2D con etiquetas claras de cada caracter del dataset font."""
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

    z1 = df_latent["z1"].values
    z2 = df_latent["z2"].values
    labels = df_latent["label"].values

    # Puntos en el plano latente
    scatter = ax.scatter(z1, z2, c=np.arange(len(z1)), cmap="tab20", s=180, edgecolors="black", linewidth=1.2, alpha=0.9, zorder=3)

    # Anotar cada letra con una caja estética
    for i in range(len(z1)):
        lbl = labels[i]
        # Representación visible para caracteres especiales
        disp_lbl = lbl if lbl != " " else "SPC"
        ax.annotate(
            disp_lbl,
            (z1[i], z2[i]),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#94A3B8", alpha=0.85),
            zorder=4,
        )

    ax.axhline(0, color="gray", linestyle="--", lw=0.8, alpha=0.6, zorder=1)
    ax.axvline(0, color="gray", linestyle="--", lw=0.8, alpha=0.6, zorder=1)

    ax.set_xlabel("Coordenada Latente $z_1$", fontsize=12)
    ax.set_ylabel("Coordenada Latente $z_2$", fontsize=12)
    ax.set_title(f"Espacio Latente 2D del Autoencoder (32 Caracteres de Font)\n{run_title}", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Scatter del espacio latente guardado en: {out_path}")


def plot_latent_manifold(model, df_latent: pd.DataFrame, out_path: pathlib.Path, run_title: str, grid_steps: int = 16):
    """
    Evalúa el decoder en una grilla regular del plano 2D y dibuja los caracteres resultantes,
    superponiendo las posiciones reales de entrenamiento para evidenciar la continuidad del espacio.
    """
    z1 = df_latent["z1"].values
    z2 = df_latent["z2"].values

    margin_x = (z1.max() - z1.min()) * 0.15
    margin_y = (z2.max() - z2.min()) * 0.15

    x_min, x_max = z1.min() - margin_x, z1.max() + margin_x
    y_min, y_max = z2.min() - margin_y, z2.max() + margin_y

    grid_x, grid_y, decoded_images = evaluate_decoder_grid(
        model, x_min, x_max, y_min, y_max, steps=grid_steps, shape=(7, 5)
    )

    fig, ax = plt.subplots(figsize=(12, 10), dpi=300)

    # Dibujar cada celda de la grilla como un pequeño glifo de 7x5
    for i, gx in enumerate(grid_x):
        for j, gy in enumerate(grid_y):
            char_img = decoded_images[j, i]  # y en filas, x en cols
            # Normalizar para visualización: negro es 0, blanco es 1 (o invertido para fondo claro)
            imagebox = OffsetImage(char_img, cmap="binary", zoom=1.1, alpha=0.8)
            ab = AnnotationBbox(imagebox, (gx, gy), frameon=True, pad=0.08, bboxprops=dict(edgecolor="#CBD5E1", lw=0.6))
            ax.add_artist(ab)

    # Superponer los 32 puntos reales
    ax.scatter(z1, z2, color="#DC2626", s=130, edgecolors="white", linewidth=1.5, zorder=5, label="Datos de Entrenamiento")
    for k in range(len(z1)):
        lbl = df_latent.iloc[k]["label"]
        ax.annotate(
            lbl,
            (z1[k], z2[k]),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="#991B1B",
            zorder=6,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="#FEF2F2", edgecolor="#F87171", alpha=0.9),
        )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Coordenada Latente $z_1$", fontsize=12)
    ax.set_ylabel("Coordenada Latente $z_2$", fontsize=12)
    ax.set_title(f"Variedad Continua Latente y Reconstrucción del Decoder (Grilla {grid_steps}x{grid_steps})\n{run_title}", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Manifold latente guardado en: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Graficar espacio latente 2D y mapa continuo.")
    parser.add_argument("--run", type=str, default=None, help="Ruta o ID de corrida. Si no se indica, usa la mejor.")
    parser.add_argument("--summary", type=str, default="results/summary/runs.csv", help="Ruta a runs.csv.")
    parser.add_argument("--output-dir", type=str, default="figures/a3", help="Directorio para guardar gráficos.")
    args = parser.parse_args()

    out_dir = pathlib.Path(args.output_dir)

    # Determinar directorio de la corrida
    if args.run is None:
        run_id = find_best_run(pathlib.Path(args.summary))
        run_dir = pathlib.Path("results/runs") / run_id
        print(f"Seleccionando automáticamente la mejor corrida: {run_id}")
    else:
        run_dir = pathlib.Path(args.run)
        if not run_dir.exists():
            # Buscar en results/runs/
            run_dir = pathlib.Path("results/runs") / args.run
        if not run_dir.exists():
            print(f"Error: no existe el directorio de corrida {args.run}")
            sys.exit(1)
        run_id = run_dir.name

    print(f"Cargando modelo y pesos de: {run_dir}")
    model, cfg = load_model_from_run(run_dir)
    df_latent = load_latent_data(run_dir)

    run_title = f"Run: {run_id} | Arch: {model.arch_str()} | Pérdida: {cfg['loss']['recon'].upper()}"

    print("Generando scatter 2D del espacio latente...")
    plot_latent_scatter(df_latent, out_dir / "latent_space.png", run_title)

    print("Generando mapa de variedad continua (decoder manifold grid)...")
    plot_latent_manifold(model, df_latent, out_dir / "latent_manifold.png", run_title, grid_steps=16)

    print(f"¡Gráficos de la Parte 1.a.3 generados exitosamente en {out_dir}!")


if __name__ == "__main__":
    main()
