"""
scripts/generate.py
===================
Generación e interpolación de nuevos caracteres a partir del espacio latente (Consigna 1.a.4).
Soporta tres modos:
  (a) Punto específico manual (z1, z2) en zona vacía.
  (b) Punto medio entre dos caracteres conocidos: z_mid = (zA + zB) / 2.
  (c) Interpolación lineal en N pasos entre dos caracteres: z(t) = (1-t)*zA + t*zB.

Uso:
  python scripts/generate.py --mode all
  python scripts/generate.py --mode midpoint --char1 c --char2 o
  python scripts/generate.py --mode interpolate --char1 a --char2 z --steps 9
  python scripts/generate.py --mode point --z 0.0 0.0
"""

import argparse
import pathlib
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tp5.analysis.latent import load_model_from_run, load_latent_data
from tp5.analysis.generate import generate_from_point, generate_midpoint, interpolate_linear


def find_best_run(summary_csv: pathlib.Path) -> str:
    """Encuentra la mejor corrida en runs.csv."""
    if not summary_csv.exists():
        raise FileNotFoundError(f"No existe {summary_csv}")
    df = pd.read_csv(summary_csv)
    sorted_df = df.sort_values(by=["final_n_patterns_ok", "final_loss"], ascending=[False, True])
    return str(sorted_df.iloc[0]["run_id"])


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
    parser = argparse.ArgumentParser(description="Generar e interpolar letras nuevas desde el espacio latente.")
    parser.add_argument("--run", type=str, default=None, help="Directorio o ID de corrida. Usa la mejor por defecto.")
    parser.add_argument("--summary", type=str, default="results/summary/runs.csv", help="Ruta a runs.csv.")
    parser.add_argument("--mode", type=str, default="all", choices=["point", "midpoint", "interpolate", "all"], help="Modo de generación.")
    parser.add_argument("--z", nargs=2, type=float, default=[0.0, 0.0], help="Coordenadas z1 z2 para modo point.")
    parser.add_argument("--char1", type=str, default="c", help="Primer caracter para punto medio / interpolación.")
    parser.add_argument("--char2", type=str, default="o", help="Segundo caracter para punto medio / interpolación.")
    parser.add_argument("--steps", type=int, default=7, help="Cantidad de pasos para interpolación.")
    parser.add_argument("--output-dir", type=str, default="figures/a4", help="Directorio de figuras de salida.")
    args = parser.parse_args()

    out_dir = pathlib.Path(args.output_dir)

    # Identificar corrida
    if args.run is None:
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
    df_latent = load_latent_data(run_dir)
    run_title = f"Run: {run_id} | Arch: {model.arch_str()} | Pérdida: {cfg['loss']['recon'].upper()}"

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
