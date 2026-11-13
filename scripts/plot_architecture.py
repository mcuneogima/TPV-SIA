"""
scripts/plot_architecture.py
============================
Genera un diagrama visual claro y estilizado de la arquitectura del Autoencoder
a partir de un archivo YAML de configuración o de una corrida entrenada.

Uso:
  python scripts/plot_architecture.py [--config configs/ae_basic.yaml] [--output figures/a2/architecture.png]
"""

import argparse
import pathlib
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tp5.train.config import load_config
from tp5.models.registry import build_model


def draw_architecture(cfg: dict, out_path: pathlib.Path):
    model_cfg = cfg.get("model", {})
    encoder_layers = model_cfg.get("encoder", {}).get("layers", [])
    decoder_layers = model_cfg.get("decoder", {}).get("layers", [])
    latent_dim = model_cfg.get("latent_dim", 2)
    init_name = model_cfg.get("init", "xavier")

    # Construir lista de capas: (nombre, unidades, activacion, rol)
    layers = [("Input", 35, "Vector 7x5", "input")]
    for idx, l in enumerate(encoder_layers):
        u = l["units"]
        act = l.get("activation", "identity")
        rol = "latent" if u == latent_dim and idx == len(encoder_layers) - 1 else "encoder"
        name = "Latent (z)" if rol == "latent" else f"Enc {idx+1}"
        layers.append((name, u, act, rol))

    for idx, l in enumerate(decoder_layers):
        u = l["units"]
        act = l.get("activation", "identity")
        rol = "output" if idx == len(decoder_layers) - 1 else "decoder"
        name = "Output (x̂)" if rol == "output" else f"Dec {idx+1}"
        layers.append((name, u, act, rol))

    # Crear figura
    fig, ax = plt.subplots(figsize=(14, 6), dpi=300)
    ax.set_xlim(-0.5, len(layers) + 0.5)
    ax.set_ylim(-1.5, 7.5)
    ax.axis("off")

    colors = {
        "input": "#E2E8F0",
        "encoder": "#BFDBFE",
        "latent": "#FDE047",
        "decoder": "#BBF7D0",
        "output": "#FED7AA",
    }
    edge_colors = {
        "input": "#64748B",
        "encoder": "#2563EB",
        "latent": "#CA8A04",
        "decoder": "#16A34A",
        "output": "#EA580C",
    }

    n_layers = len(layers)
    for i, (name, units, act, rol) in enumerate(layers):
        x = i + 0.5
        # Altura proporcional a sqrt(unidades) para que visualmente represente el cuello de botella
        h = max(1.2, min(5.5, 1.2 + 0.7 * (units ** 0.5)))
        w = 0.65
        y = 3.0 - h / 2.0

        # Caja de la capa
        rect = patches.FancyBboxPatch(
            (x - w / 2, y),
            w,
            h,
            boxstyle="round,pad=0.08,rounding_size=0.15",
            facecolor=colors[rol],
            edgecolor=edge_colors[rol],
            linewidth=2.0 if rol == "latent" else 1.5,
            zorder=3,
        )
        ax.add_patch(rect)

        # Texto interior
        ax.text(x, 3.0, f"{units} d", ha="center", va="center", fontsize=11, fontweight="bold", color="#1E293B", zorder=4)

        # Etiquetas arriba y abajo
        ax.text(x, y + h + 0.35, name, ha="center", va="bottom", fontsize=10, fontweight="bold", color="#0F172A")
        ax.text(x, y - 0.35, f"act: {act}", ha="center", va="top", fontsize=9, fontstyle="italic", color="#475569")

        # Flecha a la siguiente capa
        if i < n_layers - 1:
            next_x = i + 1 + 0.5
            ax.annotate(
                "",
                xy=(next_x - w / 2 - 0.05, 3.0),
                xytext=(x + w / 2 + 0.05, 3.0),
                arrowprops=dict(arrowstyle="->", color="#94A3B8", lw=2.0, mutation_scale=15),
                zorder=2,
            )

    # Título y metadatos
    model_type = model_cfg.get("type", "autoencoder").upper()
    arch_seq = "-".join(str(u) for _, u, _, _ in layers)
    fig.suptitle(
        f"Arquitectura del {model_type} con Cuello de Botella 2D\nSecuencia: {arch_seq}",
        fontsize=14,
        fontweight="bold",
        y=0.96,
    )
    
    # Subtítulo explicativo
    plt.figtext(
        0.5,
        0.05,
        f"Estructura simétrica de compresión no lineal a subespacio latente continuo bidimensional (z1, z2).\n"
        f"Inicialización: {init_name.upper()} | Salida: Sigmoide binaria (0/1) sobre patrón 7x5 (35 píxeles).",
        ha="center",
        fontsize=10,
        color="#334155",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0.08, 1, 0.92])
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Diagrama guardado en: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Graficar arquitectura de red.")
    parser.add_argument("--config", type=str, default="configs/ae_basic.yaml", help="Ruta al YAML base.")
    parser.add_argument("--output", type=str, default="figures/a2/architecture.png", help="Ruta del PNG de salida.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_path = pathlib.Path(args.output)
    draw_architecture(cfg, out_path)


if __name__ == "__main__":
    main()
