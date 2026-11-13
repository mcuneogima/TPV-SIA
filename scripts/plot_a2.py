"""
scripts/plot_a2.py
==================
Genera los gráficos del estudio de screening de hiperparámetros (Consigna 1.a.2)
siguiendo la metodología One-Factor-At-A-Time (OFAT).

Lee results/summary/runs.csv y los epochs.csv correspondientes para generar:
  1. Curvas por época agrupadas por familias de hiperparámetros (media y banda +- 1 desvío estándar entre semillas):
     - Familia 1: Funciones de activación (tanh, relu, leakyrelu, sigmoid)
     - Familia 2: Arquitectura y profundidad (small, medium, deep)
     - Familia 3: Función de pérdida (BCE vs MSE)
     - Familia 4: Tamaño de batch (mini-batch 8 vs full-batch 32)
     - Familia 5: Esquema de inicialización (Xavier, He, Normal Small)
     - Familia 6: Optimizadores (SGD, Momentum, RMSprop, Adam)
     - Familia 7: Learning Rate (1e-4, 1e-3, 1e-2)
     - Familia 8: Schedules de LR (exp_decay vs constant)
  2. Gráficos de distribución (boxplots) por cada factor OFAT sin variables de confusión.
  3. Ranking visual impecable de las mejores configuraciones (top performers), con textos alineados sin solapamientos.

Uso:
  python scripts/plot_a2.py [--summary results/summary/runs.csv] [--output-dir figures/a2]
"""

import argparse
import pathlib
import sys
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
})

PALETTE = ["#2563EB", "#DC2626", "#16A34A", "#9333EA", "#EA580C", "#0D9488", "#CA8A04", "#4F46E5"]

# Configuración base para el análisis One-Factor-At-A-Time
DEFAULT_BASELINE = {
    "optimizer": "adam",
    "lr0": 0.001,
    "schedule": "exp_decay",
    "recon_loss_fn": "bce",
    "init": "xavier",
    "arch": "35-24-12-2-12-24-35",
    "activation_hidden": "tanh",
    "batch_size": "full",
}


def load_epochs_for_run(run_dir: pathlib.Path) -> Optional[pd.DataFrame]:
    """Carga epochs.csv de una corrida dada."""
    csv_path = run_dir / "epochs.csv"
    if not csv_path.exists():
        return None
    try:
        return pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error leyendo {csv_path}: {e}")
        return None


def get_ofat_subset(df: pd.DataFrame, factor_col: str, baseline: Dict[str, any]) -> pd.DataFrame:
    """
    Filtra las corridas donde todos los hiperparámetros excepto factor_col coinciden con la configuración base.
    Deduplica por (factor_col, seed) para evitar duplicación del baseline.
    """
    cond = pd.Series(True, index=df.index)
    for other_col, val in baseline.items():
        if other_col != factor_col:
            cond = cond & (df[other_col] == val)
    sub = df[cond].copy()
    sub = sub.drop_duplicates(subset=[factor_col, "seed"])
    return sub


def aggregate_curves(
    runs_subset: pd.DataFrame,
    base_runs_dir: pathlib.Path,
    group_col: str,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Agrupa corridas según group_col y calcula la curva promedio y desviación estándar
    para 'loss', 'n_patterns_ok' y 'mean_pixel_errors' a lo largo de las épocas.
    Maneja correctamente las corridas con early-stopping mediante forward-fill de su estado convergido,
    garantizando que el eje temporal sea completo y continuo para todas las variantes.
    """
    # 1. Cargar todos los dataframes de épocas y recolectar el conjunto universal de épocas
    grouped_dfs = {}
    all_epochs_set = set()

    for group_val, group_df in runs_subset.groupby(group_col):
        epoch_dfs = []
        for _, row in group_df.iterrows():
            rdir = base_runs_dir / str(row["run_id"])
            edf = load_epochs_for_run(rdir)
            if edf is not None and not edf.empty:
                epoch_dfs.append(edf)
                all_epochs_set.update(edf["epoch"].values)
        if epoch_dfs:
            grouped_dfs[str(group_val)] = epoch_dfs

    if not all_epochs_set:
        return {}

    all_epochs = np.sort(np.array(list(all_epochs_set)))

    # 2. Reindexar con forward-fill para cada grupo
    results = {}
    for group_val, epoch_dfs in grouped_dfs.items():
        loss_matrix = np.zeros((len(epoch_dfs), len(all_epochs)))
        ok_matrix = np.zeros((len(epoch_dfs), len(all_epochs)))
        err_matrix = np.zeros((len(epoch_dfs), len(all_epochs)))

        for i, edf in enumerate(epoch_dfs):
            # Reindexar sobre el eje maestro de épocas con forward-fill:
            # Si una corrida se detuvo por early-stopping, su estado final óptimo se mantiene constante hasta el final.
            reindexed = edf.set_index("epoch").reindex(all_epochs).ffill().bfill()
            loss_matrix[i, :] = reindexed["loss"].values
            ok_matrix[i, :] = reindexed["n_patterns_ok"].values
            err_matrix[i, :] = reindexed["mean_pixel_errors"].values

        results[group_val] = {
            "epochs": all_epochs,
            "loss_mean": np.mean(loss_matrix, axis=0),
            "loss_std": np.std(loss_matrix, axis=0),
            "ok_mean": np.mean(ok_matrix, axis=0),
            "ok_std": np.std(ok_matrix, axis=0),
            "err_mean": np.mean(err_matrix, axis=0),
            "err_std": np.std(err_matrix, axis=0),
            "n_seeds": len(epoch_dfs),
        }
    return results


def plot_family_evolution(
    agg_data: Dict[str, Dict[str, np.ndarray]],
    title_prefix: str,
    family_label: str,
    out_path: pathlib.Path,
    metric_right: str = "ok",
    log_loss: bool = True,
):
    """Genera una figura de 2 paneles: evolución de Loss (izq) y Métrica de Calidad (der)."""
    fig, (ax_loss, ax_metric) = plt.subplots(1, 2, figsize=(15, 5.5), dpi=300)

    for idx, (label, data) in enumerate(agg_data.items()):
        color = PALETTE[idx % len(PALETTE)]
        epochs = data["epochs"]

        # Panel izquierdo: Pérdida
        l_mean = data["loss_mean"]
        l_std = data["loss_std"]
        ax_loss.plot(epochs, l_mean, label=f"{label} (N={data['n_seeds']})", color=color, lw=2.0)
        ax_loss.fill_between(epochs, np.maximum(1e-6, l_mean - l_std), l_mean + l_std, color=color, alpha=0.18)

        # Panel derecho: Métrica de aprendizaje
        if metric_right == "ok":
            m_mean = data["ok_mean"]
            m_std = data["ok_std"]
            ax_metric.plot(epochs, m_mean, label=f"{label}", color=color, lw=2.0)
            ax_metric.fill_between(epochs, np.maximum(0, m_mean - m_std), np.minimum(32, m_mean + m_std), color=color, alpha=0.18)
        else:
            m_mean = data["err_mean"]
            m_std = data["err_std"]
            ax_metric.plot(epochs, m_mean, label=f"{label}", color=color, lw=2.0)
            ax_metric.fill_between(epochs, np.maximum(0, m_mean - m_std), m_mean + m_std, color=color, alpha=0.18)

    ax_loss.set_xlabel("Épocas")
    ax_loss.set_ylabel("Pérdida de Reconstrucción" + (" (escala log)" if log_loss else ""))
    if log_loss:
        ax_loss.set_yscale("log")
    ax_loss.set_title("Evolución de la Función de Pérdida")
    ax_loss.legend(title=family_label, loc="best", framealpha=0.9)

    ax_metric.set_xlabel("Épocas")
    if metric_right == "ok":
        ax_metric.set_ylabel("Patrones Aprendidos OK (error ≤ 1 px)")
        ax_metric.set_ylim(-0.5, 33)
        ax_metric.axhline(32, color="gray", linestyle=":", lw=1.5, label="Objetivo Total (32/32)")
        ax_metric.set_title("Patrones Aprendidos por Época")
    else:
        ax_metric.set_ylabel("Error Promedio de Píxeles por Patrón")
        ax_metric.set_title("Error Medio de Píxeles por Época")
    ax_metric.legend(title=family_label, loc="best", framealpha=0.9)

    fig.suptitle(f"Familia OFAT: {title_prefix} (Media y Desvío Estándar entre Semillas)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Gráfico guardado: {out_path}")


def generate_all_family_plots(df: pd.DataFrame, runs_dir: pathlib.Path, out_dir: pathlib.Path, baseline: Dict[str, any]):
    """Genera las curvas de evolución para todos los factores OFAT."""
    # 1. Activación
    sub_act = get_ofat_subset(df, "activation_hidden", baseline)
    agg_act = aggregate_curves(sub_act, runs_dir, "activation_hidden")
    plot_family_evolution(
        agg_act,
        title_prefix="Funciones de Activación Ocultas (Base: Adam, BCE, Xavier, Full)",
        family_label="Activación",
        out_path=out_dir / "family_activation.png",
    )

    # 2. Arquitectura
    sub_arch = get_ofat_subset(df, "arch", baseline)
    arch_map = {
        "35-16-2-12-24-35": "Small (35-16-2)",
        "35-24-12-2-12-24-35": "Medium (35-24-12-2)",
        "35-30-20-10-2-12-24-35": "Deep (35-30-20-10-2)",
    }
    sub_arch["arch_label"] = sub_arch["arch"].map(lambda x: arch_map.get(x, x))
    agg_arch = aggregate_curves(sub_arch, runs_dir, "arch_label")
    plot_family_evolution(
        agg_arch,
        title_prefix="Arquitecturas de Red (Base: Tanh, Adam, BCE, Xavier, Full)",
        family_label="Arquitectura",
        out_path=out_dir / "family_architecture.png",
    )

    # 3. Función de pérdida
    sub_loss = get_ofat_subset(df, "recon_loss_fn", baseline)
    sub_loss["loss_label"] = sub_loss["recon_loss_fn"].str.upper()
    agg_loss = aggregate_curves(sub_loss, runs_dir, "loss_label")
    plot_family_evolution(
        agg_loss,
        title_prefix="Función de Pérdida: BCE vs MSE (Base: Med, Tanh, Adam, Xavier, Full)",
        family_label="Pérdida",
        out_path=out_dir / "family_loss_function.png",
        metric_right="ok",
        log_loss=True,
    )
    plot_family_evolution(
        agg_loss,
        title_prefix="Error Promedio de Píxeles: BCE vs MSE",
        family_label="Pérdida",
        out_path=out_dir / "family_loss_pixel_errors.png",
        metric_right="error",
        log_loss=True,
    )

    # 4. Tamaño de batch
    sub_bs = get_ofat_subset(df, "batch_size", baseline)
    sub_bs["bs_label"] = sub_bs["batch_size"].map(lambda b: "Mini-batch (8)" if str(b) == "8" else "Full-batch (32)")
    agg_bs = aggregate_curves(sub_bs, runs_dir, "bs_label")
    plot_family_evolution(
        agg_bs,
        title_prefix="Tamaño de Batch: Mini-batch (8) vs Full-batch (Base: Med, Tanh, Adam, BCE)",
        family_label="Batch Size",
        out_path=out_dir / "family_batch_size.png",
    )

    # 5. Inicialización
    sub_init = get_ofat_subset(df, "init", baseline)
    sub_init["init_label"] = sub_init["init"].str.capitalize()
    agg_init = aggregate_curves(sub_init, runs_dir, "init_label")
    plot_family_evolution(
        agg_init,
        title_prefix="Inicialización de Pesos: Xavier vs He vs Normal Small (Control Negativo)",
        family_label="Inicialización",
        out_path=out_dir / "family_initialization.png",
    )

    # 6. Optimizadores
    sub_opt = get_ofat_subset(df, "optimizer", baseline)
    sub_opt["opt_label"] = sub_opt["optimizer"].str.upper()
    agg_opt = aggregate_curves(sub_opt, runs_dir, "opt_label")
    plot_family_evolution(
        agg_opt,
        title_prefix="Optimizadores: SGD vs Momentum vs RMSprop vs Adam (Base: lr=0.001, Med)",
        family_label="Optimizador",
        out_path=out_dir / "family_optimizer.png",
    )

    # 7. Learning Rate
    sub_lr = get_ofat_subset(df, "lr0", baseline)
    sub_lr["lr_label"] = sub_lr["lr0"].map(lambda l: f"lr = {l}")
    agg_lr = aggregate_curves(sub_lr, runs_dir, "lr_label")
    plot_family_evolution(
        agg_lr,
        title_prefix="Tasa de Aprendizaje Inicial: lr0 en Adam (1e-4, 1e-3, 1e-2)",
        family_label="Learning Rate",
        out_path=out_dir / "family_lr.png",
    )

    # 8. Schedule
    sub_sched = get_ofat_subset(df, "schedule", baseline)
    sub_sched["sched_label"] = sub_sched["schedule"].str.capitalize()
    agg_sched = aggregate_curves(sub_sched, runs_dir, "sched_label")
    plot_family_evolution(
        agg_sched,
        title_prefix="Esquema de Tasa de Aprendizaje: Exp Decay vs Constante",
        family_label="Schedule",
        out_path=out_dir / "family_schedule.png",
    )


def generate_global_boxplots(df: pd.DataFrame, out_path: pathlib.Path, baseline: Dict[str, any]):
    """Genera boxplots estadísticos de Patrones OK finales para cada factor OFAT sin variables de confusión."""
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), dpi=300)

    factors_spec = [
        ("optimizer", "Optimizador", 0, 0, "#BFDBFE", lambda x: str(x).upper()),
        ("lr0", "Learning Rate", 0, 1, "#BBF7D0", lambda x: f"{x}"),
        ("schedule", "Schedule LR", 0, 2, "#FED7AA", lambda x: str(x).capitalize()),
        ("recon_loss_fn", "Pérdida", 0, 3, "#FDE047", lambda x: str(x).upper()),
        ("init", "Inicialización", 1, 0, "#DDD6FE", lambda x: str(x).capitalize()),
        ("arch", "Arquitectura", 1, 1, "#FBCFE8", lambda x: {"35-16-2-12-24-35": "Small", "35-24-12-2-12-24-35": "Med", "35-30-20-10-2-12-24-35": "Deep"}.get(x, x)),
        ("activation_hidden", "Activación", 1, 2, "#A7F3D0", lambda x: str(x)),
        ("batch_size", "Batch Size", 1, 3, "#E2E8F0", lambda x: "Full (32)" if str(x) == "full" else "Mini (8)"),
    ]

    for col, title, r, c, bg_color, label_fn in factors_spec:
        ax = axes[r, c]
        sub = get_ofat_subset(df, col, baseline)
        sub["mapped_label"] = sub[col].map(label_fn)

        groups = [g["final_n_patterns_ok"].values for _, g in sub.groupby("mapped_label")]
        labels = [k for k, _ in sub.groupby("mapped_label")]

        ax.boxplot(groups, tick_labels=labels, patch_artist=True, boxprops=dict(facecolor=bg_color))
        ax.set_title(title, fontweight="bold")
        ax.set_ylim(-1, 33)
        ax.axhline(32, color="gray", linestyle=":", lw=1.2, alpha=0.7)
        if c == 0:
            ax.set_ylabel("Patrones OK finales (≤ 1 px)")

    fig.suptitle("Sensibilidad One-Factor-At-A-Time (OFAT) por Familia de Hiperparámetros", fontsize=15, fontweight="bold")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Boxplots OFAT guardados: {out_path}")


def generate_top_performers_chart(df: pd.DataFrame, out_path: pathlib.Path, top_n: int = 10):
    """
    Genera gráfico de barras horizontales con las mejores configuraciones del screening.
    Alinea las etiquetas numéricas limpiamente a la derecha del gráfico en una columna vertical dedicada,
    evitando superposiciones con barras de error y con la línea de meta.
    """
    cfg_all_cols = ["arch", "activation_hidden", "recon_loss_fn", "optimizer", "lr0", "schedule", "init", "batch_size"]
    agg = df.groupby(cfg_all_cols).agg(
        mean_ok=("final_n_patterns_ok", "mean"),
        std_ok=("final_n_patterns_ok", "std"),
        mean_loss=("final_loss", "mean"),
        n_seeds=("seed", "count"),
    ).reset_index()

    agg = agg.sort_values(by=["mean_ok", "mean_loss"], ascending=[False, True]).head(top_n)

    labels = []
    arch_short = {"35-16-2-12-24-35": "Small", "35-24-12-2-12-24-35": "Med", "35-30-20-10-2-12-24-35": "Deep"}
    for _, r in agg.iterrows():
        a = arch_short.get(r["arch"], r["arch"])
        bs = f"b={r['batch_size']}"
        sched = "exp" if r["schedule"] == "exp_decay" else "const"
        labels.append(f"{a} | {r['activation_hidden']} | {r['recon_loss_fn'].upper()} | {r['optimizer']} (lr={r['lr0']}, {sched}) | {r['init']} | {bs}")

    fig, ax = plt.subplots(figsize=(15, 7.0), dpi=300)
    y_pos = np.arange(len(labels))
    means = agg["mean_ok"].values
    stds = agg["std_ok"].fillna(0.0).values

    # Limitar físicamente las barras de error al rango admisible [0, 32]
    xerr_low = np.minimum(means, stds)
    xerr_high = np.minimum(32.0 - means, stds)

    bars = ax.barh(y_pos, means, xerr=[xerr_low, xerr_high], align="center", color="#3B82F6", edgecolor="#1D4ED8", alpha=0.85, capsize=4)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()  # Mejor arriba
    ax.set_xlabel("Patrones OK promedio (≤ 1 píxel de error)", fontsize=11)
    
    # Expandir límite horizontal para que los números a la derecha tengan columna dedicada
    ax.set_xlim(0, 42.5)
    ax.axvline(32, color="#DC2626", linestyle="--", lw=1.6, label="Objetivo Total (32/32)")

    # Alineación vertical perfecta a la derecha de la línea objetivo (x = 32.8)
    for i, bar in enumerate(bars):
        w = bar.get_width()
        loss_val = agg.iloc[i]["mean_loss"]
        ax.text(
            32.8,
            bar.get_y() + bar.get_height() / 2,
            f"{w:.1f}/32   (loss: {loss_val:.3f})",
            va="center",
            ha="left",
            fontsize=9.5,
            fontweight="bold",
            color="#0F172A",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#F8FAFC", edgecolor="#CBD5E1", alpha=0.9),
        )

    ax.set_title(f"Top {top_n} Configuraciones del Screening (Promedio entre Semillas)", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", framealpha=0.95)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Top performers guardado: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generar gráficos del screening de hiperparámetros OFAT (1.a.2).")
    parser.add_argument("--summary", type=str, default="results/summary/runs.csv", help="Ruta al CSV runs.csv.")
    parser.add_argument("--runs-dir", type=str, default="results/runs", help="Directorio con las carpetas de corrida.")
    parser.add_argument("--output-dir", type=str, default="figures/a2", help="Directorio de salida para figuras.")
    args = parser.parse_args()

    summary_path = pathlib.Path(args.summary)
    runs_dir = pathlib.Path(args.runs_dir)
    out_dir = pathlib.Path(args.output_dir)

    if not summary_path.exists():
        print(f"Error: no existe el archivo de resumen {summary_path}")
        sys.exit(1)

    print("Cargando resultados de corridas...")
    df = pd.read_csv(summary_path)
    df["batch_size"] = df["batch_size"].astype(str)
    print(f"Total de corridas cargadas: {len(df)}")

    # Detectar o aplicar baseline
    baseline = DEFAULT_BASELINE.copy()
    print(f"Línea de base OFAT: {baseline}")

    print("\n1. Generando curvas de evolución por familias de hiperparámetros OFAT...")
    generate_all_family_plots(df, runs_dir, out_dir, baseline)

    print("\n2. Generando análisis de sensibilidad por factores (boxplots)...")
    generate_global_boxplots(df, out_dir / "boxplot_families.png", baseline)

    print("\n3. Generando ranking de mejores configuraciones...")
    generate_top_performers_chart(df, out_dir / "top_performers.png", top_n=10)

    print(f"\n¡Todos los gráficos de la Fase 3 parte 1.a.2 fueron generados exitosamente en {out_dir}!")


if __name__ == "__main__":
    main()
