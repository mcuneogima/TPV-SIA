"""Rank 1.a.2 autoencoders, train low-noise DAEs, and compare robustness."""

import argparse
import json
import pathlib
import sys
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tp5.analysis.latent import load_model_from_run
from tp5.analysis.noise_eval import evaluate_noise_levels, reconstruct
from tp5.data.noise import corrupt
from tp5.train.io import generate_run_id
from tp5.train.trainer import Trainer


CONFIG_COLUMNS = [
    "arch",
    "activation_hidden",
    "recon_loss_fn",
    "optimizer",
    "lr0",
    "schedule",
    "init",
    "batch_size",
]


def _parse_levels(value: str, name: str) -> List[float]:
    try:
        levels = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} debe ser una lista de números.") from exc
    if not levels or any(level < 0.0 or level > 1.0 for level in levels):
        raise argparse.ArgumentTypeError(f"{name} debe contener niveles en [0, 1].")
    return levels


def _run_path(runs_dir: pathlib.Path, run_id: str) -> pathlib.Path:
    path = runs_dir / str(run_id)
    if not path.is_dir():
        raise FileNotFoundError(f"No existe la corrida {run_id}: {path}")
    return path


def _evaluate_candidates(
    rows: pd.DataFrame,
    runs_dir: pathlib.Path,
    X_clean: np.ndarray,
    levels: List[float],
    trials: int,
    kinds: Tuple[str, ...],
    seed_offset: int,
) -> pd.DataFrame:
    records = []
    for row_index, row in rows.reset_index(drop=True).iterrows():
        run_id = str(row["run_id"])
        run_dir = _run_path(runs_dir, run_id)
        model, _ = load_model_from_run(run_dir)
        for kind_index, kind in enumerate(kinds):
            evaluations = evaluate_noise_levels(
                model,
                X_clean,
                noise_kind=kind,
                levels=levels,
                n_trials=trials,
                seed=seed_offset + row_index * 1009 + kind_index,
            )
            for result in evaluations:
                records.append({
                    "run_id": run_id,
                    "model_type": str(row.get("model_type", "autoencoder")),
                    "kind": result["kind"],
                    **{column: row.get(column) for column in CONFIG_COLUMNS},
                    **result,
                })
    return pd.DataFrame(records)


def _rank_candidates(low_results: pd.DataFrame, top_n: int) -> pd.DataFrame:
    scores = low_results.groupby(["run_id"] + CONFIG_COLUMNS, dropna=False).agg(
        low_noise_frac_ok=("frac_ok", "mean"),
        low_noise_mean_pixel_errors=("mean_pixel_errors", "mean"),
    ).reset_index()
    ranked = scores.groupby(CONFIG_COLUMNS, dropna=False).agg(
        low_noise_frac_ok=("low_noise_frac_ok", "mean"),
        low_noise_std=("low_noise_frac_ok", "std"),
        low_noise_mean_pixel_errors=("low_noise_mean_pixel_errors", "mean"),
        n_seeds=("run_id", "nunique"),
        source_run_ids=("run_id", lambda values: json.dumps(sorted(set(values)))),
    ).reset_index()
    ranked["low_noise_std"] = ranked["low_noise_std"].fillna(0.0)
    ranked = ranked.sort_values(
        ["low_noise_frac_ok", "low_noise_mean_pixel_errors", "low_noise_std"],
        ascending=[False, True, True],
    ).head(top_n).reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    return ranked


def _candidate_key(row: pd.Series) -> Tuple[object, ...]:
    return tuple(row[column] for column in CONFIG_COLUMNS)


def _train_daes(
    ranked: pd.DataFrame,
    source_rows: pd.DataFrame,
    source_runs_dir: pathlib.Path,
    output_dir: pathlib.Path,
    train_level: float,
    epochs: int | None,
    include_vae: bool,
) -> List[Dict[str, str]]:
    run_dir = output_dir / "runs"
    summary_dir = output_dir / "summary"
    ranked_by_key = {_candidate_key(row): int(row["rank"]) for _, row in ranked.iterrows()}
    pairs = []

    for _, row in source_rows.iterrows():
        rank = ranked_by_key.get(_candidate_key(row))
        if rank is None:
            continue
        source_run_id = str(row["run_id"])
        source_path = _run_path(source_runs_dir, source_run_id)
        with (source_path / "config_resolved.yaml").open("r", encoding="utf-8") as file:
            cfg = yaml.safe_load(file)
        cfg["run_name"] = f"dae_b1_top{rank:02d}_{cfg.get('run_name', 'run')}"
        cfg["model"]["type"] = "denoising"
        cfg["noise"] = {
            **cfg.get("noise", {}),
            "kind": "flip",
            "level": train_level,
            "regenerate_each_epoch": True,
        }
        cfg["output"] = {
            **cfg.get("output", {}),
            "dir": str(run_dir),
            "summary_dir": str(summary_dir),
            "save_weights": True,
        }
        if epochs is not None:
            cfg["train"]["epochs"] = epochs

        dae_run_id = generate_run_id(cfg)
        dae_path = run_dir / dae_run_id
        if not (dae_path / "weights.npz").exists():
            print(f"[DAE] Entrenando top {rank:02d}, seed={cfg.get('seed')}")
            Trainer(cfg).train()
        else:
            print(f"[DAE] Reutilizando corrida existente: {dae_run_id}")
        pair = {
            "rank": str(rank),
            "source_run_id": source_run_id,
            "dae_run_id": dae_run_id,
            "seed": str(cfg.get("seed")),
        }

        if include_vae:
            vae_cfg = yaml.safe_load((source_path / "config_resolved.yaml").read_text(encoding="utf-8"))
            vae_cfg["run_name"] = f"vae_b1_top{rank:02d}_{vae_cfg.get('run_name', 'run')}"
            vae_cfg["model"]["type"] = "vae"
            encoder_layers = vae_cfg["model"]["encoder"]["layers"]
            if (
                encoder_layers
                and int(encoder_layers[-1]["units"]) == int(vae_cfg["model"]["latent_dim"])
                and encoder_layers[-1].get("activation") == "identity"
            ):
                vae_cfg["model"]["encoder"]["layers"] = encoder_layers[:-1]
            vae_cfg["loss"]["beta_warmup_epochs"] = max(
                int(vae_cfg["loss"].get("beta_warmup_epochs", 0)), 100
            )
            vae_cfg["noise"] = {
                **vae_cfg.get("noise", {}),
                "kind": "flip",
                "level": train_level,
                "regenerate_each_epoch": True,
            }
            vae_cfg["output"] = {
                **vae_cfg.get("output", {}),
                "dir": str(run_dir),
                "summary_dir": str(summary_dir),
                "save_weights": True,
            }
            if epochs is not None:
                vae_cfg["train"]["epochs"] = epochs
            vae_run_id = generate_run_id(vae_cfg)
            vae_path = run_dir / vae_run_id
            if not (vae_path / "weights.npz").exists():
                print(f"[VAE] Entrenando referencia top {rank:02d}, seed={vae_cfg.get('seed')}")
                Trainer(vae_cfg).train()
            else:
                print(f"[VAE] Reutilizando corrida existente: {vae_run_id}")
            pair["vae_run_id"] = vae_run_id
        pairs.append(pair)
    return pairs


def _save_ranking_plot(ranked: pd.DataFrame, path: pathlib.Path) -> None:
    if ranked.empty:
        return
    labels = [
        f"#{int(row.rank)} {row.arch} | {row.activation_hidden} | {row.optimizer} | "
        f"{row.recon_loss_fn.upper()} | seed runs={int(row.n_seeds)}"
        for row in ranked.itertuples()
    ]
    fig, ax = plt.subplots(figsize=(13, max(4.5, len(ranked) * 0.42)), dpi=180)
    ax.barh(np.arange(len(ranked)), ranked["low_noise_frac_ok"], color="#2563EB", alpha=0.88)
    ax.errorbar(
        ranked["low_noise_frac_ok"],
        np.arange(len(ranked)),
        xerr=ranked["low_noise_std"],
        fmt="none",
        ecolor="#0F172A",
        capsize=3,
    )
    ax.set_yticks(np.arange(len(ranked)), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Proporción promedio de patrones OK (≤ 1 píxel), flip y mask de bajo nivel")
    ax.set_title("Top configuraciones de 1.a.2 por robustez a bajo ruido")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _save_robustness_plots(results: pd.DataFrame, figures_dir: pathlib.Path) -> None:
    if results.empty:
        return
    for kind in ("flip", "mask"):
        subset = results[results["kind"] == kind]
        if subset.empty:
            continue
        summary = subset.groupby(["model_type", "level"]).agg(
            ok_mean=("frac_ok", "mean"),
            ok_std=("frac_ok", "std"),
            err_mean=("mean_pixel_errors", "mean"),
            err_std=("mean_pixel_errors", "std"),
        ).reset_index()
        fig, (ax_ok, ax_err) = plt.subplots(1, 2, figsize=(13, 5), dpi=180)
        for model_type, group in summary.groupby("model_type"):
            group = group.sort_values("level")
            label = {
                "noisy_input": "Entrada ruidosa",
                "autoencoder": "AE",
                "denoising": "DAE",
                "vae": "VAE",
            }.get(model_type, model_type)
            for axis, mean_key, std_key, ylabel in (
                (ax_ok, "ok_mean", "ok_std", "Patrones OK (fracción)"),
                (ax_err, "err_mean", "err_std", "Error medio (píxeles por patrón)"),
            ):
                mean = group[mean_key].to_numpy(dtype=float)
                std = group[std_key].fillna(0).to_numpy(dtype=float)
                levels = group["level"].to_numpy(dtype=float)
                axis.plot(levels, mean, marker="o", linewidth=2, label=label)
                axis.fill_between(levels, mean - std, mean + std, alpha=0.16)
                axis.set_xlabel(f"Nivel de ruido {kind}")
                axis.set_ylabel(ylabel)
                axis.grid(alpha=0.3)
                axis.legend()
                if mean_key == "ok_mean":
                    axis.set_ylim(0, 1.05)
        fig.suptitle(f"Referencia sin procesar y robustez de AE/DAE/VAE ante ruido {kind}")
        fig.tight_layout()
        figures_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(figures_dir / f"robustness_{kind}.png", bbox_inches="tight")
        plt.close(fig)

        heat = subset.groupby(["model_type", "rank", "level"])["frac_ok"].mean().reset_index()
        ranks = sorted(heat["rank"].dropna().unique())
        levels = sorted(heat["level"].unique())
        model_types = ("noisy_input", "autoencoder", "denoising", "vae")
        fig, axes = plt.subplots(1, len(model_types), figsize=(22, max(4, len(ranks) * 0.45)), dpi=180)
        for axis, model_type in zip(axes, model_types):
            matrix = (
                heat[heat["model_type"] == model_type]
                .pivot(index="rank", columns="level", values="frac_ok")
                .reindex(index=ranks, columns=levels)
            )
            if matrix.empty:
                axis.set_visible(False)
                continue
            image = axis.imshow(matrix.to_numpy(), vmin=0, vmax=1, aspect="auto", cmap="viridis")
            axis.set_title({
                "noisy_input": "Entrada ruidosa",
                "autoencoder": "AE",
                "denoising": "DAE",
                "vae": "VAE",
            }[model_type])
            axis.set_xlabel(f"Nivel {kind}")
            axis.set_xticks(np.arange(len(levels)), [f"{value:g}" for value in levels])
            axis.set_ylabel("Ranking de configuración")
            axis.set_yticks(np.arange(len(ranks)), [f"Top {int(rank)}" for rank in ranks])
            for r in range(matrix.shape[0]):
                for c in range(matrix.shape[1]):
                    value = matrix.iloc[r, c]
                    if pd.notna(value):
                        axis.text(c, r, f"{value:.2f}", ha="center", va="center", color="white", fontsize=8)
        fig.suptitle(f"Mapa de calor de tolerancia al ruido {kind}")
        fig.subplots_adjust(left=0.055, right=0.91, top=0.88, bottom=0.14, wspace=0.38)
        colorbar_axis = fig.add_axes([0.935, 0.18, 0.012, 0.62])
        fig.colorbar(image, cax=colorbar_axis, label="Fracción de patrones OK")
        fig.savefig(figures_dir / f"heatmap_{kind}.png", bbox_inches="tight")
        plt.close(fig)


def _save_examples(
    pairs: List[Dict[str, str]],
    source_runs_dir: pathlib.Path,
    dae_runs_dir: pathlib.Path,
    figures_dir: pathlib.Path,
    levels: List[float],
) -> None:
    if not pairs:
        return
    best = next((pair for pair in pairs if pair["rank"] == "1"), None)
    if best is None:
        return
    X_clean = np.load("data/processed/font.npz")["X"].astype(np.float64)
    indices = [0, 1, 2, 3]
    sample = X_clean[indices]
    ae, _ = load_model_from_run(_run_path(source_runs_dir, best["source_run_id"]))
    dae, dae_cfg = load_model_from_run(_run_path(dae_runs_dir, best["dae_run_id"]))
    vae = None
    if "vae_run_id" in best:
        vae, _ = load_model_from_run(_run_path(dae_runs_dir, best["vae_run_id"]))

    for kind in ("flip", "mask"):
        shown_levels = [level for level in (0.1, 0.3, 0.5) if level in levels]
        if not shown_levels:
            shown_levels = [levels[0], levels[-1]] if len(levels) > 1 else levels
        rng = np.random.default_rng(42)
        row_labels = ["Original limpio", "Entrada ruidosa", "Reconstrucción AE", "Reconstrucción DAE"]
        if vae is not None:
            row_labels.append("Reconstrucción VAE")
        n_rows = len(row_labels)
        fig, axes = plt.subplots(n_rows, len(shown_levels), figsize=(3.4 * len(shown_levels), 1.5 * n_rows), dpi=180)
        if len(shown_levels) == 1:
            axes = np.asarray(axes).reshape(n_rows, 1)
        for col, level in enumerate(shown_levels):
            noisy = corrupt(sample, kind, level, rng)
            ae_hat = reconstruct(ae, noisy)
            dae_hat = reconstruct(dae, noisy)
            blocks = [sample, noisy, ae_hat, dae_hat]
            if vae is not None:
                blocks.append(reconstruct(vae, noisy))
            for row, block in enumerate(blocks):
                image = block.reshape(4, 7, 5)
                montage = np.concatenate(list(image), axis=1)
                axis = axes[row, col]
                axis.imshow(montage, cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
                axis.set_xticks([])
                axis.set_yticks([])
                if row == 0:
                    axis.set_title(f"{kind}={level:g}")
                if col == 0:
                    axis.set_ylabel(row_labels[row])
        fig.suptitle(
            f"Ejemplos de reconstrucción, configuración top 1 "
            f"(flip de entrenamiento={dae_cfg['noise']['level']:g})"
        )
        fig.tight_layout()
        figures_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(figures_dir / f"examples_{kind}.png", bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Selecciona configuraciones de 1.a.2 y evalúa AE/DAE/VAE ante flip y mask."
    )
    parser.add_argument("--summary", default="results/summary/runs.csv")
    parser.add_argument("--runs-dir", default="results/runs")
    parser.add_argument("--output-dir", default="results/denoising")
    parser.add_argument("--figures-dir", default="figures/b")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--low-levels", default="0.05,0.1")
    parser.add_argument("--test-levels", default="0,0.05,0.1,0.2,0.3,0.4,0.5")
    parser.add_argument("--train-noise-level", type=float, default=0.1)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=None, help="Sobrescribe las épocas de entrenamiento de 1.a.2.")
    parser.add_argument("--skip-training", action="store_true", help="Reutiliza pesos DAE/VAE existentes.")
    parser.add_argument("--without-vae", action="store_true", help="Omite el VAE de referencia sobre font.h.")
    args = parser.parse_args()

    if args.top_n < 1 or args.trials < 1:
        parser.error("--top-n y --trials deben ser positivos.")
    if not 0.0 <= args.train_noise_level <= 1.0:
        parser.error("--train-noise-level debe estar en [0, 1].")
    low_levels = _parse_levels(args.low_levels, "--low-levels")
    test_levels = _parse_levels(args.test_levels, "--test-levels")
    summary_path = pathlib.Path(args.summary)
    runs_dir = pathlib.Path(args.runs_dir)
    output_dir = pathlib.Path(args.output_dir)
    figures_dir = pathlib.Path(args.figures_dir)
    if not summary_path.exists():
        parser.error(f"No existe {summary_path}; primero ejecutá el barrido de 1.a.2.")
    source_df = pd.read_csv(summary_path)
    eligible = source_df[
        (source_df["dataset"] == "font")
        & (source_df["model_type"] == "autoencoder")
        & source_df["status"].isin(["ok", "early_stopped"])
    ].copy()
    eligible = eligible.drop_duplicates("run_id")
    eligible = eligible[eligible["run_id"].map(lambda run_id: (runs_dir / str(run_id) / "weights.npz").exists())]
    if eligible.empty:
        parser.error("No hay corridas AE de font con pesos disponibles para evaluar.")
    missing_columns = [column for column in CONFIG_COLUMNS if column not in eligible.columns]
    if missing_columns:
        parser.error(f"Faltan columnas de hiperparámetros en el resumen: {', '.join(missing_columns)}")

    X_clean = np.load("data/processed/font.npz")["X"].astype(np.float64)
    print(f"[RANK] Midiendo {len(eligible)} corridas ante bajo ruido...")
    low_results = _evaluate_candidates(
        eligible, runs_dir, X_clean, low_levels, args.trials, ("flip", "mask"), seed_offset=2026
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    low_results.to_csv(output_dir / "low_noise_eval.csv", index=False)
    ranked = _rank_candidates(low_results, args.top_n)
    ranked.to_csv(output_dir / "shortlist.csv", index=False)
    _save_ranking_plot(ranked, figures_dir / "low_noise_ranking.png")
    print(f"[RANK] Shortlist guardado en {output_dir / 'shortlist.csv'}")

    selected_ids = {
        run_id
        for value in ranked["source_run_ids"]
        for run_id in json.loads(value)
    }
    selected_sources = eligible[eligible["run_id"].astype(str).isin(selected_ids)].copy()
    pairs: List[Dict[str, str]] = []
    if not args.skip_training:
        pairs = _train_daes(
            ranked,
            selected_sources,
            runs_dir,
            output_dir,
            args.train_noise_level,
            args.epochs,
            include_vae=not args.without_vae,
        )
    else:
        dae_summary = output_dir / "summary" / "runs.csv"
        if dae_summary.exists():
            dae_df = pd.read_csv(dae_summary)
            rank_by_key = {
                _candidate_key(row): int(row["rank"])
                for _, row in ranked.iterrows()
            }
            for _, source in selected_sources.iterrows():
                rank = rank_by_key.get(_candidate_key(source))
                if rank is None:
                    continue
                dae_matches = dae_df[
                    dae_df["run_name"].astype(str).str.startswith(f"dae_b1_top{rank:02d}_")
                    & (dae_df["seed"].astype(int) == int(source["seed"]))
                ]
                vae_matches = (
                    dae_df[
                        dae_df["run_name"].astype(str).str.startswith(f"vae_b1_top{rank:02d}_")
                        & (dae_df["seed"].astype(int) == int(source["seed"]))
                    ]
                    if not args.without_vae
                    else dae_df.iloc[0:0]
                )
                for _, dae in dae_matches.iterrows():
                    pairs.append({
                        "rank": str(rank),
                        "source_run_id": str(source["run_id"]),
                        "dae_run_id": str(dae["run_id"]),
                        "seed": str(source["seed"]),
                    })
                    if not vae_matches.empty:
                        pairs[-1]["vae_run_id"] = str(vae_matches.iloc[0]["run_id"])
    if not pairs:
        parser.error("No hay corridas DAE compatibles; quitá --skip-training o entrená los modelos primero.")

    eval_records = []
    source_by_id = selected_sources.set_index(selected_sources["run_id"].astype(str))
    for pair in pairs:
        source_path = _run_path(runs_dir, pair["source_run_id"])
        dae_path = _run_path(output_dir / "runs", pair["dae_run_id"])
        runs_to_evaluate = [(source_path, "autoencoder"), (dae_path, "denoising")]
        if "vae_run_id" in pair:
            runs_to_evaluate.append((_run_path(output_dir / "runs", pair["vae_run_id"]), "vae"))
        source_row = source_by_id.loc[pair["source_run_id"]]
        metadata = {column: source_row.get(column) for column in CONFIG_COLUMNS}
        for run_path, model_type in runs_to_evaluate:
            model, _ = load_model_from_run(run_path)
            for kind in ("flip", "mask"):
                noise_seed = 31415 + int(pair["seed"]) + int(pair["rank"]) * 1009
                evaluations = evaluate_noise_levels(
                    model,
                    X_clean,
                    kind,
                    test_levels,
                    args.trials,
                    seed=noise_seed,
                )
                for result in evaluations:
                    eval_records.append({
                        "run_id": run_path.name,
                        "source_run_id": pair["source_run_id"],
                        "rank": int(pair["rank"]),
                        "seed": int(pair["seed"]),
                        "model_type": model_type,
                        **metadata,
                        **result,
                    })
        for kind in ("flip", "mask"):
            baseline = evaluate_noise_levels(
                None,
                X_clean,
                kind,
                test_levels,
                args.trials,
                seed=31415 + int(pair["seed"]) + int(pair["rank"]) * 1009,
            )
            for result in baseline:
                eval_records.append({
                    "run_id": "noisy_input",
                    "source_run_id": pair["source_run_id"],
                    "rank": int(pair["rank"]),
                    "seed": int(pair["seed"]),
                    "model_type": "noisy_input",
                    **metadata,
                    **result,
                })
    results = pd.DataFrame(eval_records)
    if not results.empty:
        results.to_csv(output_dir / "noise_eval.csv", index=False)
        _save_robustness_plots(results, figures_dir)
        _save_examples(pairs, runs_dir, output_dir / "runs", figures_dir, test_levels)
    print(f"[DONE] CSVs: {output_dir}; gráficos: {figures_dir}")


if __name__ == "__main__":
    main()
