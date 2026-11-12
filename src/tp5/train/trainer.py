"""
tp5.train.trainer
=================
Motor central de entrenamiento para Autoencoder, DAE y VAE.
Controla el ciclo de optimización, evaluación, métricas, early stopping y guardado de resultados.
"""

import time
import pathlib
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

from ..data.loader import load_dataset
from ..data.noise import corrupt
from ..nn.losses import get_loss
from ..nn.optimizers import build_optimizer
from ..nn.schedules import build_schedule
from ..models.registry import build_model
from .metrics import compute_metrics
from .io import (
    generate_run_id,
    ensure_run_dir,
    save_config_resolved,
    save_epochs_csv,
    save_weights_npz,
    save_latent_csv,
    save_reconstructions_npz,
    append_summary_run,
)


class Trainer:
    """
    Entrenador desacoplado y reproducible.
    Todo el comportamiento se define a partir de un diccionario de configuración.
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.seed = int(cfg.get("seed", 42))

        # Generadores reproducibles
        self.rng_init = np.random.default_rng(self.seed)
        self.rng_train = np.random.default_rng(self.seed + 1)
        self.rng_noise = np.random.default_rng(self.seed + 2)

        # 1. Cargar datos
        self.dataset = load_dataset(self.cfg)
        self.X_train: np.ndarray = self.dataset["X_train"]
        self.X_val: Optional[np.ndarray] = self.dataset.get("X_val", None)
        self.labels: Optional[List[str]] = self.dataset.get("labels", None)
        self.image_shape: Tuple[int, ...] = self.dataset["image_shape"]

        self.input_dim = self.X_train.shape[1]
        self.is_binary = (self.cfg.get("data", {}).get("dataset", "font") == "font")

        # 2. Construir modelo
        self.model = build_model(self.cfg, input_dim=self.input_dim, rng=self.rng_init)

        # 3. Construir pérdida
        recon_loss_name = self.cfg.get("loss", {}).get("recon", "bce")
        self.loss_fn = get_loss(recon_loss_name)

        # 4. Construir optimizador y scheduler
        opt_cfg = self.cfg.get("optimizer", {})
        self.optimizer = build_optimizer(opt_cfg, self.model.params())
        self.scheduler = build_schedule(opt_cfg)

        # Historial de métricas
        self.history: List[Dict[str, Any]] = []

    def train(self) -> Dict[str, Any]:
        """Ejecuta el ciclo completo de entrenamiento."""
        t_start = time.time()
        train_cfg = self.cfg.get("train", {})
        n_epochs = int(train_cfg.get("epochs", 1000))
        batch_size_cfg = train_cfg.get("batch_size", "full")
        eval_every = max(1, int(train_cfg.get("eval_every", 10)))
        early_stop_cfg = train_cfg.get("early_stop", None)

        n_samples = len(self.X_train)
        if batch_size_cfg == "full":
            batch_size = n_samples
        else:
            batch_size = max(1, min(int(batch_size_cfg), n_samples))

        # Parámetros de ruido para Denoising
        model_type = self.cfg.get("model", {}).get("type", "autoencoder").lower()
        noise_cfg = self.cfg.get("noise", {})
        noise_level = float(noise_cfg.get("level", 0.0))
        noise_kind = noise_cfg.get("kind", "flip")
        is_denoising = (model_type == "denoising") or (noise_level > 0.0)

        # VAE Beta warmup
        is_vae = (model_type == "vae")
        target_beta = float(self.cfg.get("loss", {}).get("beta", 1.0))
        warmup_epochs = int(self.cfg.get("loss", {}).get("beta_warmup_epochs", 0))

        # Control de Early Stopping
        status = "ok"
        epoch_goal_reached = None
        best_patterns_ok = -1
        epoch_of_best = None
        consecutive_patience = 0

        for epoch in range(1, n_epochs + 1):
            # 1. Actualizar LR
            current_lr = self.scheduler.get_lr(epoch)
            self.optimizer.set_lr(current_lr)

            # 2. VAE Warmup de Beta
            if is_vae:
                if warmup_epochs > 0:
                    current_beta = target_beta * min(1.0, epoch / warmup_epochs)
                else:
                    current_beta = target_beta
                self.model.beta = current_beta

            # 3. Barajar datos
            indices = self.rng_train.permutation(n_samples)

            # 4. Iterar sobre batches
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_idx = indices[start_idx:end_idx]

                x_clean_batch = self.X_train[batch_idx]

                # Aplicar ruido si corresponde
                if is_denoising:
                    x_in_batch = corrupt(
                        x_clean_batch,
                        kind=noise_kind,
                        level=noise_level,
                        rng=self.rng_noise,
                    )
                else:
                    x_in_batch = x_clean_batch

                # Forward
                self.model.zero_grad()
                if is_vae:
                    x_hat_batch = self.model.forward(x_in_batch, rng=self.rng_train)
                else:
                    x_hat_batch = self.model.forward(x_in_batch)

                # Pérdida contra la entrada limpia
                self.loss_fn.forward(x_hat_batch, x_clean_batch)
                grad_loss = self.loss_fn.backward()

                # Backward
                self.model.backward(grad_loss)

                # Paso del optimizador
                self.optimizer.step()

            # 5. Evaluación periódica sobre todo el dataset
            is_eval_epoch = (epoch % eval_every == 0) or (epoch == 1) or (epoch == n_epochs)
            if is_eval_epoch:
                metrics_row = self._evaluate(epoch, current_lr, t_start)
                self.history.append(metrics_row)

                # Seguimiento del mejor punto
                if self.is_binary:
                    curr_ok = metrics_row["n_patterns_ok"]
                    if curr_ok > best_patterns_ok:
                        best_patterns_ok = curr_ok
                        epoch_of_best = epoch

                # Chequeo de divergencia numérica
                loss_val = metrics_row["loss"]
                if np.isnan(loss_val) or np.isinf(loss_val):
                    status = "diverged"
                    break

                # Chequeo de Early Stopping
                if early_stop_cfg:
                    target_metric = early_stop_cfg.get("metric", "max_pixel_errors")
                    threshold = float(early_stop_cfg.get("threshold", 0))
                    patience = int(early_stop_cfg.get("patience", 100))

                    val = metrics_row.get(target_metric, None)
                    if val is not None:
                        condition_met = (
                            (val <= threshold) if "error" in target_metric else (val >= threshold)
                        )
                        if condition_met:
                            if epoch_goal_reached is None:
                                epoch_goal_reached = epoch
                            consecutive_patience += 1
                            if consecutive_patience >= patience:
                                status = "early_stopped"
                                break
                        else:
                            consecutive_patience = 0

        total_time_s = time.time() - t_start

        # 6. Evaluación final y persistencia de artefactos
        result = self._save_results(
            total_time_s=total_time_s,
            status=status,
            epoch_goal_reached=epoch_goal_reached,
            best_patterns_ok=best_patterns_ok,
            epoch_of_best=epoch_of_best,
        )
        return result

    def _evaluate(self, epoch: int, lr: float, t_start: float) -> Dict[str, Any]:
        """Calcula métricas globales de evaluación sobre el conjunto completo."""
        if hasattr(self.model, "encode_mean"):
            # Para VAE: reconstrucción determinista con mu
            z_mean = self.model.encode_mean(self.X_train)
            x_hat_train = self.model.decode(z_mean)
            # Divergencia KL sobre el dataset
            _ = self.model.forward(self.X_train, rng=None)
            kl = self.model.kl_loss()
        else:
            x_hat_train = self.model.forward(self.X_train)
            kl = 0.0

        recon_loss = self.loss_fn.forward(x_hat_train, self.X_train)

        metrics = compute_metrics(
            y_true=self.X_train,
            y_pred=x_hat_train,
            recon_loss=recon_loss,
            kl_loss=kl,
            is_binary=self.is_binary,
        )

        row: Dict[str, Any] = {
            "epoch": epoch,
            "loss": metrics["loss"],
            "recon_loss": metrics["recon_loss"],
            "kl_loss": metrics["kl_loss"],
            "val_loss": None,
            "max_pixel_errors": metrics.get("max_pixel_errors"),
            "mean_pixel_errors": metrics.get("mean_pixel_errors"),
            "n_patterns_ok": metrics.get("n_patterns_ok"),
            "n_patterns_perfect": metrics.get("n_patterns_perfect"),
            "grad_norm": self.optimizer.last_grad_norm,
            "lr": lr,
            "time_s": round(time.time() - t_start, 3),
        }

        # Evaluación en validación si existe
        if self.X_val is not None:
            if hasattr(self.model, "encode_mean"):
                z_val = self.model.encode_mean(self.X_val)
                x_hat_val = self.model.decode(z_val)
            else:
                x_hat_val = self.model.forward(self.X_val)
            row["val_loss"] = float(self.loss_fn.forward(x_hat_val, self.X_val))

        return row

    def _save_results(
        self,
        total_time_s: float,
        status: str,
        epoch_goal_reached: Optional[int],
        best_patterns_ok: int,
        epoch_of_best: Optional[int],
    ) -> Dict[str, Any]:
        """Genera y persiste todos los artefactos de salida."""
        run_id = generate_run_id(self.cfg)
        out_base_dir = self.cfg.get("output", {}).get("dir", "results/runs")
        run_dir = ensure_run_dir(out_base_dir, run_id)

        # 1. config_resolved.yaml
        config_path = save_config_resolved(run_dir, self.cfg)

        # 2. epochs.csv
        save_epochs_csv(run_dir, self.history)

        # 3. weights.npz
        if self.cfg.get("output", {}).get("save_weights", True):
            save_weights_npz(run_dir, self.model.get_weights())

        # 4. latent.csv
        if hasattr(self.model, "encode_mean"):
            mu, logvar = self.model.encode(self.X_train)
            save_latent_csv(run_dir, latent_data=mu, labels=self.labels, logvar_data=logvar)
        else:
            z = self.model.encode(self.X_train)
            save_latent_csv(run_dir, latent_data=z, labels=self.labels)

        # 5. reconstructions.npz
        if hasattr(self.model, "encode_mean"):
            x_hat = self.model.decode(self.model.encode_mean(self.X_train))
        else:
            x_hat = self.model.forward(self.X_train)

        save_reconstructions_npz(run_dir, X_clean=self.X_train, X_hat=x_hat)

        # 6. Fila para summary/runs.csv
        last_metric = self.history[-1] if self.history else {}
        hidden_acts = []
        for l in getattr(self.model, "encoder", getattr(self.model, "encoder_trunk", None)).layers:
            if hasattr(l, "__class__") and "Dense" not in l.__class__.__name__:
                hidden_acts.append(l.__class__.__name__.lower())
        activation_hidden = hidden_acts[0] if hidden_acts else "none"

        summary_row: Dict[str, Any] = {
            "run_id": run_id,
            "run_name": self.cfg.get("run_name", "run"),
            "model_type": self.cfg.get("model", {}).get("type", "autoencoder"),
            "dataset": self.cfg.get("data", {}).get("dataset", "font"),
            "seed": self.seed,
            "arch": self.model.arch_str(),
            "n_params": self.model.count_params(),
            "activation_hidden": activation_hidden,
            "recon_loss_fn": self.cfg.get("loss", {}).get("recon", "bce"),
            "optimizer": self.cfg.get("optimizer", {}).get("name", "adam"),
            "lr0": self.cfg.get("optimizer", {}).get("lr", 0.001),
            "schedule": self.cfg.get("optimizer", {}).get("schedule", {}).get("name", "constant"),
            "init": self.cfg.get("model", {}).get("init", "xavier"),
            "batch_size": self.cfg.get("train", {}).get("batch_size", "full"),
            "beta": self.cfg.get("loss", {}).get("beta", 1.0) if hasattr(self.model, "beta") else None,
            "noise_kind": self.cfg.get("noise", {}).get("kind", None),
            "noise_level": self.cfg.get("noise", {}).get("level", 0.0),
            "epochs_run": last_metric.get("epoch", 0),
            "final_loss": last_metric.get("loss", None),
            "final_recon": last_metric.get("recon_loss", None),
            "final_kl": last_metric.get("kl_loss", None),
            "final_max_pixel_errors": last_metric.get("max_pixel_errors", None),
            "final_mean_pixel_errors": last_metric.get("mean_pixel_errors", None),
            "final_n_patterns_ok": last_metric.get("n_patterns_ok", None),
            "final_n_patterns_perfect": last_metric.get("n_patterns_perfect", None),
            "best_n_patterns_ok": best_patterns_ok if best_patterns_ok >= 0 else None,
            "epoch_of_best": epoch_of_best,
            "epoch_goal_reached": epoch_goal_reached,
            "train_time_s": round(total_time_s, 3),
            "status": status,
            "config_path": str(config_path),
        }

        summary_dir = self.cfg.get("output", {}).get("summary_dir", "results/summary")
        append_summary_run(summary_dir, summary_row)

        return {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "summary": summary_row,
            "last_metric": last_metric,
        }
