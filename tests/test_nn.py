"""
tests/test_nn.py
================
Suite de pruebas unitarias y de integración para el motor de la red (FASE 2).
Incluye chequeo numérico riguroso de gradientes (gradient check), optimizadores,
schedules, VAE y prueba de integración de entrenamiento.

Uso:
  python tests/test_nn.py
  o: pytest tests/test_nn.py
"""

import pathlib
import sys
import numpy as np

# Configurar path
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tp5.nn.init import xavier_init, he_init, normal_small_init
from tp5.nn.activations import Identity, Sigmoid, Tanh, ReLU, LeakyReLU
from tp5.nn.layers import Dense, Sequential, Parameter
from tp5.nn.losses import MSELoss, BCELoss
from tp5.nn.schedules import ConstantSchedule, ExpDecaySchedule, StepSchedule
from tp5.nn.optimizers import SGD, Momentum, RMSprop, Adam, clip_grad_norm
from tp5.models.autoencoder import Autoencoder
from tp5.models.vae import VAE
from tp5.train.trainer import Trainer
from tp5.train.config import load_config


def compute_rel_error(grad_ana: np.ndarray, grad_num: np.ndarray) -> float:
    """Calcula el error relativo entre gradiente analítico y numérico."""
    norm_diff = np.linalg.norm(grad_ana - grad_num)
    norm_sum = np.linalg.norm(grad_ana) + np.linalg.norm(grad_num)
    if norm_sum < 1e-12:
        return 0.0
    return float(norm_diff / norm_sum)


# ---------------------------------------------------------------------------
# 1. Gradient check de Dense Layer
# ---------------------------------------------------------------------------

def test_dense_gradient_check():
    """Valida los gradientes analíticos de dW, db y dx en Dense con diferencias finitas."""
    rng = np.random.default_rng(123)
    batch_size = 4
    in_dim = 5
    out_dim = 3
    eps = 1e-6

    dense = Dense(in_dim, out_dim, init="xavier", rng=rng)
    x = rng.normal(size=(batch_size, in_dim))
    grad_out = rng.normal(size=(batch_size, out_dim))

    # Backward analítico
    dense.forward(x)
    grad_in_ana = dense.backward(grad_out)
    grad_W_ana = dense.W.grad.copy()
    grad_b_ana = dense.b.grad.copy()

    # Función objetivo escalar: f = sum(forward(x) * grad_out)
    # Por regla de la cadena: df/dW = x.T @ grad_out
    def f_eval(w_val, b_val, x_val):
        return np.sum((x_val @ w_val + b_val) * grad_out)

    # 1. Numérico dW
    grad_W_num = np.zeros_like(dense.W.data)
    for i in range(in_dim):
        for j in range(out_dim):
            dense.W.data[i, j] += eps
            f_plus = f_eval(dense.W.data, dense.b.data, x)
            dense.W.data[i, j] -= 2 * eps
            f_minus = f_eval(dense.W.data, dense.b.data, x)
            dense.W.data[i, j] += eps
            grad_W_num[i, j] = (f_plus - f_minus) / (2 * eps)

    err_W = compute_rel_error(grad_W_ana, grad_W_num)
    assert err_W < 1e-7, f"Error relativo en dW demasiado alto: {err_W}"

    # 2. Numérico db
    grad_b_num = np.zeros_like(dense.b.data)
    for j in range(out_dim):
        dense.b.data[0, j] += eps
        f_plus = f_eval(dense.W.data, dense.b.data, x)
        dense.b.data[0, j] -= 2 * eps
        f_minus = f_eval(dense.W.data, dense.b.data, x)
        dense.b.data[0, j] += eps
        grad_b_num[0, j] = (f_plus - f_minus) / (2 * eps)

    err_b = compute_rel_error(grad_b_ana, grad_b_num)
    assert err_b < 1e-7, f"Error relativo en db demasiado alto: {err_b}"

    # 3. Numérico dx
    grad_x_num = np.zeros_like(x)
    for b in range(batch_size):
        for i in range(in_dim):
            x[b, i] += eps
            f_plus = f_eval(dense.W.data, dense.b.data, x)
            x[b, i] -= 2 * eps
            f_minus = f_eval(dense.W.data, dense.b.data, x)
            x[b, i] += eps
            grad_x_num[b, i] = (f_plus - f_minus) / (2 * eps)

    err_x = compute_rel_error(grad_in_ana, grad_x_num)
    assert err_x < 1e-7, f"Error relativo en grad_in demasiado alto: {err_x}"


# ---------------------------------------------------------------------------
# 2. Gradient check de Activaciones
# ---------------------------------------------------------------------------

def test_activations_gradient_check():
    """Valida numéricamente las derivadas de Sigmoid, Tanh, LeakyReLU."""
    rng = np.random.default_rng(456)
    x = rng.normal(size=(5, 4))
    grad_out = rng.normal(size=(5, 4))
    eps = 1e-6

    for act in [Sigmoid(), Tanh(), LeakyReLU(alpha=0.1), Identity()]:
        # Analítico
        act.forward(x)
        grad_ana = act.backward(grad_out)

        # Numérico
        grad_num = np.zeros_like(x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                x[i, j] += eps
                f_plus = np.sum(act.forward(x) * grad_out)
                x[i, j] -= 2 * eps
                f_minus = np.sum(act.forward(x) * grad_out)
                x[i, j] += eps
                grad_num[i, j] = (f_plus - f_minus) / (2 * eps)

        err = compute_rel_error(grad_ana, grad_num)
        assert err < 1e-7, f"Error relativo en {act.__class__.__name__}: {err}"


# ---------------------------------------------------------------------------
# 3. Gradient check de Pérdidas (MSE y BCE)
# ---------------------------------------------------------------------------

def test_losses_gradient_check():
    """Valida los gradientes de MSELoss y BCELoss contra diferencias finitas."""
    rng = np.random.default_rng(789)
    n_samples, n_features = 4, 6
    eps = 1e-7

    # 1. MSE
    mse = MSELoss()
    y_pred = rng.uniform(0.1, 0.9, size=(n_samples, n_features))
    y_true = rng.uniform(0.0, 1.0, size=(n_samples, n_features))

    mse.forward(y_pred, y_true)
    grad_ana = mse.backward()

    grad_num = np.zeros_like(y_pred)
    for i in range(n_samples):
        for j in range(n_features):
            y_pred[i, j] += eps
            f_plus = mse.forward(y_pred, y_true)
            y_pred[i, j] -= 2 * eps
            f_minus = mse.forward(y_pred, y_true)
            y_pred[i, j] += eps
            grad_num[i, j] = (f_plus - f_minus) / (2 * eps)

    err_mse = compute_rel_error(grad_ana, grad_num)
    assert err_mse < 1e-7, f"Error en MSELoss: {err_mse}"

    # 2. BCE
    bce = BCELoss()
    y_pred_bce = rng.uniform(0.1, 0.9, size=(n_samples, n_features))
    y_true_bce = rng.choice([0.0, 1.0], size=(n_samples, n_features))

    bce.forward(y_pred_bce, y_true_bce)
    grad_ana_bce = bce.backward()

    grad_num_bce = np.zeros_like(y_pred_bce)
    for i in range(n_samples):
        for j in range(n_features):
            y_pred_bce[i, j] += eps
            f_plus = bce.forward(y_pred_bce, y_true_bce)
            y_pred_bce[i, j] -= 2 * eps
            f_minus = bce.forward(y_pred_bce, y_true_bce)
            y_pred_bce[i, j] += eps
            grad_num_bce[i, j] = (f_plus - f_minus) / (2 * eps)

    err_bce = compute_rel_error(grad_ana_bce, grad_num_bce)
    assert err_bce < 1e-7, f"Error en BCELoss: {err_bce}"


# ---------------------------------------------------------------------------
# 4. Gradient check del Autoencoder Completo
# ---------------------------------------------------------------------------

def test_autoencoder_gradient_check():
    """Valida la retropropagación en un Autoencoder multicapa completo de extremo a extremo."""
    rng = np.random.default_rng(999)
    eps = 1e-6

    # Arquitectura: 6 -> 4 -> 2 -> 4 -> 6
    enc = Sequential([
        Dense(6, 4, init="xavier", rng=rng),
        Tanh(),
        Dense(4, 2, init="xavier", rng=rng),
        Identity(),
    ])
    dec = Sequential([
        Dense(2, 4, init="xavier", rng=rng),
        Tanh(),
        Dense(4, 6, init="xavier", rng=rng),
        Sigmoid(),
    ])

    ae = Autoencoder(enc, dec, latent_dim=2, input_dim=6, output_dim=6)
    loss_fn = BCELoss()

    X = rng.choice([0.0, 1.0], size=(3, 6))

    # Analítico
    ae.zero_grad()
    x_hat = ae.forward(X)
    loss_val = loss_fn.forward(x_hat, X)
    grad_loss = loss_fn.backward()
    ae.backward(grad_loss)

    # Chequear un parámetro de cada capa (W del encoder y W del decoder)
    for param in ae.params():
        grad_ana = param.grad.copy()
        grad_num = np.zeros_like(param.data)

        # Probar sobre una muestra de elementos para velocidad
        flat_size = param.data.size
        sample_indices = np.random.default_rng(0).choice(flat_size, size=min(flat_size, 6), replace=False)

        orig = param.data.copy()
        for idx in sample_indices:
            coord = np.unravel_index(idx, param.data.shape)
            param.data[coord] += eps
            f_plus = loss_fn.forward(ae.forward(X), X)
            param.data[coord] -= 2 * eps
            f_minus = loss_fn.forward(ae.forward(X), X)
            param.data[coord] = orig[coord]

            grad_num[coord] = (f_plus - f_minus) / (2 * eps)
            err = abs(grad_ana[coord] - grad_num[coord]) / max(abs(grad_ana[coord]) + abs(grad_num[coord]), 1e-8)
            assert err < 1e-6, f"Error en parámetro {param.name} en {coord}: {err}"


# ---------------------------------------------------------------------------
# 5. Gradient check del VAE (Reparametrización + Divergencia KL)
# ---------------------------------------------------------------------------

def test_vae_gradient_check():
    """Valida el backward completo del VAE incluyendo cabezas mu y logvar y término KL."""
    rng = np.random.default_rng(2026)
    eps = 1e-6
    beta = 1.0
    n_samples = 3

    enc_trunk = Sequential([Dense(8, 6, init="xavier", rng=rng), Tanh()])
    mu_head = Dense(6, 3, init="xavier", rng=rng, name="mu")
    logvar_head = Dense(6, 3, init="xavier", rng=rng, name="logvar")
    decoder = Sequential([Dense(3, 8, init="xavier", rng=rng), Sigmoid()])

    vae = VAE(enc_trunk, mu_head, logvar_head, decoder, latent_dim=3, input_dim=8, output_dim=8, beta=beta)
    loss_fn = MSELoss()

    X = rng.uniform(0.0, 1.0, size=(n_samples, 8))

    # Fijar epsilon para reproducir la misma realización del forward
    fixed_eps = rng.normal(size=(n_samples, 3))

    def forward_eval():
        h = vae.encoder_trunk.forward(X)
        mu = vae.mu_head.forward(h)
        logvar = vae.logvar_head.forward(h)
        std = np.exp(0.5 * logvar)
        z = mu + std * fixed_eps
        x_hat = vae.decoder.forward(z)
        recon = loss_fn.forward(x_hat, X)
        kl = -0.5 * np.mean(np.sum(1.0 + logvar - (mu ** 2) - np.exp(logvar), axis=1))
        return recon + beta * kl

    # Paso analítico
    vae.zero_grad()
    # Ejecutamos forward manual fijando los caches exactos
    h = vae.encoder_trunk.forward(X)
    mu = vae.mu_head.forward(h)
    logvar = vae.logvar_head.forward(h)
    std = np.exp(0.5 * logvar)
    z = mu + std * fixed_eps
    x_hat = vae.decoder.forward(z)

    vae._h_cache = h
    vae._mu_cache = mu
    vae._logvar_cache = logvar
    vae._std_cache = std
    vae._eps_cache = fixed_eps
    vae._z_cache = z

    loss_fn.forward(x_hat, X)
    grad_recon = loss_fn.backward()
    vae.backward(grad_recon)

    # Validar gradientes numéricos en parámetros de mu_head, logvar_head y tronco
    for p in [vae.mu_head.W, vae.logvar_head.W, vae.decoder.layers[0].W]:
        orig = p.data.copy()
        for i in range(p.data.shape[0]):
            for j in range(p.data.shape[1]):
                p.data[i, j] += eps
                f_plus = forward_eval()
                p.data[i, j] -= 2 * eps
                f_minus = forward_eval()
                p.data[i, j] = orig[i, j]

                num_g = (f_plus - f_minus) / (2 * eps)
                ana_g = p.grad[i, j]
                err = abs(ana_g - num_g) / max(abs(ana_g) + abs(num_g), 1e-8)
                assert err < 1e-6, f"Error en VAE {p.name}[{i},{j}]: err={err}, ana={ana_g}, num={num_g}"


# ---------------------------------------------------------------------------
# 6. Test de Optimizadores y Recorte de Gradiente
# ---------------------------------------------------------------------------

def test_optimizers_and_clipping():
    """Valida los pasos de actualización y la reducción de norma en clip_grad_norm."""
    p1 = Parameter(np.array([[1.0, 2.0], [3.0, 4.0]]))
    p2 = Parameter(np.array([[5.0, 6.0]]))
    params = [p1, p2]

    # Gradientes artificiales grandes
    p1.grad = np.array([[10.0, 20.0], [30.0, 40.0]])
    p2.grad = np.array([[50.0, 60.0]])

    initial_norm = clip_grad_norm(params, max_norm=5.0)
    assert initial_norm > 5.0

    post_norm = float(np.sqrt(np.sum(p1.grad ** 2) + np.sum(p2.grad ** 2)))
    assert abs(post_norm - 5.0) < 1e-6

    # Test de paso Adam
    opt = Adam([p1, p2], lr=0.01)
    orig_p1 = p1.data.copy()
    opt.step()
    assert not np.allclose(p1.data, orig_p1), "Adam debió actualizar los parámetros."


# ---------------------------------------------------------------------------
# 7. Test de Schedules
# ---------------------------------------------------------------------------

def test_schedules():
    """Valida los schedulers constant, exp_decay y step."""
    s_const = ConstantSchedule(0.01)
    assert s_const.get_lr(100) == 0.01

    s_exp = ExpDecaySchedule(0.01, gamma=0.9)
    assert abs(s_exp.get_lr(2) - (0.01 * 0.81)) < 1e-9

    s_step = StepSchedule(0.01, step_size=10, gamma=0.5)
    assert s_step.get_lr(5) == 0.01
    assert s_step.get_lr(15) == 0.005
    assert s_step.get_lr(25) == 0.0025


# ---------------------------------------------------------------------------
# 8. Test de Integración: Entrenamiento real de Autoencoder en font.h
# ---------------------------------------------------------------------------

def test_trainer_integration_font():
    """Ejecuta una corrida corta en font.h verificando disminución de pérdida y guardado de resultados."""
    cfg = load_config("configs/ae_basic.yaml")
    # Configuración reducida para prueba rápida
    cfg["train"]["epochs"] = 60
    cfg["train"]["eval_every"] = 20
    cfg["output"]["dir"] = "tests/output/test_runs"
    cfg["output"]["summary_dir"] = "tests/output/test_summary"

    trainer = Trainer(cfg)
    res = trainer.train()

    assert res["summary"]["status"] == "ok"
    assert len(trainer.history) >= 3

    initial_loss = trainer.history[0]["loss"]
    final_loss = trainer.history[-1]["loss"]

    assert final_loss < initial_loss, f"La pérdida debió disminuir: initial={initial_loss}, final={final_loss}"

    # Verificar que los artefactos fueron escritos en disco
    run_dir = pathlib.Path(res["run_dir"])
    assert (run_dir / "epochs.csv").exists()
    assert (run_dir / "config_resolved.yaml").exists()
    assert (run_dir / "weights.npz").exists()
    assert (run_dir / "latent.csv").exists()
    assert (run_dir / "reconstructions.npz").exists()


# ---------------------------------------------------------------------------
# Main ejecutable
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 65)
    print("  TP5 — Test Suite del Motor de Red Neuronal (Hito 2)")
    print("=" * 65)

    print("\n[1/8] Verificando gradientes de capa Dense...")
    test_dense_gradient_check()
    print("  ✓ Dense gradient check OK (err < 1e-7)")

    print("\n[2/8] Verificando gradientes de Activaciones...")
    test_activations_gradient_check()
    print("  ✓ Activaciones gradient check OK (err < 1e-7)")

    print("\n[3/8] Verificando gradientes de Pérdidas (MSE y BCE)...")
    test_losses_gradient_check()
    print("  ✓ Pérdidas gradient check OK (err < 1e-7)")

    print("\n[4/8] Verificando gradientes del Autoencoder Completo...")
    test_autoencoder_gradient_check()
    print("  ✓ Autoencoder completo gradient check OK (err < 1e-6)")

    print("\n[5/8] Verificando gradientes del VAE (reparametrización + KL)...")
    test_vae_gradient_check()
    print("  ✓ VAE backward y KL gradient check OK (err < 1e-6)")

    print("\n[6/8] Verificando Optimizadores y Recorte de Gradiente...")
    test_optimizers_and_clipping()
    print("  ✓ Optimizadores y clip_grad_norm OK")

    print("\n[7/8] Verificando Schedulers de Learning Rate...")
    test_schedules()
    print("  ✓ Schedulers OK")

    print("\n[8/8] Ejecutando integración de entrenamiento sobre font.h...")
    test_trainer_integration_font()
    print("  ✓ Integración de entrenamiento y persistencia de artefactos OK")

    print("\n" + "=" * 65)
    print("  TODAS LAS PRUEBAS DEL MOTOR (HITO 2) PASARON EXITOSAMENTE.")
    print("=" * 65 + "\n")
