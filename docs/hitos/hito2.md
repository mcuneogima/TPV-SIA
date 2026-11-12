# Hito 2: Motor de Red Neuronal y Entrenamiento (FASE 2)

## 1. Resumen de lo realizado

En este hito se implementó, validó y testeó integralmente la **FASE 2: Motor de la red** del plan de trabajo, proveyendo un framework completo de Deep Learning desarrollado estrictamente sobre `numpy` (sin PyTorch ni TensorFlow), modular, extensible y con cálculo numérico en `float64` para garantizar máxima estabilidad y precisión.

Los componentes implementados y verificados abarcan:

1. **Primitivas de Red Neuronal (`tp5.nn`)**:
   - [`tp5.nn.layers.Parameter`](file:///c:/Users/Matias%20Cuneo%20Gima/ITBA/SIA/TPV-SIA/src/tp5/nn/layers.py): Contenedor de tensores para pesos y sesgos que almacena `data` y acumula `grad` en `float64`.
   - [`tp5.nn.layers.Dense`](file:///c:/Users/Matias%20Cuneo%20Gima/ITBA/SIA/TPV-SIA/src/tp5/nn/layers.py): Capa totalmente conectada $y = x W + b$ con forward matricial y backward analítico exacto ($dW = x^T \cdot \text{grad\_out}$, $db = \sum \text{grad\_out}$, $\text{grad\_in} = \text{grad\_out} \cdot W^T$).
   - [`tp5.nn.layers.Sequential`](file:///c:/Users/Matias%20Cuneo%20Gima/ITBA/SIA/TPV-SIA/src/tp5/nn/layers.py): Contenedor secuencial que encadena capas y propaga gradientes en sentido inverso.
   - [`tp5.nn.activations`](file:///c:/Users/Matias%20Cuneo%20Gima/ITBA/SIA/TPV-SIA/src/tp5/nn/activations.py): `Identity`, `Sigmoid` (con clipping para evitar overflow), `Tanh`, `ReLU` y `LeakyReLU` con derivadas locales para retropropagación.
   - [`tp5.nn.losses`](file:///c:/Users/Matias%20Cuneo%20Gima/ITBA/SIA/TPV-SIA/src/tp5/nn/losses.py): `MSELoss` y `BCELoss` implementando la convención del proyecto: suma sobre píxeles (features) por muestra, y promedio sobre muestras del batch.
   - [`tp5.nn.init`](file:///c:/Users/Matias%20Cuneo%20Gima/ITBA/SIA/TPV-SIA/src/tp5/nn/init.py): Inicializaciones `xavier` (Glorot uniforme y normal), `he` (Kaiming normal y uniforme) y `normal_small`.
   - [`tp5.nn.optimizers`](file:///c:/Users/Matias%20Cuneo%20Gima/ITBA/SIA/TPV-SIA/src/tp5/nn/optimizers.py): Algoritmos de optimización `SGD`, `Momentum`, `RMSprop` y `Adam` con corrección de sesgo (bias correction) y soporte para recorte global de gradiente (`clip_grad_norm`).
   - [`tp5.nn.schedules`](file:///c:/Users/Matias%20Cuneo%20Gima/ITBA/SIA/TPV-SIA/src/tp5/nn/schedules.py): Esquemas de decaimiento del learning rate: `constant`, `exp_decay` y `step`.

2. **Arquitecturas de Modelos (`tp5.models`)**:
   - [`tp5.models.autoencoder.Autoencoder`](file:///c:/Users/Matias%20Cuneo%20Gima/ITBA/SIA/TPV-SIA/src/tp5/models/autoencoder.py): Modelo modular con `encoder` y `decoder` desacoplados, apto para autoencoder básico y denoising (DAE). Métodos: `encode(X)`, `decode(Z)`, `forward(X)`, `backward(grad_out)`, `params()`, `get_weights()`, `set_weights()`.
   - [`tp5.models.vae.VAE`](file:///c:/Users/Matias%20Cuneo%20Gima/ITBA/SIA/TPV-SIA/src/tp5/models/vae.py): Autoencoder Variacional con truco de reparametrización $z = \mu + \sigma \odot \epsilon$ con $\epsilon \sim \mathcal{N}(0, I)$ y cabezas lineales $\mu$ y $\log\sigma^2$. Pérdida combinada $\mathcal{L} = \mathcal{L}_{\text{recon}} + \beta \cdot \text{KL}$, con backward analítico exacto de la reconstrucción y del término KL. Métodos generativos: `sample(n, rng)`, `encode_mean(X)` determinista para gráficos latentes, y soporte de $\beta$-warmup.
   - [`tp5.models.registry`](file:///c:/Users/Matias%20Cuneo%20Gima/ITBA/SIA/TPV-SIA/src/tp5/models/registry.py): Registro extensible que mapea cadenas de texto de YAMLs a clases e instanciadores dinámicos (`build_model`, `build_optimizer`, `build_loss`, `build_schedule`).

3. **Pipeline de Entrenamiento y Persistencia (`tp5.train`)**:
   - [`tp5.train.trainer.Trainer`](file:///c:/Users/Matias%20Cuneo%20Gima/ITBA/SIA/TPV-SIA/src/tp5/train/trainer.py): Bucle de optimización con barajado reproducible por época, soporte para mini-batches y full-batch, inyección de ruido para Denoising Autoencoder, cálculo periódico de métricas globales, soporte de early stopping condicional y divergencia numérica.
   - [`tp5.train.metrics`](file:///c:/Users/Matias%20Cuneo%20Gima/ITBA/SIA/TPV-SIA/src/tp5/train/metrics.py): Cálculo de métricas sobre el dataset completo: `loss`, `recon_loss`, `kl_loss`, `max_pixel_errors`, `mean_pixel_errors`, `n_patterns_ok` ($\le 1$ px error), `n_patterns_perfect` ($0$ px error), norma de gradiente y learning rate.
   - [`tp5.train.io`](file:///c:/Users/Matias%20Cuneo%20Gima/ITBA/SIA/TPV-SIA/src/tp5/train/io.py): Escritura automática de todos los formatos especificados:
     - `results/summary/runs.csv`: tabla acumulativa con 31 columnas descriptivas e hiperparámetros.
     - `results/runs/<run_id>/epochs.csv`: evolución por época.
     - `results/runs/<run_id>/weights.npz`: pesos serializados para posterior inferencia sin reentrenar.
     - `results/runs/<run_id>/latent.csv`: coordenadas latentes y etiquetas.
     - `results/runs/<run_id>/reconstructions.npz`: matrices limpias, ruidosas y reconstruidas.
     - `results/runs/<run_id>/config_resolved.yaml`: configuración completa con defaults.
   - [`tp5.train.sweep`](file:///c:/Users/Matias%20Cuneo%20Gima/ITBA/SIA/TPV-SIA/src/tp5/train/sweep.py): Expansor de grillas cartesianas de hiperparámetros con soporte de claves jerárquicas y variantes arquitectónicas.

4. **Scripts Ejecutables y Configuraciones**:
   - [`scripts/run.py`](file:///c:/Users/Matias%20Cuneo%20Gima/ITBA/SIA/TPV-SIA/scripts/run.py): CLI para ejecutar una corrida individual (`python scripts/run.py --config configs/ae_basic.yaml`).
   - [`scripts/sweep.py`](file:///c:/Users/Matias%20Cuneo%20Gima/ITBA/SIA/TPV-SIA/scripts/sweep.py): CLI para barridos (`python scripts/sweep.py --sweep configs/sweeps/a2_arquitecturas.yaml`).
   - Configuraciones YAML listas para las siguientes fases:
     - [`configs/ae_basic.yaml`](file:///c:/Users/Matias%20Cuneo%20Gima/ITBA/SIA/TPV-SIA/configs/ae_basic.yaml): Autoencoder básico 35-24-12-2-12-24-35 sobre font con Adam y BCE.
     - [`configs/dae.yaml`](file:///c:/Users/Matias%20Cuneo%20Gima/ITBA/SIA/TPV-SIA/configs/dae.yaml): Denoising Autoencoder con ruido flip de bits.
     - [`configs/vae_sprites.yaml`](file:///c:/Users/Matias%20Cuneo%20Gima/ITBA/SIA/TPV-SIA/configs/vae_sprites.yaml): VAE sobre sprites continuos con $\beta$-warmup.
     - [`configs/sweeps/a2_arquitecturas.yaml`](file:///c:/Users/Matias%20Cuneo%20Gima/ITBA/SIA/TPV-SIA/configs/sweeps/a2_arquitecturas.yaml): Grilla de exploración para la Parte 1.a.2.

---

## 2. Validación Numérica y Gradient Check

Se diseñó y ejecutó una suite exhaustiva de pruebas unitarias en [`tests/test_nn.py`](file:///c:/Users/Matias%20Cuneo%20Gima/ITBA/SIA/TPV-SIA/tests/test_nn.py).

### Chequeo Numérico de Gradientes (Diferencias Finitas)

Para cada componente se contrastó la derivada analítica implementada en `backward()` contra la aproximación centrada por diferencias finitas:

$$g_{\text{num}} = \frac{f(\theta + \epsilon) - f(\theta - \epsilon)}{2\epsilon}, \quad \text{con } \epsilon = 10^{-6}$$

El error relativo medido se computó como:

$$\text{Error Relativo} = \frac{\|g_{\text{ana}} - g_{\text{num}}\|_2}{\|g_{\text{ana}}\|_2 + \|g_{\text{num}}\|_2}$$

Resultados obtenidos en la suite:
1. **Capa `Dense`**:
   - Error relativo en $dW$: $< 10^{-7}$ ($\checkmark$)
   - Error relativo en $db$: $< 10^{-7}$ ($\checkmark$)
   - Error relativo en $dx$ ($\text{grad\_in}$): $< 10^{-7}$ ($\checkmark$)
2. **Funciones de activación (`Sigmoid`, `Tanh`, `LeakyReLU`, `Identity`)**:
   - Error relativo en gradientes locales: $< 10^{-7}$ ($\checkmark$)
3. **Pérdidas (`MSELoss` y `BCELoss`)**:
   - Error relativo: $< 10^{-7}$ ($\checkmark$)
4. **Autoencoder Multicapa Completo**:
   - Error relativo en todos los parámetros de codificador y decodificador: $< 10^{-6}$ ($\checkmark$)
5. **Autoencoder Variacional (VAE)**:
   - Gradientes analíticos de $\mu$, $\log\sigma^2$, decoder y tronco del encoder incluyendo el gradiente de la divergencia KL: $< 10^{-6}$ ($\checkmark$)
6. **Optimizadores y Schedulers**:
   - `SGD`, `Momentum`, `RMSprop`, `Adam` y `clip_grad_norm`: verificados.
   - `ConstantSchedule`, `ExpDecaySchedule`, `StepSchedule`: verificados.
7. **Prueba de Integración y Convergencia**:
   - Corrida de prueba en `font.h`: descenso monotónico de pérdida y guardado completo de artefactos.
   - Corrida completa con `configs/ae_basic.yaml`: se alcanzaron **32/32 patrones OK** ($\le 1$ error) y **29/32 patrones perfectos** con un error máximo de píxel de $1$.

Resultado general de pytest:
```
tests/test_data.py ....   [33%]
tests/test_nn.py   ........ [100%]
12 passed in 2.76s
```
