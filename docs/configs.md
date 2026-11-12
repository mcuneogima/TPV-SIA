# Especificación y Guía de Archivos de Configuración (YAML)

Este documento detalla el diseño, la sintaxis y el funcionamiento de los archivos YAML utilizados en el proyecto:
1. **Configuraciones de corrida individual (`run.yaml`)**: definen un entrenamiento determinista y autocontenido.
2. **Configuraciones de barrido (`sweep.yaml`)**: automatizan la exploración de hiperparámetros mediante dot-notation, producto cartesiano y variantes.

---

## 1. Configuración de Corrida Individual

Un archivo de corrida define un único experimento reproducible de principio a fin a partir de una semilla y sus hiperparámetros.

### Estructura general comentada (`configs/ae_basic.yaml`)

```yaml
run_name: ae_basic_adam_bce
seed: 42                           # Semilla global (inicialización de pesos, batches, etc.)

# 1. Dataset
data:
  dataset: font                    # 'font' | 'sprites'
  path: data/processed/font.npz    # Ruta generada en Fase 1
  val_split: 0.0                   # Fracción de validación (0.0 para memorizar los 32 patrones)
  seed: 42                         # Semilla para el split (si val_split > 0)

# 2. Arquitectura de la red
model:
  type: autoencoder                # 'autoencoder' | 'denoising' | 'vae'
  latent_dim: 2                    # Dimensión del espacio latente (2 para consigna 1.a)
  encoder:
    layers:
      - {units: 24, activation: tanh}
      - {units: 12, activation: tanh}
      - {units: 2,  activation: identity}   # Cuello de botella latente
  decoder:
    layers:
      - {units: 12, activation: tanh}
      - {units: 24, activation: tanh}
      - {units: 35, activation: sigmoid}    # 35 píxeles de salida binarizables
  init: xavier                     # 'xavier' | 'he' | 'normal_small'

# 3. Función de pérdida
loss:
  recon: bce                       # 'bce' (Binary Cross Entropy) | 'mse' (Mean Squared Error)
  beta: 1.0                        # Ponderación del término KL (solo VAE)
  beta_warmup_epochs: 0            # Épocas de ascenso lineal de beta (solo VAE)

# 4. Inyección de ruido (solo model.type == 'denoising')
noise:
  kind: flip                       # 'flip' | 'mask' | 'gaussian'
  level: 0.1                       # Probabilidad de corrupción o desvío gaussiano
  regenerate_each_epoch: true      # Si regenera el ruido estocástico en cada época

# 5. Optimizador y schedule de learning rate
optimizer:
  name: adam                       # 'sgd' | 'momentum' | 'rmsprop' | 'adam'
  lr: 0.001                        # Learning rate inicial
  momentum: 0.9                    # Para 'momentum'
  beta1: 0.9                       # Para 'adam'
  beta2: 0.999                     # Para 'adam'
  eps: 1.0e-8                      # Estabilidad numérica
  schedule:
    name: exp_decay                # 'constant' | 'exp_decay' | 'step'
    gamma: 0.9995                  # Factor de decaimiento por época
  grad_clip: null                  # Float para recorte de gradiente por norma (ej. 5.0) o null

# 6. Parámetros del bucle de entrenamiento
train:
  epochs: 20000
  batch_size: full                 # Entero (ej. 8, 16) o 'full' (batch completo de 32)
  eval_every: 10                   # Calcular métricas globales y loguear cada N épocas
  early_stop:                      # Opcional: criterio de corte anticipado
    metric: max_pixel_errors       # Métrica a monitorear
    threshold: 0                   # Cortar si max_pixel_errors <= threshold
    patience: 200                  # Épocas consecutivas cumpliendo la condición

# 7. Salida y persistencia
output:
  dir: results/runs                # Carpeta destino de la corrida
  save_weights: true               # Si guarda weights.npz al finalizar
```

---

## 2. Configuración de Sweeps (`configs/sweeps/*.yaml`)

Un **Sweep** resuelve la consigna 1.a.2 (*"estudien las diferentes arquitecturas y técnicas de optimización"*) sin requerir decenas de archivos YAML manuales ni código duplicado.

### Anatomía de un archivo de Sweep

```yaml
# configs/sweeps/a2_arquitecturas.yaml

# 1. Configuración base: hereda todos los defaults
base: configs/ae_basic.yaml

# 2. Semillas: cada configuración se corre con estas semillas para evaluar robustez
seeds: [1, 2, 3]

# 3. Grilla de exploración (producto cartesiano de los ejes)
grid:
  optimizer.name: [adam, momentum, sgd]
  loss.recon: [bce, mse]
  model.init: [xavier, he]
  model.encoder.layers: [arch_small, arch_medium, arch_deep]

# 4. Variantes complejas (alias referenciados en grid)
variants:
  arch_small:
    - {units: 16, activation: tanh}
    - {units: 2,  activation: identity}
  arch_medium:
    - {units: 24, activation: tanh}
    - {units: 12, activation: tanh}
    - {units: 2,  activation: identity}
  arch_deep:
    - {units: 30, activation: tanh}
    - {units: 20, activation: tanh}
    - {units: 10, activation: tanh}
    - {units: 2,  activation: identity}
```

---

## 3. ¿Cómo funciona el mecanismo de Sweep internamente?

El runner de sweeps (`scripts/sweep.py`) ejecuta el siguiente pipeline:

### 1. Resolución por Dot-Notation
Las claves del diccionario `grid` utilizan notación por puntos (`section.subsection.key`) para sobreescribir campos en profundidad dentro de la configuración base:
- `optimizer.name: adam` $\to$ modifica `cfg['optimizer']['name'] = 'adam'`
- `model.encoder.layers: arch_small` $\to$ busca en `variants['arch_small']` y asigna la lista correspondiente en `cfg['model']['encoder']['layers']`. También actualiza simétricamente el decodificador si la arquitectura lo requiere.

### 2. Producto Cartesiano
Si el `grid` define:
- 3 optimizadores $\times$ 2 pérdidas $\times$ 2 inicializaciones $\times$ 3 arquitecturas = **36 combinaciones**.
- Multiplicado por las 3 semillas de `seeds` = **108 corridas individuales**.

> [!TIP]
> **Estrategia escalonada recomendada por el plan:**
> En vez de un grid gigante inicial de 108 corridas, se pueden correr sweeps modulares y focalizados:
> 1. `sweep_opt_lr.yaml`: optimizadores (`adam`, `momentum`, `sgd`) vs lr ($10^{-2}, 10^{-3}, 10^{-4}$).
> 2. `sweep_loss_act.yaml`: pérdidas (`bce`, `mse`) vs activaciones (`tanh`, `relu`, `sigmoid`).
> 3. `sweep_depth.yaml`: anchos y profundidades de la capa latente y capas ocultas.

### 3. Asignación de `run_id` y aislamiento
Por cada corrida generada se calcula un identificador determinista:
$$\text{run\_id} = \langle\text{base\_name}\rangle\_\langle\text{hash\_hiperparámetros}\rangle\_\text{s}\langle\text{seed}\rangle$$

Por ejemplo: `ae_opt-adam_loss-bce_arch-medium_s1`.

### 4. Estructura de resultados producida
Al ejecutarse el sweep:
1. **Registro global consolidado (`results/summary/runs.csv`)**:
   Cada corrida que finaliza escribe atómicamente una fila con su configuración expandida y sus métricas finales (`final_loss`, `max_pixel_errors`, `n_patterns_ok`, `epoch_goal_reached`, `train_time_s`, etc.).
2. **Directorio individual (`results/runs/<run_id>/`)**:
   - `config_resolved.yaml`: configuración completa y unívoca con la que corrió.
   - `epochs.csv`: evolución histórica época a época (loss, pixel errors, gradient norm, etc.).
   - `weights.npz`: pesos finales de la red.
   - `latent.csv`: coordenadas $(z_1, z_2)$ de los 32 patrones.

---

## 4. Ejemplos de uso por CLI

```bash
# 1. Ejecutar una corrida individual
python scripts/run.py --config configs/ae_basic.yaml

# 2. Ejecutar un sweep completo
python scripts/sweep.py --sweep configs/sweeps/a2_arquitecturas.yaml

# 3. Dry-run del sweep (imprime cuántas corridas son y sus nombres sin entrenar)
python scripts/sweep.py --sweep configs/sweeps/a2_arquitecturas.yaml --dry-run
```
