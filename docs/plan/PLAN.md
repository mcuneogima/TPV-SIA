# TP5 Deep Learning: Plan de implementación y documentación

Sistemas de Inteligencia Artificial. Autoencoders, Denoising Autoencoder y VAE.

## 0. Alcance, restricciones y principios

**Restricciones de la cátedra.** Prohibido PyTorch/TensorFlow o cualquier librería que resuelva la red. Permitido: `numpy` (todo el cómputo de la red), `pyyaml` (configs), `pandas` o `csv` (lectura/escritura de resultados), `matplotlib` (gráficos), `Pillow` (solo para leer imágenes de sprites, equivalente a leer un CSV). Conviene confirmar con la cátedra el uso de Pillow; si no se permite, convertir los sprites a `.npy` fuera del repo con un script aparte.

**Principios de diseño.**

- Todo experimento es reproducible desde un único YAML + una semilla.
- El motor de red no sabe nada de `font.h`, ni de gráficos, ni de la consigna: solo recibe arrays y configuración.
- Los scripts de gráficos no entrenan: solo leen lo que dejó el motor en disco (CSVs y pesos). Así se pueden rehacer gráficos sin reentrenar.
- Cada componente variable (activación, pérdida, optimizador, inicialización, tipo de AE) se elige por nombre en el YAML mediante un registro (diccionario nombre → clase).
- No se implementa nada que no responda a una pregunta de la consigna o que no produzca un resultado analizable.

**Fuera de alcance (decisión explícita).** Capas convolucionales: el backward de convolución sumaría mucho código y los sprites se mantienen chicos (16x16) para que un MLP alcance. Batch norm y dropout tampoco: con 32 muestras donde memorizar es el objetivo, regularizar va en contra.

## 1. Estructura del repositorio

```
tp5/
  configs/                 # YAMLs de entrada (una corrida o un sweep)
    ae_basic.yaml
    dae.yaml
    vae_sprites.yaml
    sweeps/a2_arquitecturas.yaml
  data/
    raw/font.h
    raw/sprites/*.png
    processed/font.npz
    processed/sprites.npz
  src/tp5/
    data/        fontparser.py  sprites.py  noise.py  loader.py
    nn/          layers.py  activations.py  losses.py  optimizers.py  init.py  schedules.py
    models/      autoencoder.py  vae.py  registry.py
    train/       trainer.py  metrics.py  config.py  io.py  sweep.py
    analysis/    latent.py  generate.py  noise_eval.py
  scripts/       run.py  sweep.py  plot_*.py
  results/       summary/runs.csv   runs/<run_id>/...
  figures/       a2/  a3/  a4/  b/  vae/
  docs/          este plan + informe
```

Comandos principales:

- `python scripts/run.py --config configs/ae_basic.yaml` entrena una corrida.
- `python scripts/sweep.py --sweep configs/sweeps/a2_arquitecturas.yaml` expande un grid y corre todas.
- `python scripts/plot_<nombre>.py --run results/runs/<run_id>` genera los gráficos de una parte.

---

## FASE 1: Datos

### 1.1 Procesador de `font.h`

**Entrada.** `data/raw/font.h`: arreglo C con 32 caracteres, cada uno 7 filas (bytes), cada byte con 5 bits útiles (columnas). Se espera líneas del estilo `{0x04, 0x04, 0x02, 0x00, 0x00, 0x00, 0x00}, // 0x60, ...`.

**Parser (`fontparser.py`).**

1. Leer el archivo como texto y extraer con regex todos los literales `0x[0-9a-fA-F]{2}`, agrupados de a 7 por caracter (o por cada par de llaves). Ignorar comentarios para el conteo de bytes pero capturarlos para las etiquetas.
2. Por cada byte tomar los 5 bits menos significativos. Convención a verificar visualmente: bit 4 es la columna izquierda, bit 0 la derecha.
3. Armar matriz 7x5 y aplanar por filas a vector de 35 elementos.
4. Etiquetas: si la fuente trae comentarios con el caracter, usarlos; si no, usar el código ASCII (en esta fuente típicamente 0x60 a 0x7F) y mostrar el caracter imprimible o su código.

**Validaciones obligatorias (en tests).**

- Exactamente 32 patrones, todos de 35 bits, valores en {0,1}.
- Todos los patrones distintos (si hubiera duplicados, el objetivo de 1 píxel de error sería ambiguo).
- Distancia de Hamming mínima entre pares: se reporta en el informe, porque es la cota natural del problema (si dos letras difieren en 2 píxeles, un error de 1 píxel ya es peligroso).
- Render ASCII de los 32 caracteres para inspección manual.

**Salida.** `data/processed/font.npz` con:

- `X`: uint8 (32, 35)
- `labels`: array de strings (32)
- `shape`: (7, 5)
- `source_sha1`: hash del `font.h` (trazabilidad)

### 1.2 Dataset nuevo: sprites

**Entrada.** tileset_legacy.png + tileset_colored.tsx (1024 tiles de 16x16, separación de 1 px, 32 columnas), en data/raw/sprites/. Licencia a confirmar en la página del pack.

**Pipeline (`sprites.py`).**

- Leer geometría desde el .tsx y cortar la hoja en tiles (sin redimensionar).
- Detectar el color de fondo como el más frecuente y descartar los tiles que son solo fondo.
- Descartar tiles duplicados exactos.
- Seleccionar el subconjunto temático según regions del YAML (rectángulos de filas y columnas de tiles), definidos tras revisar la hoja de contacto.
- Convertir a float32 en [0,1] y aplanar (channels: rgb da 768, gray da 256).
- Separar train/val (80/20, semilla fija), agrupando familias casi idénticas del mismo lado.
- Opcional: flip horizontal en train, solo para familias donde tenga sentido.

**Ejemplo de YAML (`dataset.yaml`).**
```yaml
data:
  dataset: sprites
  tsx: data/raw/sprites/tileset_colored.tsx
  png: data/raw/sprites/tileset_legacy.png
  channels: rgb            # rgb | gray
  regions:                 # a completar tras mirar la hoja de contacto
    - {name: personajes, rows: [0, 10], cols: [24, 31]}
  dedup: true
  hflip_augment: false
  val_split: 0.2
```

**Salida.** data/processed/sprites.npz con X_train, X_val, tile_index_train, tile_index_val (fila, columna de origen), shape. meta.json con hash del .tsx, color de fondo, regiones elegidas, cantidad de tiles descartados (vacíos y duplicados) y parámetros del split. Figura figures/data/contact_sheet.png: hoja con los índices de fila y columna, usada para elegir las regiones, más una grilla de muestras del subconjunto final.

### 1.3 Ruido (`noise.py`) para el Denoising Autoencoder

Funciones puras `corrupt(X, kind, level, rng)`:

- `flip`: cada bit se invierte con probabilidad `level` (principal, natural para binarios).
- `mask`: cada píxel se fuerza a 0 con probabilidad `level` (comparación útil: ¿se tolera igual de bien un ruido que el otro?).
- Gaussiano solo si se trabaja con sprites continuos (no se usa en `font.h`).

El generador `rng` se pasa explícito para que el ruido sea reproducible y regenerable por época.

### 1.4 Interfaz del módulo de datos (`loader.py`)

`load_dataset(cfg) -> Dataset` con `X_train`, `X_val` (puede ser None), `labels`, `image_shape`. Toda corrida entra por acá, así el trainer es agnóstico del dataset.


---

## FASE 2: Motor de la red

### 2.1 Convenciones

- Los datos van en matrices `(batch, features)`. Todo en `float64` (la estabilidad numérica ayuda y el costo es despreciable a este tamaño).
- Cada capa implementa `forward(x)` guardando lo necesario en caché y `backward(grad_out)` devolviendo `grad_in` y dejando gradientes en sus parámetros.
- Pérdida por muestra = suma sobre píxeles; pérdida del batch = promedio sobre muestras (mantiene comparables corridas con distinto batch size y entradas de distinto tamaño).

### 2.2 Componentes (con su justificación)

| Módulo | Opciones | Por qué está |
| --- | --- | --- |
| Capas | `dense` | Única necesaria para MLP |
| Activaciones | `identity`, `sigmoid`, `tanh`, `relu`, `leaky_relu` | `identity` para la capa latente, `sigmoid` en la salida binaria, `tanh` y `relu`/`leaky_relu` para comparar en a.2 (vanishing gradients, neuronas muertas) |
| Pérdidas | `mse`, `bce` | Comparar en a.2: BCE con sigmoide da gradientes más fuertes para el objetivo de píxel exacto |
| Optimizadores | `sgd`, `momentum`, `rmsprop`, `adam` | Escalera clásica; Adam será el candidato, el resto sirve para mostrar por qué |
| Schedules de lr | `constant`, `exp_decay`, `step` | Afinar al final del entrenamiento es clave para bajar de 2 a 1 o 0 píxeles erróneos |
| Inicialización | `xavier`, `he`, `normal_small` | Documentar su efecto en convergencia con latente de 2D |
| Extras | `grad_clip` (norma global) | Estabilizar al VAE y a corridas con lr alto |

Todo se instancia desde un registro (`registry.py`): `ACTIVATIONS['relu']`, etc. Agregar una opción nueva es agregar una clase y una línea.

### 2.3 Modelos

**`Autoencoder`** (tipos `basic` y `denoising`, misma clase; difieren en el entrenamiento).

- `encoder`: lista de capas densas hasta la capa latente.
- `decoder`: lista de capas densas hasta la salida.
- Métodos: `encode(X)`, `decode(Z)`, `forward(X)`, `backward(grad)`, `params()`.
- El tipo `denoising` activa en el trainer la corrupción de la entrada y calcula la pérdida contra la entrada limpia.

**`VAE`.**

- `encoder_trunk` (densas) seguido de dos cabezas lineales: `mu` y `logvar`, de dimensión `latent_dim`.
- Forward: `eps ~ N(0,I)`, `z = mu + exp(0.5*logvar) * eps`, `x_hat = decoder(z)`.
- Pérdida: `recon + beta * KL`, con `KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))`.
- Backward (a implementar a mano y validar con gradient check):
  - `dL/dz` viene del decoder.
  - `dL/dmu = dL/dz + beta * mu`
  - `dL/dlogvar = dL/dz * eps * 0.5 * exp(0.5*logvar) + beta * 0.5 * (exp(logvar) - 1)`
  - Ambos gradientes se suman al llegar al tronco del encoder.
- Métodos extra: `sample(n, rng)` (decodifica `z ~ N(0,I)`), `encode_mean(X)` (usa `mu`, sin ruido, para graficar).
- Parámetro `beta` y opcionalmente `beta_warmup_epochs` (subida lineal de 0 a beta) para evitar colapso posterior al inicio.

### 2.4 Trainer

Bucle: para cada época, mezclar datos (con semilla), recorrer mini-batches, forward, pérdida, backward, clip opcional, paso del optimizador, y al final de la época calcular métricas sobre todo el set y escribir una fila al CSV de épocas.

Parámetros: `epochs`, `batch_size` (entero o `full`), `seed`, `early_stop` opcional (corta cuando se cumple el objetivo, por ejemplo `max_pixel_errors <= 1` sostenido N épocas, y lo registra), `eval_every`.

**Métricas por época (todas sobre el set completo, no solo el batch).**

- `loss`, `recon_loss`, `kl_loss` (0 si no es VAE)
- `max_pixel_errors`: máximo, sobre los 32 patrones, de píxeles distintos tras binarizar con umbral 0.5
- `mean_pixel_errors`
- `n_patterns_ok`: cantidad de patrones con error <= 1
- `n_patterns_perfect`: cantidad con error 0
- `grad_norm`, `lr`, `time_s`

En el VAE con sprites las métricas de píxeles no aplican (no son binarios): se usa MSE de reconstrucción en train y val.

### 2.6 Formato de entrada: YAML de corrida

```yaml
run_name: ae_basic_adam_bce
seed: 42

data:
  dataset: font            # font | sprites
  path: data/processed/font.npz
  val_split: 0.0

model:
  type: autoencoder        # autoencoder | denoising | vae
  latent_dim: 2
  encoder:
    layers:
      - {units: 24, activation: tanh}
      - {units: 12, activation: tanh}
      - {units: 2,  activation: identity}   # capa latente
  decoder:
    layers:
      - {units: 12, activation: tanh}
      - {units: 24, activation: tanh}
      - {units: 35, activation: sigmoid}
  init: xavier

loss:
  recon: bce               # bce | mse
  beta: 1.0                # solo vae
  beta_warmup_epochs: 0    # solo vae

noise:                     # solo denoising
  kind: flip               # flip | mask
  level: 0.1
  regenerate_each_epoch: true

optimizer:
  name: adam               # sgd | momentum | rmsprop | adam
  lr: 0.001
  momentum: 0.9
  beta1: 0.9
  beta2: 0.999
  schedule: {name: exp_decay, gamma: 0.9995}
  grad_clip: null

train:
  epochs: 20000
  batch_size: full
  eval_every: 10
  early_stop: {metric: max_pixel_errors, threshold: 0, patience: 200}

output:
  dir: results/runs
  save_weights: true
```

Para el VAE, `encoder.layers` termina antes de las cabezas y se agrega `model.latent_dim`; el motor crea `mu` y `logvar` automáticamente.

**YAML de sweep** (`configs/sweeps/*.yaml`): una configuración base más una lista de ejes a variar. El runner expande el producto cartesiano (o una lista explícita de variantes), y repite cada combinación con varias semillas.

```yaml
base: configs/ae_basic.yaml
seeds: [1, 2, 3]
grid:
  optimizer.name: [sgd, momentum, adam]
  loss.recon: [mse, bce]
  model.encoder.layers: [arch_small, arch_medium, arch_deep]   # definidas en el mismo archivo
  init: [xavier, he]
variants:
  arch_small:  {...}
```

### 2.7 Formatos de salida

**`results/summary/runs.csv`**: una fila por corrida (se agrega al terminar cada una).

| Columna | Descripción |
| --- | --- |
| run\_id | identificador único (nombre + hash corto del config + semilla) |
| run\_name, model\_type, dataset, seed | identificación |
| arch | resumen legible, por ejemplo `35-24-12-2-12-24-35` |
| n\_params | cantidad de parámetros |
| activation\_hidden, recon\_loss\_fn, optimizer, lr0, schedule, init, batch\_size, beta, noise\_kind, noise\_level | hiperparámetros principales |
| epochs\_run | épocas efectivamente ejecutadas |
| final\_loss, final\_recon, final\_kl | métricas finales |
| final\_max\_pixel\_errors, final\_mean\_pixel\_errors | métricas finales de píxeles |
| final\_n\_patterns\_ok, final\_n\_patterns\_perfect | cantidad aprendida |
| best\_n\_patterns\_ok, epoch\_of\_best | mejor punto de la corrida |
| epoch\_goal\_reached | primera época en que se cumplió el objetivo (vacío si nunca) |
| train\_time\_s, status | tiempo total y `ok`/`diverged`/`early_stopped` |
| config\_path | ruta al config resuelto |

**`results/runs/<run_id>/epochs.csv`**: una fila por época evaluada, con columnas `epoch, loss, recon_loss, kl_loss, val_loss, max_pixel_errors, mean_pixel_errors, n_patterns_ok, n_patterns_perfect, grad_norm, lr, time_s`.

**Archivos adicionales por corrida.**

- `config_resolved.yaml`: config completa con defaults aplicados.
- `weights.npz`: todos los parámetros (permite regenerar gráficos y generar letras sin reentrenar).
- `latent.csv`: `index, label, z1, z2` (para VAE además `logvar1, logvar2`), calculado al final.
- `reconstructions.npz`: entradas y salidas finales (y entradas ruidosas en el DAE).
- `noise_eval.csv` (solo DAE, lo escribe `noise_eval.py`): `kind, level, trial, mean_pixel_errors, max_pixel_errors, frac_perfect, frac_ok`.
- `generated.csv` / `.npz`: muestras generadas, con el `z` usado.

---

## FASE 3: Experimentos, gráficos y scripts por ítem de la consigna

Cada script lee de `results/` y escribe en `figures/<parte>/`. A continuación, cada ítem de la consigna con qué se corre, qué se grafica y qué debe concluir el informe.

### Parte 1.a: Autoencoder básico sobre `font.h`

**1.a.1 Arquitectura (Codificador/Decodificador con latente 2D).**

- Qué se entrega: diagrama de capas (script `plot_architecture.py` que dibuja bloques con unidades y activaciones a partir del YAML) y justificación: latente `identity` de 2 unidades, salida `sigmoid` de 35, simetría encoder/decoder, anchos intermedios generosos.

**1.a.2 Estudio de arquitecturas y técnicas de optimización.**

- Experimento: `configs/sweeps/a2_arquitecturas.yaml`. Se barre de forma escalonada, no con un grid gigante: primero optimizador y lr con una arquitectura fija, luego pérdida (mse vs bce), luego activaciones e inicialización, luego profundidad y ancho, y por último schedule de lr y cantidad de épocas con la mejor configuración. Cada punto con 3 semillas.
- Objetivo a verificar: los 32 patrones con `max_pixel_errors <= 1`. Si no se logra, identificar el máximo de patrones aprendidos y explicar por qué.
- Gráficos (`plot_a2.py`):
  - Curvas de `loss` y de `n_patterns_ok` por época, una línea por configuración (media y banda entre semillas).
  - Tabla resumen ordenada (exportada a CSV y a tabla del informe): arquitectura, optimizador, pérdida, patrones aprendidos, máx. píxeles erróneos, épocas hasta el objetivo.
  - Gráfico de barras o boxplot de `final_n_patterns_ok` por optimizador y por pérdida (efecto aislado de cada decisión).
- Si el dataset completo no se aprende: script `plot_failure.py` que muestra los patrones no aprendidos (entrada vs reconstrucción con los píxeles erróneos marcados) y, en el espacio latente, qué puntos quedan superpuestos o cercanos, más la relación con la distancia de Hamming mínima entre letras. Esto responde el pedido de mostrar por qué no se pudo.

**1.a.3 Gráfico del espacio latente.**

- `plot_latent.py`: scatter de `(z1, z2)` de los 32 patrones con la letra como etiqueta de cada punto. Opcional: miniatura del patrón reconstruido junto al punto, y un mapa de fondo con el decoder evaluado en una grilla (permite ver qué zonas del plano producen caracteres legibles y cuáles basura).

**1.a.4 Generar una letra nueva.**

- `generate.py` con tres modos: (a) punto elegido a mano en una zona vacía, (b) punto medio entre dos letras, (c) interpolación lineal en N pasos entre dos letras.
- Gráfico: fila de caracteres decodificados con el recorrido de interpolación y el punto marcado en el scatter latente. Dejar en el informe la evaluación crítica: ¿el resultado parece una letra o es una mezcla sin sentido? Comparar luego con el VAE.

### Parte 1.b: Denoising Autoencoder

**1.b.1 Arquitectura y justificación.**

- Config `dae.yaml`. Hipótesis a justificar: latente más grande que 2 (por ejemplo 8 a 16) porque el cuello de 2D limita la capacidad de corregir ruido, y encoders con algo más de ancho. Se entrena con ruido regenerado en cada época y pérdida contra la entrada limpia.
- Experimento corto de comparación para justificar la elección: DAE con latente 2, 8 y 16 (mismas condiciones), midiendo error de reconstrucción sobre entradas ruidosas.

**1.b.2 Estudio del efecto del ruido.**

- Dos ejes: nivel de ruido en entrenamiento (por ejemplo 0.05, 0.1, 0.2) y nivel en test (0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5), con muchos ensayos por nivel (por ejemplo 100 corrupciones por patrón) para estimaciones estables.
- Script `noise_eval.py` calcula y escribe `noise_eval.csv`; `plot_denoise.py` produce:
  - Curva de píxeles erróneos promedio vs nivel de ruido, comparando entrada ruidosa sin procesar, salida del AE básico y salida del DAE (mide el aporte real del denoising).
  - Curva de fracción de patrones recuperados exactamente vs nivel de ruido.
  - Mapa de calor entrenado en nivel A y evaluado en nivel B.
  - Grilla visual por nivel de ruido: original, ruidosa, reconstruida (con 4 o 5 letras).
  - Comparación `flip` vs `mask`.

### Parte 2: VAE

**2.a Dataset nuevo.** Sprites preprocesados en la Fase 1. Gráfico previo: grilla de muestras del dataset y distribución de colores si es relevante.

**2.b Esquema variacional.**

- Config `vae_sprites.yaml` (encoder tronco, cabezas `mu`/`logvar`, decoder con salida sigmoide, `recon: bce` o `mse`, `beta`).
- Experimentos que justifican decisiones (pocos pero informativos):
  - Barrido de `beta` (por ejemplo 0.1, 0.5, 1, 2): trade-off entre calidad de reconstrucción y regularidad del espacio latente.
  - Latente 2D (visualizable) frente a 8 o 16 (mejor calidad), para elegir el valor que se usa en la generación.
  - Con y sin `beta_warmup_epochs`, si se observa colapso posterior.
- Gráficos (`plot_vae.py`):
  - Curvas por época de `recon_loss` y `kl_loss` por separado, más la total (diagnostica el balance).
  - Scatter 2D de `mu` coloreado por alguna propiedad disponible (por ejemplo categoría del sprite, si existe) con las elipses de `sigma` de unos pocos puntos para mostrar la naturaleza probabilística.
  - Grilla de decodificación del plano latente 2D (cuantiles de la normal) para ver la variedad continua.
  - Reconstrucciones: fila de originales y fila de reconstruidas (train y val).
  - Comparación directa AE determinista vs VAE con el mismo dataset: interpolación entre dos puntos en ambos. Es el argumento central de por qué el esquema variacional soluciona el problema del espacio latente.

**2.c Generación.**

- `generate.py --mode sample`: muestras `z ~ N(0,I)` pasadas por el decoder, grilla de 8x8 de resultados.
- Selección de una muestra que los autores juzguen perteneciente al dataset, con el criterio explícito en el informe (por ejemplo: forma coherente, paleta consistente con el dataset, no copia de un ejemplo de entrenamiento).
- Chequeo anti-memorización: distancia (MSE) de la muestra elegida al ejemplo de entrenamiento más cercano, y mostrar ambos lado a lado.
- Interpolación entre dos muestras reales como complemento.

---

## 4. Mapa consigna → entregable

| Ítem | Scripts / config | Salida clave | En el informe |
| --- | --- | --- | --- |
| 1.a.1 | `ae_basic.yaml`, `plot_architecture.py` | diagrama | Arquitectura y justificación |
| 1.a.2 | `sweeps/a2_arquitecturas.yaml`, `plot_a2.py`, `plot_failure.py` | `runs.csv`, `epochs.csv`, curvas | Tabla comparativa, análisis de qué funcionó, si fue subconjunto el porqué |
| 1.a.3 | `plot_latent.py` | scatter etiquetado | Estructura del espacio latente |
| 1.a.4 | `generate.py` | letras nuevas e interpolación | Evaluación crítica |
| 1.b.1 | `dae.yaml`, comparación de latentes | tabla | Elección de arquitectura |
| 1.b.2 | `noise_eval.py`, `plot_denoise.py` | `noise_eval.csv`, curvas, grillas | Capacidad de eliminar ruido por nivel |
| 2.a | `sprites.py` | `sprites.npz`, grilla | Descripción del dataset |
| 2.b | `vae_sprites.yaml`, barridos de beta y latente, `plot_vae.py` | curvas recon/KL, grilla latente | Cómo el esquema variacional resuelve el problema |
| 2.c | `generate.py --mode sample` | muestras, interpolaciones, vecino más cercano | Muestra elegida y criterio |

## 5. Orden de implementación e hitos

1. **Hito 1 (datos):** parser de `font.h`, validaciones, render de los 32 caracteres, pipeline de sprites, ruido.
2. **Hito 2 (núcleo):** dense + activaciones + pérdidas + SGD, con gradient check. Primer AE entrenando sobre 2 o 3 patrones, momentum, RMSprop, Adam, schedules, inicializaciones, clipping. Trainer completo con escritura de CSVs y YAML parser.
4. **Hito 3 (parte 1.a):** sweep escalonado, gráficos de a.2, latente, generación.
5. **Hito 4 (parte 1.b):** DAE, evaluación de ruido, gráficos.
6. **Hito 5 (VAE):** reparametrización, KL, gradient check del VAE completo, experimentos de beta y dimensión, generación.