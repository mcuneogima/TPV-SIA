# Hito 3: Estudio de Hiperparámetros, Espacio Latente y Generación (FASE 3 Parte 1.a)

## 1. Resumen de lo realizado

En este hito se implementaron, validaron y ejecutaron los scripts de análisis, visualización y generación correspondientes a la **FASE 3: Parte 1.a (Autoencoder básico sobre `font.h`)** del plan de trabajo, procesando los resultados del barrido de screening según la metodología **One-Factor-At-A-Time (OFAT)** persistidos en [`results/summary/runs.csv`](file:///c:/Users/Matias%20Cuneo%20Gima/ITBA/SIA/TPV-SIA/results/summary/runs.csv) y [`results/runs/`](file:///c:/Users/Matias%20Cuneo%20Gima/ITBA/SIA/TPV-SIA/results/runs/).

### Configuración Base del Análisis OFAT

Para aislar limpiamente el impacto de cada decisión sin variables de confusión, el screening toma la siguiente configuración base:
- **Arquitectura**: `35-24-12-2-12-24-35` (Medium)
- **Activación Oculta**: `tanh`
- **Función de Pérdida**: `bce`
- **Optimizador**: `adam`
- **Tasa de Aprendizaje (`lr0`)**: `0.001`
- **Schedule de LR**: `exp_decay`
- **Inicialización**: `xavier`
- **Tamaño de Batch**: `full` (32 muestras)

A partir de esta base, se varía un factor a la vez con 3 semillas independientes ($s \in \{1, 2, 3\}$), totalizando 72 corridas.

---

## 2. Componentes y Scripts Implementados

1. **Diagrama de Arquitectura de Red (Consigna 1.a.1)**:
   - Script [`scripts/plot_architecture.py`](file:///c:/Users/Matias%20Cuneo%20Gima/ITBA/SIA/TPV-SIA/scripts/plot_architecture.py): Genera diagramas esquemáticos con dimensiones de capa, funciones de activación, flujo encoder $\to$ cuello de botella 2D $\to$ decoder $\to$ salida sigmoide $7 \times 5$, guardado en [`figures/a2/architecture.png`](file:///c:/Users/Matias%20Cuneo%20Gima/ITBA/SIA/TPV-SIA/figures/a2/architecture.png).

2. **Estudio de Hiperparámetros por Familias OFAT (Consigna 1.a.2)**:
   - Script [`scripts/plot_a2.py`](file:///c:/Users/Matias%20Cuneo%20Gima/ITBA/SIA/TPV-SIA/scripts/plot_a2.py): Automatiza la lectura de historiales `epochs.csv` y genera curvas por época con promedio y banda de desvío estándar ($\pm 1\sigma$) entre semillas para cada factor:
     - **Familia 1: Activaciones Ocultas** (`tanh`, `relu`, `leakyrelu`, `sigmoid`): [`figures/a2/family_activation.png`](file:///c:/Users/Matias%20Cuneo%20Gima/ITBA/SIA/TPV-SIA/figures/a2/family_activation.png)
     - **Familia 2: Arquitectura y Profundidad** (`small`, `medium`, `deep`): [`figures/a2/family_architecture.png`](file:///c:/Users/Matias%20Cuneo%20Gima/ITBA/SIA/TPV-SIA/figures/a2/family_architecture.png)
     - **Familia 3: Función de Pérdida** (`bce` vs `mse`): [`figures/a2/family_loss_function.png`](file:///c:/Users/Matias%20Cuneo%20Gima/ITBA/SIA/TPV-SIA/figures/a2/family_loss_function.png) y [`figures/a2/family_loss_pixel_errors.png`](file:///c:/Users/Matias%20Cuneo%20Gima/ITBA/SIA/TPV-SIA/figures/a2/family_loss_pixel_errors.png)
     - **Familia 4: Tamaño de Batch** (mini-batch 8 vs full-batch 32): [`figures/a2/family_batch_size.png`](file:///c:/Users/Matias%20Cuneo%20Gima/ITBA/SIA/TPV-SIA/figures/a2/family_batch_size.png)
     - **Familia 5: Esquema de Inicialización** (Xavier, He, Normal Small): [`figures/a2/family_initialization.png`](file:///c:/Users/Matias%20Cuneo%20Gima/ITBA/SIA/TPV-SIA/figures/a2/family_initialization.png)
     - **Familia 6: Optimizadores** (SGD, Momentum, RMSprop, Adam): [`figures/a2/family_optimizer.png`](file:///c:/Users/Matias%20Cuneo%20Gima/ITBA/SIA/TPV-SIA/figures/a2/family_optimizer.png)
     - **Familia 7: Tasa de Aprendizaje Inicial** (1e-4, 1e-3, 1e-2): [`figures/a2/family_lr.png`](file:///c:/Users/Matias%20Cuneo%20Gima/ITBA/SIA/TPV-SIA/figures/a2/family_lr.png)
     - **Familia 8: Schedules de LR** (Decaimiento exponencial vs Constante): [`figures/a2/family_schedule.png`](file:///c:/Users/Matias%20Cuneo%20Gima/ITBA/SIA/TPV-SIA/figures/a2/family_schedule.png)
   - **Sensibilidad OFAT global**: [`figures/a2/boxplot_families.png`](file:///c:/Users/Matias%20Cuneo%20Gima/ITBA/SIA/TPV-SIA/figures/a2/boxplot_families.png) en una grilla de 2x4 boxplots con distribuciones sin variables confundidas.
   - **Ranking Top 10**: [`figures/a2/top_performers.png`](file:///c:/Users/Matias%20Cuneo%20Gima/ITBA/SIA/TPV-SIA/figures/a2/top_performers.png) con barras de error acotadas en $[0, 32]$ y etiquetas numéricas alineadas verticalmente a la derecha en una columna dedicada sin solapamientos.

3. **Exploración del Espacio Latente 2D (Consigna 1.a.3)**:
   - [`scripts/plot_latent.py`](file:///c:/Users/Matias%20Cuneo%20Gima/ITBA/SIA/TPV-SIA/scripts/plot_latent.py):
     - Scatter 2D de los 32 patrones etiquetados con su glifo ASCII ([`figures/a3/latent_space.png`](file:///c:/Users/Matias%20Cuneo%20Gima/ITBA/SIA/TPV-SIA/figures/a3/latent_space.png)).
     - Grilla continua de reconstrucción latente (Manifold Grid $16 \times 16$) generada evaluando el decoder sobre una malla regular ([`figures/a3/latent_manifold.png`](file:///c:/Users/Matias%20Cuneo%20Gima/ITBA/SIA/TPV-SIA/figures/a3/latent_manifold.png)).

4. **Generación e Interpolación de Letras (Consigna 1.a.4)**:
   - [`scripts/generate.py`](file:///c:/Users/Matias%20Cuneo%20Gima/ITBA/SIA/TPV-SIA/scripts/generate.py):
     - **Modo (a) Punto manual**: decodificación de coordenadas $(z_1, z_2)$ arbitrarias ([`figures/a4/generate_point.png`](file:///c:/Users/Matias%20Cuneo%20Gima/ITBA/SIA/TPV-SIA/figures/a4/generate_point.png)).
     - **Modo (b) Punto medio**: $z_{\text{mid}} = \frac{z_A + z_B}{2}$ entre caracteres clave ([`figures/a4/generate_midpoint_c_o.png`](file:///c:/Users/Matias%20Cuneo%20Gima/ITBA/SIA/TPV-SIA/figures/a4/generate_midpoint_c_o.png) y [`figures/a4/generate_midpoint_e_f.png`](file:///c:/Users/Matias%20Cuneo%20Gima/ITBA/SIA/TPV-SIA/figures/a4/generate_midpoint_e_f.png)).
     - **Modo (c) Interpolación lineal**: trayectoria continua $z(t)$ en $N$ pasos ([`figures/a4/generate_interpolation_c_o.png`](file:///c:/Users/Matias%20Cuneo%20Gima/ITBA/SIA/TPV-SIA/figures/a4/generate_interpolation_c_o.png) y [`figures/a4/generate_interpolation_a_z.png`](file:///c:/Users/Matias%20Cuneo%20Gima/ITBA/SIA/TPV-SIA/figures/a4/generate_interpolation_a_z.png)).

---

## 3. Catálogo Completo de Figuras

| Archivo | Sección | Factor / Contenido |
|---|---|---|
| [`figures/a2/architecture.png`](file:///c:/Users/Matias%20Cuneo%20Gima/ITBA/SIA/TPV-SIA/figures/a2/architecture.png) | 1.a.1 | Diagrama en bloques de la red Autoencoder con cuello latente 2D. |
| [`figures/a2/family_optimizer.png`](file:///c:/Users/Matias%20Cuneo%20Gima/ITBA/SIA/TPV-SIA/figures/a2/family_optimizer.png) | 1.a.2 | Optimizadores: SGD vs Momentum vs RMSprop vs Adam. |
| [`figures/a2/family_lr.png`](file:///c:/Users/Matias%20Cuneo%20Gima/ITBA/SIA/TPV-SIA/figures/a2/family_lr.png) | 1.a.2 | Tasa de aprendizaje inicial: 1e-4 vs 1e-3 vs 1e-2. |
| [`figures/a2/family_schedule.png`](file:///c:/Users/Matias%20Cuneo%20Gima/ITBA/SIA/TPV-SIA/figures/a2/family_schedule.png) | 1.a.2 | Schedules de LR: exp_decay vs constant. |
| [`figures/a2/family_activation.png`](file:///c:/Users/Matias%20Cuneo%20Gima/ITBA/SIA/TPV-SIA/figures/a2/family_activation.png) | 1.a.2 | Funciones de activación: tanh vs relu vs leakyrelu vs sigmoid. |
| [`figures/a2/family_architecture.png`](file:///c:/Users/Matias%20Cuneo%20Gima/ITBA/SIA/TPV-SIA/figures/a2/family_architecture.png) | 1.a.2 | Arquitecturas: Small (35-16-2) vs Medium (35-24-12-2) vs Deep (35-30-20-10-2). |
| [`figures/a2/family_loss_function.png`](file:///c:/Users/Matias%20Cuneo%20Gima/ITBA/SIA/TPV-SIA/figures/a2/family_loss_function.png) | 1.a.2 | Comparación de pérdida y patrones OK: BCE vs MSE. |
| [`figures/a2/family_loss_pixel_errors.png`](file:///c:/Users/Matias%20Cuneo%20Gima/ITBA/SIA/TPV-SIA/figures/a2/family_loss_pixel_errors.png) | 1.a.2 | Error medio de píxeles: BCE vs MSE. |
| [`figures/a2/family_batch_size.png`](file:///c:/Users/Matias%20Cuneo%20Gima/ITBA/SIA/TPV-SIA/figures/a2/family_batch_size.png) | 1.a.2 | Tamaño de batch: Mini-batch (8) vs Full-batch (32). |
| [`figures/a2/family_initialization.png`](file:///c:/Users/Matias%20Cuneo%20Gima/ITBA/SIA/TPV-SIA/figures/a2/family_initialization.png) | 1.a.2 | Inicialización: Xavier vs He vs Normal Small (control negativo). |
| [`figures/a2/boxplot_families.png`](file:///c:/Users/Matias%20Cuneo%20Gima/ITBA/SIA/TPV-SIA/figures/a2/boxplot_families.png) | 1.a.2 | Grilla de boxplots OFAT para los 8 factores explorados. |
| [`figures/a2/top_performers.png`](file:///c:/Users/Matias%20Cuneo%20Gima/ITBA/SIA/TPV-SIA/figures/a2/top_performers.png) | 1.a.2 | Ranking Top 10 con números alineados limpiamente a la derecha. |
| [`figures/a3/latent_space.png`](file:///c:/Users/Matias%20Cuneo%20Gima/ITBA/SIA/TPV-SIA/figures/a3/latent_space.png) | 1.a.3 | Scatter 2D del espacio latente con las 32 letras etiquetadas. |
| [`figures/a3/latent_manifold.png`](file:///c:/Users/Matias%20Cuneo%20Gima/ITBA/SIA/TPV-SIA/figures/a3/latent_manifold.png) | 1.a.3 | Malla continua de reconstrucción del decoder ($16 \times 16$). |
| [`figures/a4/generate_point.png`](file:///c:/Users/Matias%20Cuneo%20Gima/ITBA/SIA/TPV-SIA/figures/a4/generate_point.png) | 1.a.4 | Generación de caracter a partir de punto manual arbitrario. |
| [`figures/a4/generate_midpoint_c_o.png`](file:///c:/Users/Matias%20Cuneo%20Gima/ITBA/SIA/TPV-SIA/figures/a4/generate_midpoint_c_o.png) | 1.a.4 | Caracter punto medio entre `'c'` y `'o'`. |
| [`figures/a4/generate_midpoint_e_f.png`](file:///c:/Users/Matias%20Cuneo%20Gima/ITBA/SIA/TPV-SIA/figures/a4/generate_midpoint_e_f.png) | 1.a.4 | Caracter punto medio entre `'e'` y `'f'`. |
| [`figures/a4/generate_interpolation_c_o.png`](file:///c:/Users/Matias%20Cuneo%20Gima/ITBA/SIA/TPV-SIA/figures/a4/generate_interpolation_c_o.png) | 1.a.4 | Tira de interpolación en 7 pasos entre `'c'` y `'o'`. |
| [`figures/a4/generate_interpolation_a_z.png`](file:///c:/Users/Matias%20Cuneo%20Gima/ITBA/SIA/TPV-SIA/figures/a4/generate_interpolation_a_z.png) | 1.a.4 | Tira de interpolación en 9 pasos entre `'a'` y `'z'`. |

---

## 4. Guía de Ejecución de los Scripts

```bash
# 1. Diagrama de arquitectura
python scripts/plot_architecture.py --config configs/ae_basic.yaml

# 2. Generación de gráficos OFAT por familias, boxplots y top performers
python scripts/plot_a2.py --summary results/summary/runs.csv --output-dir figures/a2

# 3. Espacio latente y variedad continua (selecciona automáticamente la mejor corrida)
python scripts/plot_latent.py --summary results/summary/runs.csv --output-dir figures/a3

# 4. Generación e interpolaciones de caracteres
python scripts/generate.py --mode all --output-dir figures/a4
```
