# Hito 1: Capa de Datos y Preprocesamiento

## 1. Resumen de lo realizado

En este hito se implementó y validó la **FASE 1: Datos** del plan de trabajo, garantizando la carga desacoplada, procesamiento, persistencia y reproducibilidad de los dos conjuntos de datos del TP:

1. **Dataset de caracteres (`font.h`)**:
   - Módulo [`tp5.data.fontparser`](file:///c:/Users/Matias%20Cuneo%20Gima/ITBA/SIA/TPV-SIA/src/tp5/data/fontparser.py) para extraer y procesar los 32 caracteres desde su representación C (`int const Font3[32][7]`).
   - Conversión de bits: los 5 bits menos significativos de cada byte representan las columnas (bit 4 a la izquierda, bit 0 a la derecha).
   - Aplanado de matriz $7 \times 5$ a vector binario de 35 componentes.
   - Cálculo de la distancia de Hamming mínima entre pares (valor obtenido: **2**, correspondiente a pares críticos como `'l'` vs `'|'`).
   - Guardado en [`data/processed/font.npz`](file:///c:/Users/Matias%20Cuneo%20Gima/ITBA/SIA/TPV-SIA/data/processed/font.npz) con etiquetas y forma original.

2. **Dataset de sprites (`tileset_colored.tsx` y `tileset_legacy.png`)**:
   - Módulo [`tp5.data.sprites`](file:///c:/Users/Matias%20Cuneo%20Gima/ITBA/SIA/TPV-SIA/src/tp5/data/sprites.py) para parsear la geometría del tileset (XML TSX de Tiled) y cortar tiles $16 \times 16$.
   - Detección automática de color de fondo y descarte de tiles vacíos.
   - Deduplicación exacta de tiles y partición train/val (80/20) reproducible con semilla.
   - Soporte para canales RGB ($16 \times 16 \times 3 = 768$ features) y escala de grises ($256$ features), normalizados en $[0, 1]$.
   - Persistencia en [`data/processed/sprites.npz`](file:///c:/Users/Matias%20Cuneo%20Gima/ITBA/SIA/TPV-SIA/data/processed/sprites.npz) y metadatos en [`data/processed/sprites_meta.json`](file:///c:/Users/Matias%20Cuneo%20Gima/ITBA/SIA/TPV-SIA/data/processed/sprites_meta.json).

3. **Módulo de ruido para Denoising Autoencoder**:
   - Módulo [`tp5.data.noise`](file:///c:/Users/Matias%20Cuneo%20Gima/ITBA/SIA/TPV-SIA/src/tp5/data/noise.py) con funciones puras para inyectar distorsiones controladas:
     - `flip`: invierte bits con probabilidad $p$ (apto para binario).
     - `mask`: setea píxeles a 0 con probabilidad $p$ (dropout de entrada).
     - `gaussian`: ruido normal aditivo acotado a $[0, 1]$ (apto para imágenes continuas).
   - Control estricto de reproducibilidad mediante pasaje explícito de instancias de `np.random.Generator`.

4. **Interfaz unificada de carga (`tp5.data.loader`)**:
   - Punto de acceso desacoplado para el bucle de entrenamiento, aislando al optimizador y al modelo de las particularidades de lectura de disco o tipos de archivo.

5. **Validación visual y de integridad**:
   - Suite de prueba ejecutable [`tests/test_data.py`](file:///c:/Users/Matias%20Cuneo%20Gima/ITBA/SIA/TPV-SIA/tests/test_data.py) que valida dimensiones, rangos, unicidad y genera artefactos de inspección en [`tests/output/`](file:///c:/Users/Matias%20Cuneo%20Gima/ITBA/SIA/TPV-SIA/tests/output/):
     - `font_render.txt`: renderizado ASCII de las 32 matrices de caracteres.
     - `sprites_grid.png`: grilla visual con muestra de tiles procesados.
     - `sprites_meta.txt`: resumen del preprocesamiento.

---

## 2. Contratos de Interfaz

Los siguientes contratos rigen la interacción entre los módulos de datos y las capas posteriores (red neuronal, trainer, evaluador y scripts de generación de figuras).

### Contrato 1: Especificación en Configuración (YAML)

Cualquier archivo de corrida (`run.yaml`) o dataset (`dataset.yaml`) debe incluir la sección `data:` con la siguiente estructura esperada:

```yaml
data:
  dataset: "font"             # 'font' | 'sprites'
  path: "data/processed/font.npz"
  val_split: 0.0              # float en [0.0, 1.0)
  seed: 42                    # int para partición reproducible
```

Para sprites pueden incluirse además los campos:
```yaml
data:
  dataset: "sprites"
  path: "data/processed/sprites.npz"
  channels: "rgb"             # 'rgb' | 'gray'
  val_split: 0.2
  seed: 42
```

---

### Contrato 2: `tp5.data.loader.load_dataset(cfg)`

Es la única función que deben llamar el `Trainer` y los scripts de entrenamiento para obtener los datos.

* **Firma**:
  ```python
  def load_dataset(cfg: dict) -> dict: ...
  ```
* **Entrada**:
  - `cfg`: diccionario conteniendo la clave `"data"`.
* **Salida**:
  Diccionario con las siguientes claves obligatorias:
  | Clave | Tipo | Descripción |
  |---|---|---|
  | `X_train` | `np.ndarray` (dtype `np.float64`) | Matriz 2D de dimensiones `(N_train, N_features)`. Para `font`, `(32, 35)`. Para `sprites`, `(N_train, 768)`. |
  | `X_val` | `np.ndarray` o `None` | Matriz 2D `(N_val, N_features)` con dtype `np.float64`, o `None` si `val_split == 0.0`. |
  | `labels` | `list[str]` o `None` | Lista de `N_train + N_val` strings con los identificadores de patrones (p. ej. `['`', 'a', ...]`), o `None` si el dataset no tiene etiquetas. |
  | `image_shape` | `tuple[int, ...]` | Tupla con las dimensiones espaciales para reconstrucción: `(7, 5)` para `font`, `(16, 16, 3)` o `(16, 16)` para `sprites`. |

* **Garantías**:
  - `X_train` y `X_val` siempre devuelven `float64` listos para multiplicaciones matriciales en el motor de red.
  - Los valores se encuentran acotados en el rango $[0.0, 1.0]$.
  - Las muestras están siempre vectorizadas (aplanadas en dimensión 1). La reconstrucción a imagen se hace vía `x.reshape(ds["image_shape"])`.

---

### Contrato 3: `tp5.data.noise.corrupt(X, kind, level, rng)`

Utilizado por el `Trainer` en cada época del Denoising Autoencoder y por el módulo de evaluación de robustez.

* **Firma**:
  ```python
  def corrupt(X: np.ndarray, kind: str, level: float, rng: np.random.Generator) -> np.ndarray: ...
  ```
* **Parámetros**:
  - `X`: `np.ndarray` con shape `(N, features)` o `(features,)`. No es modificado in-place (función pura).
  - `kind`: `'flip'` | `'mask'` | `'gaussian'`.
  - `level`: `float >= 0.0`. Representa probabilidad de afectación o desviación estándar del ruido.
  - `rng`: Instancia de `np.random.Generator` (creada p. ej. vía `np.random.default_rng(seed)`).
* **Salida**:
  - `np.ndarray` con el mismo shape y dtype de `X`.
* **Garantía**:
  - Idéntica semilla de `rng` produce idéntica matriz corrompida.
  - La pérdida del DAE siempre se calculará como $\mathcal{L}(\hat{X}_{\text{pred}}, X_{\text{clean}})$, donde $\hat{X}_{\text{pred}} = \text{forward}(\text{corrupt}(X_{\text{clean}}))$.

---

### Contrato 4: Persistencia y Formato de Archivos `.npz`

Los scripts de preprocesamiento deben generar los `.npz` respetando las claves internas estándar:

1. `data/processed/font.npz`:
   - `X`: `uint8` de forma `(32, 35)`.
   - `labels`: arreglo 1D de cadenas de texto `(32,)`.
   - `shape`: arreglo `[7, 5]`.
2. `data/processed/sprites.npz`:
   - `X_train`: `float32` de forma `(N_train, 768)`.
   - `X_val`: `float32` de forma `(N_val, 768)` (opcional).
   - `tile_index_train`: `int32` de forma `(N_train, 2)` con coordenadas `(fila, col)` de origen.
   - `tile_index_val`: `int32` de forma `(N_val, 2)` (opcional).
   - `shape`: arreglo `[16, 16, 3]` o `[16, 16]`.
