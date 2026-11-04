# Contexto estable del proyecto

> Este archivo cambia poco. El estado vivo está en `STATE.md`. Si retomas trabajo, lee `STATE.md` primero.

## 1. Qué se investiga

Combinar **teoría de la información / geometría de Fisher** con **aprendizaje automático** para explicar *por qué* y *dónde* una red aprende los patrones del modelo de Ising 2D cuando cruza su transición de fase (T_c = 2/ln(1+√2) ≈ 2.269185, K_c ≈ 0.440687).

Fase actual: **MLP clasificador, L = 10 (N = 100), h = 0**, etiqueta `y = 1[T > T_c]`. Después: SimpleCNN, ViT, generativo (U-Net → autorregresivo), escalado en L.

## 2. Fuentes (qué se leyó y qué no)

| Fuente | Estado |
|---|---|
| README de `JD314/Ising-ML` | Leído |
| Código de `ising_fisher/`, `scripts/`, `tests/` | **No leído** → tarea MLP-00 |
| Presentación *Ising ML* (UdeA) | Leída |
| Charla *Fisher Information Geometry … 2026* | Leída |
| `CONTEXT_ISING_ML.md` del repo (especificación y auditoría) | **No leído** → MLP-00 |

## 3. Lo que ya se sabe (de la presentación)

- Datos: Metropolis + simulated annealing; T ∈ [0.1, 4.0], ΔT = 0.1 (40 temperaturas); L ∈ {10, 20, 30}.
- Etiqueta: 0 si T < T_c (ordenado), 1 si T > T_c (desordenado).
- MLP: flatten → capas densas + ReLU, sin estructura espacial. Entrenado 7 épocas, CPU, PyTorch 2.6.0, Python 3.12.
- Resultado L = 10: accuracy ≈ 1 lejos de T_c y **cae a ≈ 0.64 en T_c**; la pérdida de test de L = 10 es ruidosa (pico ≈ 0.13 en época 6) → posible subentrenamiento.
- Observación a verificar: en las figuras las configuraciones de T bajo aparecen **todas ↑** y la magnetización promedio ≈ 1.0 a T bajo. Si es magnetización con signo, los datos no son Z₂-simétricos (el generador arranca ordenado hacia arriba).

## 4. Convenciones (obligatorias)

- `J = 1`, `k_B = 1`, `H = −J Σ s_i s_j − h Σ s_i`, spins ±1, **condiciones periódicas** (verificar en `mc.py`; si no, ver riesgo R1).
- `E ≡ E_J = −Σ s_i s_j` (a h = 0). `M = Σ s_i`. `N = L²`.
- Chart principal: **T**. Si se usa β o K, se declara y no se mezclan picos.
- `F` = total por configuración; `f = F/N`. Todo resultado dice cuál.
- Grilla de T: la de la presentación (0.1–4.0) **más** refinamiento 2.0–2.6 con ΔT = 0.02 (ver MLP-02).
- Semillas de red: 0–4. Semillas MC independientes de las de red.
- Splits agrupados por cadena MC; nunca por muestra.
- Dtype: `float64` en cualquier cálculo de Fisher / Gram / regresión; `float32` permitido solo en entrenamiento.

## 5. Decisiones tomadas

| ID | Decisión | Motivo |
|---|---|---|
| D1 | El oráculo se calcula con **matriz de transferencia exacta** (1024×1024 para L = 10), no con fórmulas cerradas | Evita errores de signo en Z₄; se verifica con enumeración exacta de L ≤ 4 del repo |
| D2 | Información mutua en capas con **sondas variacionales** (cotas inferiores) y no binning | Redes deterministas continuas: el binning mide la discretización |
| D3 | Eficiencia por neurona = ρ²(a_j, E \| T) | Identidad exacta de Cauchy–Schwarz con el score `u_T ∝ E − ⟨E⟩` |
| D4 | Fisher de parámetros solo con etiquetas del modelo | Evita Fisher empírica (Sun 2026) |
| D5 | El oráculo bayesiano usa solo E (suficiente a h = 0) | Test duro de si la red "factoriza" por E |

## 6. Riesgos conocidos

| ID | Riesgo | Mitigación |
|---|---|---|
| R1 | Generador con fronteras abiertas o sin equilibrio → el oráculo no aplica | MLP-02: comparar histograma de E contra exacto; si hay discrepancia, regenerar con `mc.py` periódico |
| R2 | Datos con signo fijo (todo ↑) → la red puede usar signo de M | Aleatorizar signo global por configuración (exacto por Z₂); entrenar también la variante sin aleatorizar como control |
| R3 | T_c no es nítida en L = 10 | Usar T_peak(L) del máximo de F_TT exacto además de T_c |
| R4 | `nθ ≫ ns` en Gram/regresiones | Held-out, ridge, nulos, escaneo en ns |
| R5 | Etiqueta por umbral construye un pico en T_c | Control con umbrales 1.5 y 3.0 y regresor de T |
| R6 | Autocorrelación MC cerca de T_c | Medir τ_int(E); espaciar muestras ≥ 2τ_int; varias cadenas |

## 7. Glosario mínimo

- **Score** `u_T = ∂_T ln q`. A h = 0: `(E − ⟨E⟩)/T²`.
- **Eficiencia termométrica** `e_g(T) = (∂_T E_T[g])² / (Var_T(g) F_TT) = ρ²(g, E | T) ∈ [0,1]`.
- **r_f(T)**: R² held-out de regresar el score físico sobre ∇θ f(x) (espacio de funciones del clasificador).
- **Oráculo bayesiano**: `P*(y=1|x)` calculado con `ln p(x|T)` exacto.
- **DPI**: desigualdad de procesamiento de datos; `F_{g(x)} ≤ F_x`, `I(T;g(x)) ≤ I(T;x)`.
- **BBP / MP**: transición de outlier de Baik–Ben Arous–Péché; borde de Marchenko–Pastur del espectro de ruido.
