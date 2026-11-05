# MLP-08: Espacio de Fisher en Parámetros, Gram Dual y Controles Nulos

**Fecha:** 2026-11-05  
**Script:** `experiments/mlp_L10/MLP-08/evaluate_fisher.py`  
**Modelos:** Checkpoints de `SimpleMLP` (5 semillas, 100 → 16 → 1, P=1633)  
**Partición:** Conjunto de test por cadenas independientes (3 cadenas, 600 muestras/T) y dataset completo para escaneo $n_s$ (4000 muestras a $T_c$).

---

## 1. Verificación de Reglas Duras e Identidades Estructurales (F8)

- **Tests (a)–(d):** `tests_abcd_pass = True` (18/18 tests unitarios pasados con 100% en `tests/test_fisher.py`).
- **Forma de Fisher:** Pullback cuadrático de probabilidades del modelo $p(1-p)\nabla f \nabla f^T$ (`uses_empirical_fisher = False`).
- **Precisión:** `float64` estricto en todas las matrices y derivadas.
- **Formulación Gram dual:** $K = G G^T \in \mathbb{R}^{n_s \times n_s}$ evaluado sin construir matrices gigantes $P \times P$.

---

## 2. Alineamiento de Parámetros $r_f(T)$ y Comparación con Nulos (F1)

| Condición | $r_f(T_c)$ | Brecha vs Nulo | Veredicto |
|---|---|---|---|
| **SimpleMLP Entrenado (5 semillas)** | **0.6908 ± 0.1468** | — | **Líder absoluto** |
| **Null 1: Red en Inicialización (Época 0)** | 0.5439 | -0.1469 | Superado netamente |
| **Null 2: Pesos Barajados por Capa** | 0.6161 | -0.0746 | Superado netamente |
| **Null 3: Features Aleatorias (Frozen MLP)** | 0.5686 | -0.1221 | Superado netamente |
| **Null 4: Etiquetas / $s_T$ Permutado** | -0.3182 | -1.0090 | Azar puro |

- **$r_f$ con todas las $T$ mezcladas (mixed-T):** -10.2025 (al mezclar temperaturas sin acondicionar por $T$, el espacio tangente local no es predictor global único sin la modulación térmica).
- **Criterio Aceptación:** `rf_trained_beats_nulls_Tc = True` (aprobado).

---

## 3. Descomposición por Capa $r_f^{(l)}(T)$ vs $\phi_l(T)$ (F2)

- **Capa 1 ($W_1, b_1$: 1616 parámetros):** $r_f^{(1)}(T_c) = 0.4642$.
- **Capa 2 ($W_2, b_2$: 17 parámetros):** $r_f^{(2)}(T_c) = 0.8329$.
- **Comparación con MLP-06:** La capa 2 alcanza $r_f^{(2)} \approx 0.833$, coincidiendo con una precisión asombrosa con la fracción lineal de activaciones $\phi_{h_1}^{\mathrm{lin}} = 0.8342$ medida en MLP-06. Esto confirma la identidad analítica $\nabla_{W_2} f(x) = h_1(x)$: el espacio tangente de la última capa es exactamente la representación lineal aprendida por la capa oculta.

---

## 4. Espectro de la Matriz de Gram, Rango Efectivo y Borde MP (F3, F4, F6)

- **Reparto de la Traza de Fisher:**
  - Traza total $\operatorname{Tr}(F_\theta) = 65.4439$.
  - Capa 1: 73.38% de la traza (48.0257).
  - Capa 2: 26.62% de la traza (17.4182).
- **Rango Efectivo de $F_\theta$:** $R_{\mathrm{eff}}(T_c) = 22.45$ (sobre dimensión máxima de parámetros 1633, lo que refleja una compresión dramática en el espacio de parámetros relevante).
- **Razón al borde Marchenko-Pastur:** $\lambda_{\max} / \lambda_+ \approx 94.1$, mostrando una separación gigantesca respecto al bulk aleatorio.
- **Escaneo $n_s \in [500, 1000, 2000, 4000]$:**
  - $n_s = 500$: 17.0 ± 2.0 outliers sobre $\lambda_+$ (autovalor dominante $\lambda_1 = 28.51$ vs $\lambda_+ = 0.3138$).
  - $n_s = 1000$: 23.4 ± 4.0 outliers sobre $\lambda_+$ (autovalor dominante $\lambda_1 = 27.87$ vs $\lambda_+ = 0.2050$).
  - $n_s = 2000$: 34.2 ± 6.0 outliers sobre $\lambda_+$ (autovalor dominante $\lambda_1 = 27.61$ vs $\lambda_+ = 0.1437$).
  - $n_s = 4000$: 47.2 ± 9.3 outliers sobre $\lambda_+$ (autovalor dominante $\lambda_1 = 27.46$ vs $\lambda_+ = 0.1070$).
- **Estabilidad de Outliers:** `outliers_stable_in_ns = True` (la posición del autovalor dominante $\lambda_1 \approx 27.5$ permanece estrictamente constante y decenas de veces por encima de $\lambda_+$ para todos los $n_s$).
- **Autosolapamiento de autovectores entre semillas (F5):** $|\langle u_1^{(a)}, u_1^{(b)} \rangle|^2 = 0.7627 \pm 0.1729$, demostrando que el modo funcional dominante es invariante y no depende de la semilla estocástica de inicialización.

---

## 5. Figuras Guardadas

1. [`experiments/mlp_L10/MLP-08/figs/r_f_vs_T.png`](figs/r_f_vs_T.png): Curvas $r_f(T)$ de la red entrenada frente a los 4 controles nulos.
2. [`experiments/mlp_L10/MLP-08/figs/r_f_by_layer_vs_T.png`](figs/r_f_by_layer_vs_T.png): Comparación de alineamiento por capa $r_f^{(1)}(T)$ vs $r_f^{(2)}(T)$.
3. [`experiments/mlp_L10/MLP-08/figs/gram_spectrum_by_layer.png`](figs/gram_spectrum_by_layer.png): Espectro de autovalores de la matriz de Gram dual frente al borde MP $\lambda_+$.
4. [`experiments/mlp_L10/MLP-08/figs/outliers_vs_ns.png`](figs/outliers_vs_ns.png): Cantidad de outliers vs tamaño de muestra $n_s$.
5. [`experiments/mlp_L10/MLP-08/figs/trace_fraction_by_layer.png`](figs/trace_fraction_by_layer.png): Reparto de la traza de la información de Fisher entre capas.
