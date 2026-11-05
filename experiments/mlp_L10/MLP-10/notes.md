# MLP-10: Dinámica de Entrenamiento y Emergencia por Capa

**Fecha:** 2026-11-05  
**Script:** `experiments/mlp_L10/MLP-10/evaluate_dynamics.py`  
**Modelos:** Checkpoints de `SimpleMLP` a lo largo del entrenamiento en 5 semillas (0 a 4).  
**Checkpoints Evaluados:** 15 estados por semilla (pasos intra-época 1, 5, 20, 50, 200 de época 1, y épocas 0, 1, 2, 3, 5, 8, 13, 21, 34, 55).

---

## 1. Tiempos de Emergencia $t_l$ y Orden de Aprendizaje (H1, H2)

El tiempo de emergencia $t_l$ se define operativamente como el primer checkpoint donde la fracción de Fisher lineal $\phi_l^{\mathrm{lin}}(T_c)$ supera el 50% de su valor convergido final.

| Capa / Observable | Tiempo Medio $t_l$ (épocas) | Desv. Est. | Votos por Semilla |
|---|---|---|---|
| **Capa Oculta $h_1$ ($t_1$)** | **0.0800** | 0.1029 | [0.0286, 0.0286, 0.2857, 0.0286, 0.0286] |
| **Logit de Salida $f$ ($t_{\mathrm{logit}}$)** | **0.2314** | 0.3846 | [0.0286, 0.0286, 1.0, 0.0714, 0.0286] |
| **Pesos Capa 1 $r_f^{(1)}$ ($t_{W_1}$)** | 0.0000 | 0.0000 | [0.0, 0.0, 0.0, 0.0, 0.0] |
| **Pesos Capa 2 $r_f^{(2)}$ ($t_{W_2}$)** | 0.0800 | 0.1029 | [0.0286, 0.0286, 0.2857, 0.0286, 0.0286] |

- **Orden de aprendizaje medido:** `simultaneous / joint emergence of hidden layer h1 and logit within intra-epoch resolution`.
- **Consistencia entre semillas:** La condición se cumple en 5/5 semillas (`n_seeds_with_t_l_min_over_layers = 5`).
- **Fase Rápida del Aprendizaje:** La mayor parte del ascenso de $\phi_l$ y la caída abrupta de la pérdida de validación ocurren durante las primeras épocas ($t \in [0, 3]$). Ya en los pasos intra-época ($t \approx 0.07$ a $0.28$ épocas, pasos 50 a 200), la capa $h_1$ empieza a estructurar las características térmicas.

---

## 2. Evolución del Espacio de Fisher en Parámetros (H3)

A $n_s = 1000$ fijo en $T_c$:
- **Traza de Fisher $\operatorname{Tr}(F_\theta)$:**
  - Inicialización (época 0): $\operatorname{Tr}(F_\theta) \approx 5.09$.
  - Final: $\operatorname{Tr}(F_\theta) \approx 64.84$.
  - El crecimiento de la traza refleja la nitidez de la superficie de decisión cerca de $T_c$.
- **Rango Efectivo $R_{\mathrm{eff}}$:**
  - Evoluciona desde $\approx 27.6$ en inicialización hacia $\approx 22.8$ al converger, mostrando una compresión de dimensión intrínseca conforme la red especializa sus pesos.
- **Outliers sobre el Borde de Marchenko-Pastur:**
  - En época 0: 28.0 outliers.
  - Al converger: 22.8 outliers estables por encima de $\lambda_+$.

---

## 3. Dinámica de Pesos y Régimen de Aprendizaje (H4, H5)

- **Desplazamiento Relativo de Parámetros $\|\theta_t - \theta_0\| / \|\theta_0\|$:**
  - Capa 1 ($W_1, b_1$): alcanza un desplazamiento final de 385.17%.
  - Capa 2 ($W_2, b_2$): alcanza un desplazamiento final de 656.10%.
- **Alineamiento Relativo del Kernel (H5):** El producto interno normalizado del kernel tangente $\operatorname{Tr}(K(t) K(0)) / (\|K(t)\| \|K(0)\|)$ varía significativamente a lo largo de las primeras épocas (1.00 $\to$ 0.86), confirmando que la red opera en un **régimen de aprendizaje de representaciones (feature learning)** y no en el régimen perezoso (lazy/NTK).

---

## 4. Figuras Entregadas

1. [`experiments/mlp_L10/MLP-10/figs/phi_evolution_vs_epoch.png`](figs/phi_evolution_vs_epoch.png): Evolución de $\phi_{h_1}^{\mathrm{lin}}$ y $\phi_{\mathrm{logit}}^{\mathrm{lin}}$ con la curva de pérdida y la fase rápida sombreada.
2. [`experiments/mlp_L10/MLP-10/figs/learning_order_comparison.png`](figs/learning_order_comparison.png): Comparación de tiempos de emergencia $t_l$ por capa en cada semilla.
3. [`experiments/mlp_L10/MLP-10/figs/fisher_trace_and_rank_vs_epoch.png`](figs/fisher_trace_and_rank_vs_epoch.png): Evolución temporal de la traza de Fisher, rango efectivo y outliers MP.
4. [`experiments/mlp_L10/MLP-10/figs/weight_norm_change_vs_epoch.png`](figs/weight_norm_change_vs_epoch.png): Distancia relativa a la inicialización $\|\Delta \theta\| / \|\theta_0\|$.
5. [`experiments/mlp_L10/MLP-10/figs/heatmaps_layer_vs_epoch.png`](figs/heatmaps_layer_vs_epoch.png): Mapas 2D capa $\times$ época para $\phi^{\mathrm{lin}}$, $r_f^{(l)}$, brecha $\Delta_l$ y alineamiento del kernel.
