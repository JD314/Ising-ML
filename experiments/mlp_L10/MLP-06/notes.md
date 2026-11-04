# MLP-06: Sondas por Capa (Accesibilidad de Fisher, Eficiencia por Neurona y Nulos)

**Fecha:** 2026-11-04  
**Script:** `experiments/mlp_L10/MLP-06/evaluate_probes.py`  
**Modelos:** Checkpoints finales de `SimpleMLP` (5 semillas, 100 $\to$ 16 $\to$ 1)  
**Partición:** Train (14 cadenas = 2,800 muestras/T) y Test (3 cadenas = 600 muestras/T) en 64 temperaturas.

---

## 1. Sanidad de Paridad $\mathbb{Z}_2$ en la Entrada (D8)

$$\phi_{\mathrm{input}}^{\mathrm{lin}} \approx 0.0 \quad (\max |R^2| = 0.000000 < 0.02)$$

- **Aceptación:** Cumplida estrictamente. Dado el balance exacto de paridad $\mathbb{Z}_2$, ninguna regresión lineal sobre los espines $s$ puede predecir la energía escalar $E$. No hay fuga ni rotura de simetría espuria.

---

## 2. Accesibilidad de Fisher y Principio de Procesamiento de Datos (D1, D2, D3, D7)

| Capa | $\phi^{\mathrm{lin}}(T_c)$ | $\phi^{\mathrm{nonlin}}(T_c)$ | $\Delta(T_c) = \phi^{\mathrm{nonlin}} - \phi^{\mathrm{lin}}$ | Interpretación Física |
|---|---|---|---|---|
| **Entrada ($s$)** | 0.0000 | 1.0000 | 1.0000 | Información presente pero **completamente inaccesible linealmente** ($\Delta = 1.0$). |
| **Capa oculta ($h_1$)** | 0.8342 | 0.8342 | 0.0000 | **Colapso de la brecha**: la primera capa hace linealmente legible la energía ($R^2 \approx 83.4\%$). |
| **Salida (Logit $f$)** | 0.8130 | 0.8130 | 0.0000 | Canal comprimido para la decisión binaria; máxima correlación en $T_c$. |

- **DPI:** DPI strictly respected across layers within finite-sample probe estimation noise; phi_input = 1.0 >= phi_h1 >= phi_logit across all T.
- **Veredicto:** El aprendizaje de representaciones en el MLP consiste precisamente en la transformación de configuraciones no lineales en un subespacio linealmente alineado con el score térmico $u_T \propto E$.

---

## 3. Eficiencia Termométrica por Neurona y Neuronas Atajo (D4, D5)

- **Eficiencia acotada:** $e_j(T) \le 1.0$ verificado en las 16 neuronas y todas las temperaturas (desigualdad de Cauchy-Schwarz satisfecha).
- **Fracción de neuronas atajo en $h_1$:** 29.2%. La gran mayoría de neuronas en $h_1$ (70.8%) codifican primariamente la energía $E$ sobre la magnetización $|M|$.

---

## 4. Controles y Nulos

1. **Control de etiquetas permutadas entre cadenas:** $\max |R^2| = 0.007384 < 0.05$ (aprobado).
2. **Separación frente a red no entrenada (época 0):** En $T_c$, la red entrenada ($\phi_{h_1}^{\mathrm{lin}} \approx 0.834$) supera netamente a la red aleatoria inicial ($\phi_{h_1}^{\mathrm{lin}} \approx 0.301$) con intervalos de confianza al 95% completamente no solapados.

---

## 5. Figuras Entregadas

1. [`experiments/mlp_L10/MLP-06/figs/phi_lin_vs_T.png`](figs/phi_lin_vs_T.png): Fracción de Fisher lineal $\phi^{\mathrm{lin}}_l(T)$ por capa.
2. [`experiments/mlp_L10/MLP-06/figs/phi_nonlin_vs_T.png`](figs/phi_nonlin_vs_T.png): Fracción de Fisher no lineal $\phi^{\mathrm{nonlin}}_l(T)$ por capa.
3. [`experiments/mlp_L10/MLP-06/figs/delta_accessibility_vs_T.png`](figs/delta_accessibility_vs_T.png): Brecha de accesibilidad $\Delta_l(T)$ mostrando su colapso en $h_1$.
4. [`experiments/mlp_L10/MLP-06/figs/neuron_efficiency_heatmap.png`](figs/neuron_efficiency_heatmap.png): Mapa de eficiencia termométrica $e_j(T)$ de las 16 neuronas.
5. [`experiments/mlp_L10/MLP-06/figs/shortcut_vs_energy_neurons.png`](figs/shortcut_vs_energy_neurons.png): Reparto de neuronas de energía vs. atajo.
