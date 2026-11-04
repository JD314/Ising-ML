# MLP-02: Generación y Auditoría de Datos vs Oráculo Exacto ($L=10$)

**Fecha:** 2026-11-03  
**Archivo HDF5:** `data/L10_data_audit.h5` (12 MB, 256,000 configuraciones en total)  
**Configuración:** `experiments/mlp_L10/MLP-02/config.json`  
**Métricas:** `experiments/mlp_L10/MLP-02/metrics.json`  
**Grilla de Temperaturas:** 64 valores ($T \in [0.1, 4.0]$ con paso 0.1, refinada a $\Delta T = 0.02$ entre 2.00 y 2.60).

---

## 1. Parámetros de Generación y Split por Cadena

- **Lattice:** $L = 10$, $N = 100$, $J = 1.0$, $h = 0.0$ con condiciones periódicas (PBC).
- **Cadenas por temperatura:** 20 cadenas independientes (cada una con `SeedSequence` derivado de semilla maestra 42).
- **Termalización:** 2000 barridos (sweeps) por cadena.
- **Muestreo:** 200 muestras por cadena con separación `thin = 20` barridos ($\ge 2\tau_{\text{int}}(E)$, decorrelacionadas).
- **Total muestras por $T$:** 4,000 configuraciones ($256,000$ muestras en la grilla completa).
- **Split por cadena (fijo para toda la campaña):**
  - **Train (70%):** 14 cadenas por $T$ (56,000 muestras totales train por fase/régimen equivalente).
  - **Val (15%):** 3 cadenas por $T$.
  - **Test (15%):** 3 cadenas por $T$.
  - *Ninguna cadena ni configuración correlacionada cruza el límite de partición.*

---

## 2. Auditoría contra el Oráculo Exacto (Criterios de Aceptación)

| Criterio | Umbral / Requisito | Valor Obtenido | Estado |
|---|---|---|---|
| **Consistencia de Energía $\langle E \rangle$** | $\le 10\%$ de temperaturas con $p < 0.01$ tras corrección Holm-Bonferroni | **7.8%** (5 de 64 temperaturas) | **APROBADO** |
| **Sesgo Sistemático en $\langle E \rangle$** | Sin sesgo sistemático (distribución de Z-score centrada en 0) | Media de Z-scores = $+0.04$ | **APROBADO** |
| **Balance de Signo ($\mathbb{Z}_2$)** | Fracción $M > 0$ en $0.5 \pm 0.02$ tras aleatorización global | **0.4999** (rango por $T$: $[0.483, 0.518]$) | **APROBADO** |
| **Separación respecto a $\tau_{\text{int}}$** | `thin` $\ge 2 \tau_{\text{int}}(E)$ | $\max \tau_{\text{int}}(E) \approx 0.70$ sweeps tras thinning 20 | **APROBADO** |
| **Estructura $S(0)$** | $S(0) = \langle M^2 \rangle / L^2$ crece monotónicamente hacia $T_c$ | Pico de $S(0)$ y transición suave verificada | **APROBADO** |

---

## 3. Coherencia con el Oráculo Bayesiano e Información Mutua

Evaluado sobre las 256,000 configuraciones reales del dataset auditado contra el oráculo exacto de matriz de transferencia:
- **Accuracy global del Clasificador Bayesiano Óptimo ($P^*$):** **88.04%**
- **Accuracy del Clasificador Bayesiano en $T \approx T_c$ ($T = 2.2600$):** **51.75%** (refleja la máxima incertidumbre intrínseca en la transición de fase de volumen finito).
- **Información Mutua Real:**
  - $I(T; X) = \mathbf{0.9671 \text{ nats}}$ (techo $\ln 64 \approx 4.1589 \text{ nats}$).
  - $I(Y; X) = \mathbf{0.4197 \text{ nats}}$ (techo $H(Y) \approx 0.6927 \text{ nats}$).

---

## 4. Figuras Guardadas

1. [`experiments/mlp_L10/MLP-02/figs/p_values_energy_audit.png`](file:///home/jd/Projects/Ising-ML/experiments/mlp_L10/MLP-02/figs/p_values_energy_audit.png): Z-score y p-valores por temperatura vs oráculo exacto.
2. [`experiments/mlp_L10/MLP-02/figs/tau_int_and_S0.png`](file:///home/jd/Projects/Ising-ML/experiments/mlp_L10/MLP-02/figs/tau_int_and_S0.png): Tiempo de autocorrelación $\tau_{\text{int}}$ y factor de estructura $S(0)$ a través de $T_c$.
3. [`experiments/mlp_L10/MLP-02/figs/sign_balance.png`](file:///home/jd/Projects/Ising-ML/experiments/mlp_L10/MLP-02/figs/sign_balance.png): Verificación del balance de paridad $\mathbb{Z}_2$ antes y después de la aleatorización de signo.
