# MLP-03: Modelos de Referencia (Baselines) vs Oráculo Bayesiano Óptimo ($L=10$)

**Fecha:** 2026-11-04  
**Script:** `experiments/mlp_L10/MLP-03/evaluate_baselines.py`  
**Configuración:** `experiments/mlp_L10/MLP-03/config.json`  
**Métricas:** `experiments/mlp_L10/MLP-03/metrics.json`  
**Partición de datos:** Idéntico split de MLP-02 (14 cadenas train, 3 cadenas val, 3 cadenas test = 38,400 muestras de prueba).

---

## 1. Resumen Comparativo Global (Conjunto de Test)

| Modelo | Accuracy Global | Entropía Cruzada Global (CE) [nats] | $\Delta \text{CE}$ vs Oráculo [nats] | Exceso CE (%) |
|---|---|---|---|---|
| **1. Oráculo Bayesiano Óptimo $P^*(y=1 \mid E)$** | **88.12%** | **0.268343** | — | — |
| **2. Logística sobre $E$** | **87.94%** | **0.274138** | $+0.005795$ | **+2.16%** |
| **3. Logística sobre $\|M\|$** | **83.90%** | **0.372589** | $+0.104246$ | $+38.85\%$ |
| **4. Logística sobre $(E, \|M\|)$** | **88.08%** | **0.270271** | $+0.001928$ | $+0.72\%$ |
| **5. Lineal sobre espines $s$** | **57.41%** | **0.692697** | $+0.424354$ | $+158.14\%$ |
| **6. Frozen MLP (Random Features, 100 $\to$ 16 $\to$ 1)** | **75.34%** | **0.528812** | $+0.260469$ | $+97.07\%$ |

---

## 2. Hallazgos Físicos y Metodológicos Clave

1. **Suficiencia de $E$ y Curvatura del Logit:**
   - La regresión logística sobre la energía $E$ alcanza una accuracy global de **87.94%**, casi idéntica a la del oráculo exacto (**88.12%**).
   - Su exceso de entropía cruzada es de **+2.16%** ($+0.0058$ nats), un valor mínimo dictado por la curvatura intrínseca del logit exacto $g^*(E) = \ln \frac{P^*(y=1|E)}{1 - P^*(y=1|E)}$ en volumen finito $L=10$ (la sigmoide mononivel aproxima estrechamente pero no idénticamente la función racional de sumas de exponenciales del oráculo).
   - Al incorporar $|M|$ en la logística conjunta $(E, |M|)$, el exceso de entropía cruzada disminuye a **+0.72%** frente al oráculo.

2. **Falla del Clasificador Lineal en $s$ (Simetría $\mathbb{Z}_2$):**
   - El clasificador lineal sobre los espines individuales $s_i$ obtiene una entropía cruzada de **0.6927 nats** ($\approx \ln 2 = 0.6931$, nivel de azar puro) y accuracy cercana al azar cerca de $T_c$.
   - Esto confirma rigurosamente el teorema de simetría $\mathbb{Z}_2$ de `THEORY.md` (T3): **ninguna combinación lineal $\sum_i w_i s_i$ contiene información térmica sobre $T$ a $h=0$**, forzando a cualquier red a aprender invariantes cuadráticos o pares.

3. **Incapacidad de Features Aleatorias sin Entrenar:**
   - El MLP congelado (*Frozen MLP*) con pesos aleatorios no entrenados alcanza sólo **75.34%** de accuracy y pierde más del 97% de entropía cruzada frente al oráculo, demostrando que la representación requiere optimización guiada por gradientes.

---

## 3. Figuras Guardadas

- [`experiments/mlp_L10/MLP-03/figs/baselines_accuracy_vs_T.png`](file:///home/jd/Projects/Ising-ML/experiments/mlp_L10/MLP-03/figs/baselines_accuracy_vs_T.png): Curvas de $\text{Accuracy}(T)$ con bandas de confianza bootstrap al 95% para los 6 modelos, señalando $T_c$ y $T_{\text{peak}}$.
- [`experiments/mlp_L10/MLP-03/figs/delta_ce_vs_oracle.png`](file:///home/jd/Projects/Ising-ML/experiments/mlp_L10/MLP-03/figs/delta_ce_vs_oracle.png): Brecha $\Delta \text{CE}(T)$ de cada baseline respecto al oráculo bayesiano exacto.
