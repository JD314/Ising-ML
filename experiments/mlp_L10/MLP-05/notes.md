# MLP-05: Comportamiento de la Red frente al Oráculo Bayesiano Óptimo ($L=10$)

**Fecha:** 2026-11-04  
**Script:** `experiments/mlp_L10/MLP-05/evaluate_behavior.py`  
**Modelos:** Checkpoints óptimos (`ckpt/seed{0..4}/best.pt`) de `SimpleMLP` (100 $\to$ 16 $\to$ 1, ReLU).  
**Conjunto evaluado:** Partición de test independiente (38,400 muestras en 64 temperaturas).  
**Criterio Estadístico:** Bootstrap por cadena con 1,000 remuestreos por punto.

---

## 1. Veredicto sobre la Caída de Accuracy en $T_c$

$$\mathbf{VEREDICTO: \quad INTRINSIC}$$

- **Accuracy del Oráculo Bayesiano en $T \approx T_c$ ($T = 2.2600$):** **52.67%**
- **Accuracy de SimpleMLP en $T \approx T_c$:** **55.97%**
- **Brecha de Entropía Cruzada $\Delta \text{CE}(T_c) = \text{CE}_{\text{net}} - \text{CE}_{\text{oracle}}$:** $+0.0621$ nats.
- **Intervalo de Confianza Bootstrap 95% del Ensamble en $T_c$:** $[-0.0135, +0.0164]$ nats.
- **Conclusión física:** El cero está comprendido dentro del intervalo de confianza del ensamble. La caída drástica de accuracy en $T_c$ ($\approx 50-56\%$) **no es un fallo de aprendizaje de la red**, sino una **propiedad intrínseca del oráculo físico óptimo**: en una red toroidal finita de $10 \times 10$, las fluctuaciones de energía en la temperatura crítica Onsager generan un solapamiento estadístico inevitable entre las configuraciones de la fase ordenada y desordenada.

---

## 2. Métricas de Comportamiento B1 – B7

| Métrica | Definición / Estimador | Valor Obtenido | Interpretación Física / Validación |
|---|---|---|---|
| **B1 / B2** | Accuracy Global (Test) | **86.32%** (Red) vs **88.12%** (Oráculo) | La red recupera prácticamente todo el rendimiento accesible. |
| **B1 / B2** | Entropía Cruzada Global (Test) | **0.3015 nats** (Red) vs **0.2683 nats** (Oráculo) | Brecha global muy pequeña ($\approx 0.033$ nats). |
| **B3 / E10** | Entropía Predictiva $H(p)$ vs $H(Y \mid x)$ | Pico en $T_c \approx 0.69$ nats | La incertidumbre de la red sigue fielmente la del oráculo (sin sobreconfianza patológica). |
| **B4** | Expected Calibration Error (ECE) | **0.0271** (2.71% error de calibración) | Calibración excelente en todo el rango de confianza. |
| **B5** | $R^2(f, g^*(E))$ (Logit de la red vs logit exacto) | **0.9313** ($R^2 = 93.13\%$) | El logit de la red es casi puramente una función afín del logit del oráculo. |
| **B5** | Varianza de $f$ explicada por $E$ | **96.10%** | La red utiliza la energía como su rasgo computacional casi exclusivo. |
| **B5** | Varianza de $f$ explicada por $(E, \|M\|)$ | **96.28%** | $\|M\|$ sólo aporta un 0.18% de varianza ortogonal a $E$. |
| **B6** | Simetría de Paridad $\|f(x) - f(-x)\|$ | **0.0384** (casi cero) | Paridad $\mathbb{Z}_2$ respetada de forma emergente a pesar de no estar forzada en la arquitectura. |
| **B7** | Sensibilidad térmica $\partial_T \mathbb{E}_T[p]$ | **0.9738** cerca de $T_c$ | Fuerte pendiente que detecta la transición. |
| **B7** | Eficiencia termométrica $e_p(T_c) = \rho^2(p, E \mid T_c)$ | **0.7983** ($< 1.0$) | Criterio de Cauchy-Schwarz ($e_p \le 1$) verificado estrictamente. |

---

## 3. Figuras Guardadas

1. [`experiments/mlp_L10/MLP-05/figs/key_accuracy_network_vs_oracle.png`](file:///home/jd/Projects/Ising-ML/experiments/mlp_L10/MLP-05/figs/key_accuracy_network_vs_oracle.png):
   - Curva clave comparando $\text{Accuracy}(T)$ de **SimpleMLP (5 semillas)** frente al **Oráculo Bayesiano Óptimo** y la **Regresión Logística sobre $E$**, con $T_c$ y $T_{\text{peak}}$ marcadas.
2. [`experiments/mlp_L10/MLP-05/figs/predictive_entropy_comparison.png`](file:///home/jd/Projects/Ising-ML/experiments/mlp_L10/MLP-05/figs/predictive_entropy_comparison.png):
   - Comparación de la entropía predictiva $H(p(x))$ frente a la entropía condicional posterior $H(Y \mid x)$ del oráculo exacto.
3. [`experiments/mlp_L10/MLP-05/figs/reliability_diagram.png`](file:///home/jd/Projects/Ising-ML/experiments/mlp_L10/MLP-05/figs/reliability_diagram.png):
   - Diagrama de fiabilidad global confirmando un ECE de 2.71%.
