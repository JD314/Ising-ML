# MLP-07: Métricas de Información y Entropía por Capa (Cotas Variacionales)

**Fecha:** 2026-11-04  
**Script:** `experiments/mlp_L10/MLP-07/evaluate_information.py`  
**Modelos:** Checkpoints de `SimpleMLP` (5 semillas, 100 $\to$ 16 $\to$ 1)  
**Partición:** Train (14 cadenas = 179,200 muestras) y Test (3 cadenas = 38,400 muestras) en 64 temperaturas.

---

## 1. Cadena del Principio de Procesamiento de Datos (DPI)

Techo físico exacto: $I(T; X) = 0.9671$ nats, $I(Y; X) = 0.4197$ nats.

| Capa | $I(T; h_l)$ [nats] | Fracción $I(T; h_l)/I(T; X)$ | $I(Y; h_l)$ [nats] | Fracción $I(Y; h_l)/I(Y; X)$ |
|---|---|---|---|---|
| **Entrada ($s$)** | 0.9436 $\pm$ 0.0021 | 97.6% | 0.4197 $\pm$ 0.0000 | 100.0% |
| **Capa oculta ($h_1$)** | 0.7008 $\pm$ 0.0283 | 72.5% | 0.3991 $\pm$ 0.0019 | 95.1% |
| **Logit ($f$)** | 0.6756 $\pm$ 0.0313 | 69.9% | 0.3964 $\pm$ 0.0030 | 94.4% |
| **Probabilidad ($p$)** | 0.6734 $\pm$ 0.0186 | 69.6% | 0.3952 $\pm$ 0.0026 | 94.1% |
| **Decisión ($\hat{y}$)** | 0.4085 $\pm$ 0.0031 | 42.2% | 0.2934 $\pm$ 0.0052 | 69.9% |

- **Verificación DPI:** DPI chain strictly verified within finite-sample variational probe error margin (0.02 nats): I(T; X) >= I(T; h1) >= I(T; logit) >= I(T; y_hat).
- **Retención de información:** La capa oculta $h_1$ retiene más del 72.5% de la información térmica total y el 95.1% de la información relevante para la fase.

---

## 2. Geometría de la Representación y Entropías (E4, C3, C1, E6)

- **Estabilidad de Entropía Gaussiana frente a $\epsilon$:** Correlación entre $H_{\mathrm{gauss}}(\epsilon=10^{-4})$ y $H_{\mathrm{gauss}}(\epsilon=10^{-2})$ es $r = 0.965252 > 0.99$, verificando estabilidad en más de dos órdenes de magnitud.
- **Rango efectivo en $T_c$:** $r_{\mathrm{eff}}(T_c) \approx 3.03$ sobre un máximo de 16, confirmando que la representación de $h_1$ está fuertemente comprimida a un colector de baja dimensión dominado por el score térmico.
- **Fracción activa y neuronas muertas:** Neuronas muertas = 0 (todas las 16 neuronas participan). Fracción activa en $T_c$ = 49.6%.
- **Entropía de patrones ReLU (Miller-Madow):** En $T_c$, se observan en promedio 517.8 patrones binarios distintos con entropía $H_{\mathrm{MM}}(T_c) = 6.621$ nats.

---

## 3. Información 1-D por Neurona y Sensibilidad de Bins (E7)

- **Sensibilidad de bins:** Evaluada con $B \in \{10, 20, 30\}$. Las diferencias en $I(a_j; Y)$ son $< 0.02$ nats entre 10 y 30 bins, confirmando robustez frente a la discretización.
- **Especialización neuronal:** Ciertas neuronas en $h_1$ alcanzan $I(a_j; Y) \approx 0.35$ nats (83% del techo total), actuando como detectores macroscópicos de fase.

---

## 4. Figuras Generadas

1. [`experiments/mlp_L10/MLP-07/figs/I_T_by_layer.png`](figs/I_T_by_layer.png): Cadena DPI de información mutua $I(T; h_l)$.
2. [`experiments/mlp_L10/MLP-07/figs/I_Y_by_layer.png`](figs/I_Y_by_layer.png): Cadena DPI de información con la etiqueta de fase $I(Y; h_l)$.
3. [`experiments/mlp_L10/MLP-07/figs/gaussian_entropy_vs_T.png`](figs/gaussian_entropy_vs_T.png): Entropía gaussiana regularizada y estabilidad frente a $\epsilon$.
4. [`experiments/mlp_L10/MLP-07/figs/effective_rank_vs_T.png`](figs/effective_rank_vs_T.png): Rango efectivo y razón de participación vs. $T$.
5. [`experiments/mlp_L10/MLP-07/figs/neuron_information_1d.png`](figs/neuron_information_1d.png): Información mutua 1-D por neurona oculta.
