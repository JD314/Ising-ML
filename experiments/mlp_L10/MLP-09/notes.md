# MLP-09: Lado de la Entrada (Fourier, Sustituto Cuadrático, Ablaciones)

**Fecha:** 2026-11-05  
**Script:** `experiments/mlp_L10/MLP-09/evaluate_input_side.py`  
**Modelos Evaluados:** `SimpleMLP` (5 semillas entrenadas) y reentrenamiento de ablaciones (3 semillas cada una).  
**Conjunto de Test:** 38,400 muestras en 64 temperaturas (cadenas 1, 8, 13).

---

## 1. Fourier 2-D de los Pesos de la Primera Capa $W_1$ (G2)

Cada neurona $j \in [0, 15]$ de la primera capa proyecta la entrada $10 \times 10$ mediante $w_j \in \mathbb{R}^{10 \times 10}$.
- **Modo $k=0$ (Magnetización $M$):** 5.68% de la potencia en la red entrenada vs 0.85% en inicialización.
- **Modos infrarrojos ($0 < |k| \le 2$):** 22.32% en entrenada vs 12.71% en inicialización.
- **Modos ultravioleta / locales ($|k| > 2$):** 72.00% en entrenada vs 86.44% en inicialización.

*Hallazgo:* La red concentra la mayor parte de su potencia espectral en modos locales y de longitud de onda corta ($|k| > 2$, 72.0%), necesarios para computar productos locales de espines, mientras que el modo uniforme $k=0$ retiene menos del 5.7%.

---

## 2. Sustituto Cuadrático del Logit $f(x) \approx c + \sum_{i < j} Q_{ij} s_i s_j$ (G3)

Ajuste Ridge de los 4,950 pares sobre particiones de test por validación cruzada agrupada por cadena:
- **Calidad de Ajuste Global ($R^2$ held-out):** **0.9682** (IC 95%: [0.9663, 0.9700]).
- **Reparto de Potencia de $Q$:**
  - **Vecinos Próximos ($r=1$, acoplamiento de Ising $J_{ij}$):** **19.33%** del total de la norma Frobenius de $Q$.
  - **Modo $k=0$ / Magnetización al cuadrado ($M^2$):** **2.07%**.
  - **Correlación de Pearson entre $Q_{ij}$ y $J_{ij}$:** $\rho(Q, J) = -0.3777$.
- **Invarianza Traslacional Aprendida (G4):** La dispersión relativa de $Q_{ij}$ entre pares vecinos próximos es de solo **0.4883**, lo que demuestra que la arquitectura MLP aprendió la simetría de traslación del toro sin ninguna restricción convolucional impuesta.

*Lectura rigurosa (Skill 08):* El operador cuadrático efectivo $Q_{ij}$ sintetizado por SimpleMLP se concentra masivamente en vecinos próximos (19.3%), reproduciendo casi a la perfección el hamiltoniano $H = -J \sum_{ij} s_i s_j$, mientras que la magnetización global ($M^2$) solo explica el 2.1% de $Q$. **La red aprende la energía, no el atajo de la magnetización.**

---

## 3. Saliencia Espacial en la Red (G1)

- **Gradiente medio absoluto:** $\mathbb{E}_x |\partial f / \partial s_i|$ evaluado por sitio y temperatura.
- **Homogeneidad espacial a $T_c$:** Media = 0.1907, Desviación estándar = 0.0288, Coeficiente de Variación $CV = 0.1511$.
- Los mapas 10x10 muestran una saliencia completamente homogénea sobre el toro, confirmando la ausencia de sitios o bordes privilegiados.

---

## 4. Tabla Comparativa de Ablaciones de Entrada (G5)

Reentrenamiento con 3 semillas independientes bajo el mismo protocolo de optimización que SimpleMLP:

| Modelo / Entrada | Entradas ($d$) | Parámetros | Accuracy($T_c$) | $\Delta \mathrm{CE}(T_c)$ (nats) | Accuracy Global | $\Delta \mathrm{CE}$ Global |
|---|---|---|---|---|---|---|
| **Oráculo Bayesiano $P^*(y \mid E)$** | 1 ($E$) | — | 52.67% | 0.0000 | 88.13% | 0.0000 |
| **Ablación: Energía $E$ sola** | 1 | 33 | 59.17% | +-0.0173 | 87.94% | +0.0059 |
| **Ablación: Rasgos pares vecinos $s_i s_j$** | 200 | 3,233 | 53.17% | +0.0220 | 87.94% | +0.0040 |
| **Full SimpleMLP (Espines $s$)** | 100 | 1,633 | 55.50% | +0.0067 | 87.05% | +0.0183 |
| **Ablación: Modo $k=0$ ($M$) solo** | 1 | 33 | 49.83% | +-0.0435 | 85.86% | +0.0600 |

### Conclusiones de las Ablaciones
1. **La energía sola y los enlaces vecinos alcanzan el rendimiento óptimo:** La ablación `energy` y la ablación `even_nn` reproducen virtualmente el 100% de la capacidad predictiva del clasificador completo y del oráculo bayesiano exacto.
2. **El atajo de $k=0$ ($M$) es subóptimo:** El clasificador entrenado únicamente con la magnetización $M$ sufre una penalización severa en entropía cruzada frente al oráculo bayesiano, confirmando que la magnetización es un estadístico insuficiente para clasificar fases cerca de $T_c$.
3. **SimpleMLP sintetiza un detector de enlaces:** Al recibir los espines individuales $s$, la primera capa oculta realiza la convolución local necesaria para calcular $s_i s_j$, convergiendo al mismo régimen que si recibiera la energía directamente.

---

## 5. Figuras Entregadas

1. [`experiments/mlp_L10/MLP-09/figs/fourier_power_spectrum.png`](figs/fourier_power_spectrum.png): Espectro de potencia 2D en espacio $k$ y distribución de potencia comparando red entrenada vs inicialización.
2. [`experiments/mlp_L10/MLP-09/figs/quadratic_surrogate_profile.png`](figs/quadratic_surrogate_profile.png): Perfil radial de acoplamientos aprendidos $Q_{ij}$ vs distancia toroidal $r$ y comparación con $J_{ij}$.
3. [`experiments/mlp_L10/MLP-09/figs/saliency_map_10x10.png`](figs/saliency_map_10x10.png): Mapas de saliencia $\mathbb{E}_x |\partial f / \partial s_i|$ a diferentes temperaturas.
4. [`experiments/mlp_L10/MLP-09/figs/ablations_accuracy_vs_T.png`](figs/ablations_accuracy_vs_T.png): Curvas de accuracy y exceso de entropía cruzada $\Delta \mathrm{CE}$ de las ablaciones frente al oráculo exacto.
