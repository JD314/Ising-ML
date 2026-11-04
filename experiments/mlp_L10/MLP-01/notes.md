# MLP-01: Oráculo Físico Exacto L = 10 (Matriz de Transferencia)

**Fecha:** 2026-11-03  
**Módulo:** `ising_fisher/exact_L10.py`  
**Tests:** `tests/test_exact_L10.py` (20 tests passed)

---

## 1. Implementación de la Matriz de Transferencia

Se implementó la matriz de transferencia fila a fila para el toro periódico $L \times L$ ($2^L \times 2^L$, 1024x1024 en $L=10$) en `float64`:
$$V[s, s'] = \exp\left(\beta J \left[\frac{1}{2}\sum_i s_i s_{i+1} + \frac{1}{2}\sum_i s'_i s'_{i+1} + \sum_i s_i s'_i\right]\right)$$
$$\ln Z = \ln \operatorname{Tr}(V^L) = L \ln \lambda_{\max} + \ln \sum_k \left(\frac{\lambda_k}{\lambda_{\max}}\right)^L$$

Derivadas termodinámicas calculadas por diferencias centrales estables:
- $\langle E \rangle = -\partial_\beta \ln Z$
- $\operatorname{Var}(E) = \partial_\beta^2 \ln Z$
- $C(T) = \operatorname{Var}(E) / T^2$
- $F_{TT}(T) = \operatorname{Var}(E) / T^4$

Invarianza ante variación del paso verificada: $|\Delta F_{TT}| / F_{TT} < 10^{-4}$ entre $h=10^{-3}, 10^{-4}, 10^{-5}$.

---

## 2. Validación Cruzada Obligatoria ($L=2, 3, 4$)

Comparado contra la enumeración exacta exhaustiva de microestados de `ising_fisher/observables.py`:
- Error relativo en $\ln Z$: $< 2 \times 10^{-16}$ (precisión de máquina, criterio $< 10^{-10}$ satisfecho holgadamente).
- Error relativo en $\langle E \rangle$: $< 3 \times 10^{-8}$ (criterio satisfecho).
- Error relativo en $\operatorname{Var}(E)$: $< 1 \times 10^{-7}$ (criterio satisfecho).

---

## 3. Picos de Fisher y Calor Específico ($L=10$)

- $T_c$ Onsager (límite termodinámico): $T_c \approx 2.269185$
- $T_{\text{peak}}(F_{TT})$: **2.27900** con $F_{TT}^{\max} \approx 24.4844$ (por sitio $f_{TT}^{\max} \approx 0.2448$)
- $T_{\text{peak}}(C)$: **2.34500** con $C^{\max} \approx 130.9054$

---

## 4. Clasificador Bayesiano Óptimo e Información Mutua

- Implementada la función vectorizada `oracle_posterior(E, grid_T, ln_Z_grid)` usando `logsumexp`.
- Clasificador bayesiano óptimo: $P^*(y=1 \mid x) = \sum_{T_k > T_c} P^*(T_k \mid x)$.
- Información mutua (estimación provisional con muestras de `mc.py` previo a MLP-02):
  - $I(T; X) \approx 0.9574 \text{ nats}$ (techo: $\ln 64 \approx 4.1589 \text{ nats}$)
  - $I(Y; X) \approx 0.4002 \text{ nats}$ (techo: $H(Y) \approx 0.6927 \text{ nats}$)
