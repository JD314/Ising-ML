---
name: ising-exact-physics-oracle
description: Use for task MLP-01. Calcula con matriz de transferencia exacta ln Z, F_TT, T_peak, el posterior de T, el clasificador bayesiano óptimo y la información mutua I(T;X), I(Y;X) para el toro Ising 10x10 a h=0.
---

# Skill 01 · Oráculo exacto L = 10

Teoría: `THEORY.md` T1, T2. Métricas: A1, A5, A6.

## Pasos
1. **Matriz de transferencia.** Para cada β construye `V` (1024×1024, estados de una fila con BC periódicas, ver T2). Usa `float64`. `ln Z = ln Σ_k λ_k^L` con `λ` de `eigvalsh(V)`, factorizando el máximo autovalor para evitar desbordes.
2. **Derivadas.** `⟨E⟩ = −∂_β ln Z`, `Var E = ∂²_β ln Z` por diferencias centrales (paso 1e-4, comprobar convergencia con 1e-3 y 1e-5) o autodiff. De ahí `C`, `F_TT = Var(E)/T⁴`, `f_TT = F_TT/100` y `T_peak` (máximo de `F_TT`; reporta también el de `C`).
3. **Validación cruzada.** Aplica el mismo código a L = 2, 3, 4 y compara con la enumeración exacta de `observables.py` (error relativo < 1e-10 en `ln Z`, `⟨E⟩`, `Var E`).
4. **Posterior y oráculo.** Con grilla `{T_k}` (la de MLP-02) y prior uniforme: `P*(T_k|x) ∝ exp(−E/T_k − ln Z_k)` calculado con `logsumexp`; `P*(y=1|x) = Σ_{T_k>T_c} P*(T_k|x)`. Implementa como función de E (vectorizada), `oracle_posterior(E)`.
5. **Información.** Con muestras `x ~ p(·|T_k)` (de MLP-02, o provisionales vía MC) estima `I(T;X) = ln K − mean(H(T|x))` y `I(Y;X) = H(Y) − mean(H(Y|x))`, con IC bootstrap por cadena.
6. **Entropía de Gibbs.** `S(T)` por integración de `C/T` desde T = 10–50 (con `S ≈ N ln 2` ajustado por expansión de alta T como comprobación).

## Entregables
- `ising_fisher/exact_L10.py` (+ `tests/test_exact_L10.py`).
- `experiments/mlp_L10/MLP-01/metrics.json`: `lnZ(T)`, `E_mean(T)`, `F_TT(T)`, `C(T)`, `T_peak`, `I(T;X)`, `I(Y;X)`, `S(T)`.
- Figura: `F_TT(T)` con `T_c` y `T_peak` marcadas.

## Aceptación
- Errores de validación en L ≤ 4 < 1e-10. `F_TT > 0`. `I(T;X) ≤ ln K`. `P*` suma 1.
- Actualiza `STATE.md`: `T_peak`, `I(T;X)`, `I(Y;X)`.

## Trampas
- Si MLP-00 halla BC abiertas, **detente**: pide decisión (regenerar datos con BC periódicas o construir el oráculo para BC abiertas).
- No uses fórmulas cerradas de Kaufman sin verificar contra la enumeración.
