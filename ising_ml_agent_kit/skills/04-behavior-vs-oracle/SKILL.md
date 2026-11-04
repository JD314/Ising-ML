---
name: ising-behavior-vs-oracle
description: Use for tasks MLP-03 (baselines) and MLP-05 (comportamiento). Compara accuracy, entropía cruzada, entropía predictiva y calibración por temperatura de la red y de los baselines frente al clasificador bayesiano óptimo exacto.
---

# Skill 04 · Comportamiento frente al oráculo y baselines

Teoría: T2, T3. Métricas: B1–B7. Requiere MLP-01/02 (y MLP-04 para MLP-05).

## Modo baselines (MLP-03)
Entrena/evalúa, con el mismo split y la misma métrica por T:
1. **Oráculo bayesiano** `P*(y=1|E)`.
2. Logística sobre `E`. 3. Logística sobre `|M|`. 4. Logística sobre `(E, |M|)`.
5. Lineal sobre `s` (esperado ≈ azar cerca de T_c por Z₂).
6. MLP con la capa oculta **congelada** (features aleatorias) y solo la salida entrenada.
Reporta `accuracy(T)`, `CE(T)`, `H(p)(T)` y `ΔCE(T)` vs. oráculo.

## Modo red (MLP-05)
1. B1, B2: curvas por T para las 5 semillas, media e IC.
2. **B3/E10:** entropía predictiva media de la red vs. entropía del posterior bayesiano `H(Y|x)` por T. La red debería quedar cerca; sobreconfianza (`H` menor) indica memorización o extrapolación.
3. **B5:** `g*(E)` = logit del oráculo. R² de `f` frente a `g*(E)` (global y a T fijo); fracción de varianza de `f` explicada por E con regresión no paramétrica (bins de E / splines), por `(E,|M|)`, y por la parte de |M| ortogonal a E.
4. **B6:** simetría `‖f(x) − f(−x)‖` por T.
5. **B7:** `∂_T E_T[p]` por diferencias finitas centradas sobre la grilla fina cerca de T_c; eficiencia `e_p(T) = ρ²(p, E | T)`.
6. **B4:** ECE por T y diagrama de fiabilidad.
7. Figura clave: accuracy(T) de la red, del oráculo y de la logística sobre E, con T_c y T_peak marcadas.

## Aceptación
- Todas las curvas con IC (cadena y semilla). Baselines (ii) ≈ oráculo (CE extra ≤ 1 %).
- Se declara explícitamente si la caída de accuracy en T_c es **intrínseca** (igual al oráculo) o **atribuible a la red** (brecha de CE > IC).

## Interpretación
La caída de accuracy en T_c no implica fallo: cerca de T_c las distribuciones de ambas clases se solapan. Solo la brecha frente al oráculo mide lo que la red pierde.
