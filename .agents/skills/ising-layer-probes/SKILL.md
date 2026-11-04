---
name: ising-layer-probes
description: Use for task MLP-06. Mide por capa y por neurona la fracción de Fisher térmica (lineal y no lineal), la brecha de accesibilidad, la eficiencia termométrica rho^2(a_j,E|T), la correlación parcial con |M| y la decodabilidad de observables físicos en el MLP de Ising 10x10.
---

# Skill 05 · Sondas por capa (D1–D8)

Teoría: **T3, T4, T5**. Requiere MLP-04 y MLP-05.

## Preparación
- Carga checkpoint final de las 5 semillas y los datos de MLP-02. Activaciones `h_0 = s` (entrada), `h_1`, `h_2` (post-ReLU), `f` (logit).
- **Split de sondas independiente:** las sondas se entrenan con cadenas del split *train* y se evalúan en *test*. Documenta el ID de cadenas.
- Trabaja **a T fijo**: un ajuste por cada T de la grilla (ns ≥ 2000 en train, ≥ 500 en test).
- Variable objetivo: `E` (equivale al score `u_T` salvo constante y escala), `|M|`, `M²`, energía local, tamaño del clúster mayor.

## Cálculos
1. **D8 (sanidad):** ridge de `E` sobre `s` (lineal) → R² ≈ 0 en todas las T. Si no es ≈ 0 hay fuga, datos no simétricos o error.
2. **D1:** `φ_l^lin(T)`: ridge con α por validación cruzada interna (agrupada por cadena); R² held-out.
3. **D2:** `φ_l^nonlin(T)`: sonda MLP (1 capa oculta de 64, Adam, parada temprana) y, como contraste, medias condicionales por bins sobre las 3 primeras componentes principales de `h_l`. Reporta el mejor valor held-out.
4. **D3:** `Δ_l = φ^nonlin − φ^lin` con IC bootstrap por cadena.
5. **D7 (DPI):** test emparejado de que `φ^nonlin` no crece con la profundidad (diferencia por semilla y T; IC). Una violación > IC se investiga como error de estimación.
6. **D4:** `e_j(T) = ρ²(a_j, E | T)` por neurona; construye el mapa neurona × T (ordena neuronas por T de máximo).
7. **D5:** correlaciones parciales `ρ²(a_j,E | |M|, T)` y `ρ²(a_j,|M| | E, T)`. Calcula la fracción de neuronas "atajo" (`ρ²_|M| > ρ²_E`) por capa y T.
8. **D6:** decodabilidad (lineal y no lineal) de los demás observables por capa.

## Controles dentro de la skill
- Repite todo con la **red inicializada** (época 0) y con la **red de etiquetas barajadas**.
- Sonda con etiquetas de E **permutadas entre cadenas** (control de selectividad): debe dar R² ≈ 0.

## Entregables
`metrics.json` por métrica, figuras: `φ_l^lin(T)`, `φ_l^nonlin(T)` y `Δ_l(T)` (una línea por capa, bandas IC); mapa de calor de `e_j(T)`; barras de neuronas atajo vs. neuronas energía.

## Aceptación
`φ_input^lin < 0.02`; DPI respetada o violaciones explicadas; controles (inicialización, permutación) < los valores entrenados cerca de T_c con IC no solapados.

## Lectura de resultados
- `φ^lin` sube entre capas ⇒ esa transición **hace accesible** la energía.
- `Δ` grande y `φ^nonlin` alto ⇒ la información está pero no es linealmente legible.
- `φ_logit(T)` pequeño lejos del umbral es esperable (T5, punto 5).
