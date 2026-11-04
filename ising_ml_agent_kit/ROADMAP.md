# Plan de ruta de experimentos · MLP, L = 10

Cada experimento tiene: **objetivo**, **método** (resumen; el detalle está en la skill), **entregables**, **criterio de aceptación** (go/no-go) y **hipótesis falsable**. Las hipótesis son predicciones; si fallan, se registra `DONE_NEGATIVE` y se interpreta.

## Resumen de dependencias

| Fase | IDs | Pregunta |
|---|---|---|
| 0 Base | MLP-00, 01, 02 | ¿Tenemos ground truth exacto y datos válidos? |
| 1 Referencias | MLP-03 | ¿Qué es "tan bueno como se puede" y qué logran baselines simples? |
| 2 Red | MLP-04, 05 | ¿Qué aprende la red y cuánto le falta al óptimo? |
| 3 Dónde | MLP-06, 07, 08, 09 | ¿En qué capa, neurona y entrada vive la información? |
| 4 Cuándo | MLP-10 | ¿En qué momento del entrenamiento emerge? |
| 5 Rigor | MLP-11 | ¿Sobrevive a los controles? |
| 6 Decisión | SYN | ¿Qué sigue (CNN, escalado en L, generativo)? |

---

## MLP-00 · Auditoría del repo y entorno
**Objetivo.** Conocer lo que existe antes de construir. **Método.** Leer `CONTEXT_ISING_ML.md`, `mc.py`, `dataset.py`, `fisher.py`, `train.py`, `models/classifier.py`; correr los 44 tests; extraer arquitectura, BC, esquema de recocido, formato HDF5.
**Entregables.** `experiments/mlp_L10/MLP-00/notes.md` con tabla de hallazgos; rellenar "Parámetros fijados" en `STATE.md`.
**Aceptación.** 44/44 tests pasan o fallos documentados; BC identificadas; arquitectura del `SimpleMLP` anotada; confirmación de si las muestras tienen signo fijo.
**Hipótesis.** H0-a: el generador usa BC periódicas. H0-b: `mc.py` no aleatoriza el signo global.

## MLP-01 · Oráculo exacto L = 10
**Objetivo.** Ground truth físico y bayesiano. **Método.** Skill 01: matriz de transferencia, `ln Z`, `F_TT`, `T_peak`, posterior de T, clasificador bayesiano, `I(T;X)`, `I(Y;X)`.
**Entregables.** Módulo `ising_fisher/exact_L10.py`, tests y `metrics.json` con `F_TT(T)`, `C(T)`, `T_peak`, `I(T;X)`, `I(Y;X)`.
**Aceptación.** Coincide con enumeración exacta L = 4 (error relativo < 1e-10); `Σ_E p(E|T) = 1`; `F_TT>0`; `T_peak` reportada.
**Hipótesis.** `T_peak(10)` difiere de `T_c` en ≈ 0.1 o menos (verificar signo).

## MLP-02 · Datos y auditoría contra lo exacto
**Objetivo.** Datos válidos para el oráculo. **Método.** Skill 02: generar con `mc.py` en BC periódicas, grilla de la presentación + refinamiento 2.0–2.6 (Δ=0.02), ≥ 20 cadenas × ≥ 200 muestras espaciadas por T, aleatorizar signo global; comparar E con exacto.
**Entregables.** HDF5 `data/L10_*.h5`, informe de KS/χ² por T, τ_int(T).
**Aceptación.** KS no rechaza sistemáticamente (≤ 10 % de las T con p < 0.01 tras corregir por múltiples tests); signo balanceado 0.5 ± 0.02; split por cadena 70/15/15.
**Hipótesis.** La generación original sin aleatorizar signo da datos desbalanceados; el recocido deja poco sesgo en E pero sí en M.

## MLP-03 · Baselines de referencia
**Objetivo.** Techos y pisos. **Método.** Skill 04 (modo baselines): (i) oráculo bayesiano; (ii) logística sobre `E`; (iii) sobre `|M|`; (iv) sobre `(E,|M|)`; (v) lineal sobre `s`; (vi) MLP con capa oculta congelada (features aleatorias).
**Entregables.** Tabla accuracy(T), CE(T), brecha al oráculo.
**Aceptación.** (ii) ≈ oráculo (≤ 1 % de CE adicional); (v) ≈ azar cerca de T_c.
**Hipótesis.** H3-a: logística sobre `E` alcanza el oráculo. H3-b: `|M|` solo es peor cerca de T_c. H3-c: el lineal sobre `s` falla cerca de T_c por Z₂.

## MLP-04 · Entrenamiento del MLP con checkpoints
**Objetivo.** Red convergida y trayectoria guardada. **Método.** Skill 03: 5 semillas, hasta convergencia de la pérdida de validación (paciencia 10), checkpoints en el calendario de H, optimizador y tasa fijos, split por cadena.
**Entregables.** `ckpt/seed{s}/epoch{e}.pt`, curvas de pérdida.
**Aceptación.** Pérdida de validación estable (variación < 1 % en las últimas 5 épocas) en ≥ 4 de 5 semillas; accuracy global ≥ lo reportado en la presentación.
**Hipótesis.** Con más épocas que las 7 de la presentación la pérdida de test de L = 10 deja de oscilar.

## MLP-05 · Comportamiento frente al oráculo
**Objetivo.** Cuánto le falta a la red. **Método.** Skill 04: B1–B7.
**Entregables.** Curvas `accuracy(T)`, `ΔCE(T)`, `H(p)(T)`, calibración; R² del logit vs `g*(E)`.
**Aceptación.** B5 calculado con IC; B6 < umbral numérico documentado.
**Hipótesis.** H5-a: la caída de accuracy en T_c de la red es **igual** a la del oráculo (límite intrínseco por solapamiento de `p(x|T<T_c)` y `p(x|T>T_c)`), no un fallo de la red. H5-b: el logit explica ≥ 90 % de su varianza vía E.

## MLP-06 · Sondas por capa (¿dónde se vuelve legible la energía?)
**Objetivo.** D1–D8. **Método.** Skill 05: `φ^lin`, `φ^nonlin`, brecha `Δ`, eficiencia por neurona, correlación parcial con |M|, decodabilidad de observables.
**Entregables.** Curvas `φ_l(T)` para `l ∈ {input, h1, h2, logit}`; mapa neurona × T de `e_j`.
**Aceptación.** `φ_input^lin ≈ 0` (|valor| < 0.02); DPI respetada en `φ^nonlin` dentro de IC.
**Hipótesis.** H6-a: `φ_input^lin = 0`. H6-b: la brecha `Δ` es grande en h1 y pequeña en h2 (la capa 1 fabrica los rasgos pares, la 2 los hace legibles). H6-c: un subconjunto pequeño de neuronas concentra `e_j` cerca de T_c. H6-d: parte de `e_j` se explica por |M| (atajo).

## MLP-07 · Información y entropía por capa
**Objetivo.** E1–E10. **Método.** Skill 06: sondas variacionales de `I(T;h_l)`, `I(Y;h_l)`, entropías gaussianas, rango efectivo, correlación total, información por neurona.
**Entregables.** Curvas de información por capa y fracción de `I(T;X)` retenida.
**Aceptación.** `I(T;h_l) ≤ I(T;X)` y no creciente en l (IC); sin uso de binning de alta dimensión.
**Hipótesis.** H7-a: `I(Y;ŷ)` se acerca a `I(Y;X)` (la red es casi eficiente). H7-b: `I(T;ŷ) ≪ I(T;h_2)` (el logit descarta casi toda la información térmica, conserva la de la etiqueta).

## MLP-08 · Fisher en parámetros, por capa
**Objetivo.** F1–F8. **Método.** Skill 07: `r_f(T)` global y por capa, Gram dual, rango efectivo, outliers sobre MP con escaneo en ns, nulos (inicialización, pesos barajados, features aleatorias).
**Entregables.** `r_f^{(l)}(T)`, espectros, tabla de outliers vs ns.
**Aceptación.** Tests (a)–(d) pasan en el MLP; `r_f(init)` y nulos significativamente por debajo de `r_f(entrenado)` cerca de T_c.
**Hipótesis.** H8-a: `r_f(T)` tiene máximo cerca de T_c (como `SimpleCNN` en L = 8: 0.99). H8-b: el MLP necesita rango efectivo mayor que el CNN. H8-c: el espectro del primer bloque está dominado por `⟨ssᵀ⟩` (confusión de entrada).

## MLP-09 · Lado de la entrada
**Objetivo.** G1–G6. **Método.** Skill 08: Fourier de pesos de capa 1, sustituto cuadrático de f, saliencia, ablaciones.
**Entregables.** Mapas por sitio, espectros k, matriz `Q` por distancia.
**Aceptación.** R² del sustituto cuadrático reportado con IC; `Q` comparada con `J_ij`.
**Hipótesis.** H9-a: `Q_ij` se concentra en vecinos próximos (aprende el hamiltoniano) o en el modo `k=0` (aprende |M|); el reparto cuantifica el atajo. H9-b: la invarianza traslacional se aprende solo parcialmente (el MLP no la impone).

## MLP-10 · Dinámica del aprendizaje
**Objetivo.** H1–H5. **Método.** Skill 09: recalcular D, E, F, G en cada checkpoint.
**Entregables.** Tiempos de emergencia por capa, orden de aprendizaje, trayectorias de rango y outliers.
**Aceptación.** Curvas con IC por semilla; `t_l` definido para ≥ 4 de 5 semillas.
**Hipótesis.** H10-a: el rasgo par se vuelve linealmente legible primero en la capa más profunda y la capa 1 se ajusta después (aprendizaje de arriba hacia abajo) o viceversa; el orden se mide, no se asume. H10-b: los outliers del Gram aparecen antes que la caída de la pérdida.

## MLP-11 · Controles y nulos
**Objetivo.** Descartar explicaciones triviales. **Método.** Skill 10: umbral desplazado (1.5 y 3.0), regresor de T (sin umbral), etiquetas barajadas, datos con signo fijo vs. aleatorizado, red sin entrenar, ancho/profundidad (64, 128, 256; 1–3 capas), augmentación Z₂.
**Entregables.** Cada métrica P0 repetida bajo cada control con IC.
**Aceptación.** Para cada hallazgo central, se declara si sobrevive, se desplaza o desaparece.
**Hipótesis.** H11-a: con umbral 1.5, el máximo de incertidumbre de salida está en 1.5, pero `F_TT` y `φ^nonlin` de capas intermedias siguen mostrando la física en T_c. H11-b: sin aleatorizar signo, aparece dependencia de M impar que desaparece con augmentación.

## SYN · Síntesis y decisión
**Objetivo.** Responder "dónde y cómo aprende la red". **Método.** Skill 11: consolidar tablas, figuras principales, lista de afirmaciones con nivel de evidencia, registro de anomalías.
**Decisión go/no-go hacia la siguiente fase:**
- **Go a CNN (misma batería):** si el oráculo y las métricas D/E/F son estables entre semillas.
- **Go a escalado en L ∈ {8, 12, 16, 24, 32}:** si el hallazgo central sobrevive a MLP-11.
- **No-go (revisar teoría):** si el oráculo no coincide con los datos o la DPI falla.

## Fases posteriores (fuera de este kit, por referencia)
CNN (`SimpleCNN` circular), ViT, generativo (U-Net → autorregresivo con equivarianza Z₂; objetivos enmascarados con eficiencia `e_h ∼ L^{-7/4}`), universalidad (Potts, 3D), estático vs dinámico (E5 del plan).
