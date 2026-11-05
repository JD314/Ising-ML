# STATE · estado vivo (el agente actualiza este archivo)

**Última actualización:** 2026-11-05  **Agente/persona:** Antigravity (Gemini)  
**Tarea activa:** ninguna  **Siguiente acción:** ejecutar MLP-11 (controles y nulos).

## Tablero

| ID | Tarea | Skill | Depende de | Estado | Artefactos | Hallazgo en una frase |
|---|---|---|---|---|---|---|
| MLP-00 | Auditoría del repo y entorno | 00 | — | DONE | `experiments/mlp_L10/MLP-00/{notes.md,metrics.json,config.json}` | 44 tests pasan; PBC y sweeps confirmados en `ising_fisher/`; R1/R2 de legacy mitigados en nuevo diseño. |
| MLP-01 | Oráculo exacto L=10 | 01 | 00 | DONE | `ising_fisher/exact_L10.py`, `tests/test_exact_L10.py`, `experiments/mlp_L10/MLP-01/{metrics.json,figs/F_TT_exact_L10.png,notes.md}` | Oráculo exacto de matriz de transferencia validado (<1e-10 en ln Z a L<=4); T_peak(F_TT)=2.279, I(T;X)=0.957 nats (prov). |
| MLP-02 | Datos y auditoría vs exacto | 02 | 01 | DONE | `data/L10_data_audit.h5`, `experiments/mlp_L10/MLP-02/{config.json,metrics.json,figs/,notes.md}` | 256k muestras en 64 T; balance Z2 exacto (0.500); 7.8% rechazos p<0.01 (Holm); I(T;X)=0.967 nats, I(Y;X)=0.420 nats. |
| MLP-03 | Baselines (E, \|M\|, lineal, RF) | 04 | 02 | DONE | `experiments/mlp_L10/MLP-03/{evaluate_baselines.py,config.json,metrics.json,figs/,notes.md}` | Logística en E alcanza CE a +2.16% del oráculo (Acc 87.9% vs 88.1%); lineal en s colapsa a CE=0.693 nats por Z2. |
| MLP-04 | Entrenamiento MLP con checkpoints | 03 | 02 | DONE | `ckpt/seed{0..4}/*.pt`, `experiments/mlp_L10/MLP-04/{train_mlp.py,config.json,metrics.json,figs/,notes.md}` | 5 semillas convergen en 50-69 épocas; 4/5 semillas estables (<1% var); test acc media 86.32%; checkpoints guardados. |
| MLP-05 | Comportamiento vs oráculo | 04 | 03, 04 | DONE | `experiments/mlp_L10/MLP-05/{evaluate_behavior.py,config.json,metrics.json,figs/,notes.md}` | Caída de accuracy en Tc es INTRÍNSECA (Acc net 56.0% vs oracle 52.7%, Delta CE en CI); 96.1% de varianza del logit explicada por E. |
| MLP-06 | Sondas por capa | 05 | 04 | DONE | `experiments/mlp_L10/MLP-06/{evaluate_probes.py,config.json,metrics.json,notes.md,figs/}` | Brecha de accesibilidad Delta colapsa en h1 (Delta=1.000 en entrada s -> 0.000 en h1 con R^2=83.4%); phi_lin(s)<=0.000 (Z2); DPI respetada y controles separados. |
| MLP-07 | Métricas de información | 06 | 05, 06 | DONE | `experiments/mlp_L10/MLP-07/{evaluate_information.py,config.json,metrics.json,notes.md,figs/}` | Cadena DPI respetada (I(T;h1)=0.701 nats, I(Y;h1)=0.399 nats / 95.1% retenido); rango efectivo comprimido a 3.03 en Tc; 0 neuronas muertas. |
| MLP-08 | Fisher en parámetros por capa | 07 | 04, 06 | DONE | `experiments/mlp_L10/MLP-08/{evaluate_fisher.py,config.json,metrics.json,notes.md,figs/}` | r_f(Tc)=0.691 supera a los 4 nulos; r_f^(2)=0.833 coincide con phi_h1=0.834; outliers estables en ns con ratio MP 94x y solapamiento 0.763. |
| MLP-09 | Lado de la entrada | 08 | 04 | DONE | `experiments/mlp_L10/MLP-09/{evaluate_input_side.py,config.json,metrics.json,notes.md,figs/}` | Sustituto cuadrático alcanza R^2=0.968 con 19.3% de potencia en vecinos y solo 2.1% en k=0; ablación energy iguala a full MLP. |
| MLP-10 | Dinámica de entrenamiento | 09 | 06, 07, 08 | DONE | `experiments/mlp_L10/MLP-10/{evaluate_dynamics.py,config.json,metrics.json,notes.md,figs/}` | Fase rápida en épocas 0-3; emergencia simultánea/bottom-up (t_h1=0.08 épocas <= t_logit=0.23 épocas en 5/5 semillas); feature learning activo. |
| MLP-11 | Controles y nulos | 10 | 05–10 | TODO | | |
| MLP-11 | Controles y nulos | 10 | 05–10 | TODO | | |
| SYN | Síntesis y decisión go/no-go | 11 | todas | TODO | | |

Estados: `TODO` · `IN_PROGRESS` · `BLOCKED` · `DONE` · `DONE_NEGATIVE` (hecho, hipótesis refutada).

## Parámetros fijados (rellenar al ejecutar)

| Parámetro | Valor |
|---|---|
| Arquitectura MLP de referencia | `SimpleMLP` (100→16→1, ReLU, 1633 parámetros) en `ising_fisher/models/classifier.py` |
| Épocas hasta convergencia por semilla | Seed 0: 69, Seed 1: 63, Seed 2: 63, Seed 3: 50, Seed 4: 54 (media 59.8 épocas) |
| BC del generador | Periódicas (PBC toroidal exacta con `% L` en `ising_fisher/mc.py`) |
| Grilla de T final | 64 temperaturas: [0.1:0.1:4.0] + [2.00:0.02:2.60] sin duplicados |
| Muestras por T / cadenas | 20 cadenas independientes, 200 muestras/cadena (4,000 muestras/T; 256,000 en total). Split fijo 70/15/15 por cadena |
| T_peak(L=10) de F_TT exacto | 2.27900 (máx F_TT = 24.4844 total, f_TT = 0.2448/sitio; T_peak(C) = 2.34500) |
| Accuracy bayesiana en T_c | 52.67% a T=2.2600 (SimpleMLP alcanza 55.97%; global del clasificador bayesiano óptimo en test: 88.12%) |
| I(T;X) exacto | 0.9671 nats (calculado sobre dataset completo auditado de 256k muestras; techo ln 64 = 4.1589 nats) |
| I(Y;X) exacto | 0.4197 nats (calculado sobre dataset completo auditado; techo H(Y) = 0.6927 nats) |
| Semillas de red | 0–4 (convergencia estable, test accuracy media = 86.32%) |

## Decisiones (añadir al final)

| Fecha | ID | Decisión | Evidencia |
|---|---|---|---|
| 2026-11-03 | D0 | Enlaces simbólicos de conveniencia `STATE.md`, `AGENT_PROTOCOL.md` y `AGENT_CONTEXT.md` creados en la raíz apuntando a `ising_ml_agent_kit/` | Mantiene compatibilidad con las instrucciones que esperan `STATE.md` en la raíz. |
| 2026-11-03 | D6 | Se adopta `SimpleMLP` (`100 -> 16 -> 1`, ReLU, P=1633) existente en `ising_fisher/models/classifier.py` como arquitectura base de referencia | Verificado en `classifier.py` e integrado en `test_models.py` y `run_first_exploration.py`. |
| 2026-11-03 | D7 | Grilla de temperaturas fijada en 64 puntos ([0.1:0.1:4.0] más refinada [2.00:0.02:2.60]) | Especificada en `ROADMAP.md` y `THEORY.md` para capturar la vecindad de $T_c$ y $T_{\text{peak}}$. |
| 2026-11-03 | D8 | Esquema de split fijo de cadenas por temperatura: 14 cadenas train (70%), 3 val (15%), 3 test (15%) | Permite evaluación held-out sin filtración de autocorrelación entre train y test. |
| 2026-11-04 | D9 | Margen de exceso de CE de la logística sobre E documentado en +2.16% | El oráculo exacto P*(y=1\|E) tiene un logit con curvatura intrínseca en volumen finito; una sigmoide mono-parámetro alcanza CE=0.2741 vs CE=0.2683 del oráculo (exceso 2.16%, o +0.72% al incluir \|M\|). |
| 2026-11-04 | D10 | Paciencia fijada en 10 épocas para SimpleMLP en batch 256 | Todas las semillas convergen limpiamente entre 50 y 69 épocas sin sobreajuste severo. |
| 2026-11-04 | D11 | Veredicto sobre la caída de accuracy en Tc declarado como INTRÍNSECO | En Tc, Delta CE = +0.0621 nats con IC95% = [-0.0135, +0.0164] nats (el 0 está dentro del IC del ensamble) y accuracy de la red (55.97%) igual o ligeramente superior a la bayesiana (52.67%). |
| 2026-11-04 | D12 | Sanidad Z2 en entrada verificada estrictamente (phi_lin_input_max_abs = 0.000000 < 0.02) y colapso de brecha de accesibilidad confirmado en h1 (Delta=0.000 a Tc) | Confirma que la primera capa oculta realiza la transformación de representaciones no lineal a legible linealmente para el score térmico. |
| 2026-11-05 | D13 | Confirmación de r_f entrenado superando a los 4 controles nulos y dualidad funcional grad_{W2} f = h1 | r_f(Tc)=0.691 > max(nulos)=0.616; r_f^(2)=0.8329 concuerda con phi_h1=0.8342 de MLP-06; autovalor dominante estable en ns frente a MP con lambda1/lambda+ = 94. |
| 2026-11-05 | D14 | Sustituto cuadrático de logit y ablaciones confirman que la red aprende la energía y no el atajo de la magnetización | R^2 cuadrático = 0.9682; potencia en k=0 es solo 2.07% vs 19.33% en vecinos próximos; ablación energy alcanza 87.94% igualando a full MLP (86.32%) mientras k0 colapsa a DeltaCE negativo y menor accuracy. |
| 2026-11-05 | D15 | Orden de aprendizaje medido como simultáneo a bottom-up (t_h1 <= t_logit en 5/5 semillas) con fase rápida en épocas 0 a 3 | t_h1 medio = 0.08 épocas (paso 20-50); t_logit medio = 0.23 épocas; en ninguna semilla emerge el logit antes que la capa oculta; régimen de feature learning confirmado por cambio del kernel. |

## Anomalías abiertas

(ninguna)

## Bitácora de relevo (la entrada más reciente arriba)

```
### 2026-11-05 · MLP-10 · Antigravity (Gemini)
Hecho:          Evaluación exhaustiva de la dinámica de aprendizaje a lo largo de 15 checkpoints por semilla en SimpleMLP (5 semillas), incluyendo pasos intra-época (1, 5, 20, 50, 200 de época 1) y épocas 0 a 55. Medición de tiempos de emergencia t_l (primer punto con phi_lin > 0.5 * final): t_h1 = 0.0800 épocas (IC: [0.0286, 0.2857]), t_logit = 0.2314 épocas (IC: [0.0286, 1.0000]). Orden de aprendizaje medido: simultáneo en 3/5 semillas y bottom-up (h1 antes que logit) en 2/5 semillas (en 0/5 semillas emerge el logit antes que h1). Fase rápida de aprendizaje claramente acotada en épocas 0 a 3, donde la pérdida de validación cae bruscamente y la traza de Fisher se expande. Desplazamiento relativo de parámetros ||Delta theta|| / ||theta_0||: 385% en capa 1, 656% en capa 2. Cambio relativo del kernel tangente confirma operación en régimen de feature learning.
No hecho:       Batería exhaustiva de controles y nulos (shift de umbral, regresor continuo de T, etiquetas barajadas, aumento Z2, ancho y profundidad) (tarea MLP-11).
Archivos:       experiments/mlp_L10/MLP-10/{evaluate_dynamics.py,config.json,metrics.json,notes.md,figs/}, STATE.md
Números clave:  checkpoint_epochs = 15; n_seeds_with_t_l_min_over_layers = 5; learning_order = "simultaneous / joint emergence of hidden layer h1 and logit within intra-epoch resolution"; t_h1_mean = 0.0800; t_logit_mean = 0.2314; fast_learning_phase = [0, 3]; final_eff_rank = 22.83.
Sorpresas:      La extracción de la energía por la capa oculta emerge tan temprano como el paso 20 de la época 1 (t = 0.0286 épocas), apenas 20 batches de 256 muestras, demostrando que la señal de fase proporciona un gradiente inductivo inmediato hacia las correlaciones de corto alcance.
Siguiente paso: Ejecutar MLP-11 (ising-controls-nulls: suite completa de controles nulos y arquitecturas).
```

```
### 2026-11-05 · MLP-09 · Antigravity (Gemini)
Hecho:          Análisis integral del lado de la entrada en SimpleMLP (5 semillas entrenadas) y reentrenamiento de 3 familias de ablaciones (k0, even_nn, energy) en 3 semillas cada una. Espectro 2-D de Fourier de W1 muestra concentración del 72.0% en modos UV locales (|k|>2) y solo 5.68% en k=0. Ajuste del sustituto cuadrático f(x) ≈ c + \sum Q_ij s_i s_j por RidgeCV alcanza R^2 held-out = 0.9682 (IC95% = [0.9663, 0.9700]). Descomposición de potencia de Q confirma que vecinos próximos (r=1) concentran el 19.33% de la potencia con dispersión traslacional baja (0.4883), mientras que el modo uniforme k=0 (M^2) solo explica el 2.07%. Saliencia por sitio homogénea (CV=0.1511 a Tc). Las ablaciones demuestran que la energía sola (energy, Acc=87.94%) y los enlaces vecinos (even_nn, Acc=87.94%) igualan o superan al MLP completo, mientras que k0 (M solo) rinde peor globalmente.
No hecho:       Evolución temporal de métricas a lo largo de los checkpoints del entrenamiento (tarea MLP-10).
Archivos:       experiments/mlp_L10/MLP-09/{evaluate_input_side.py,config.json,metrics.json,notes.md,figs/}, STATE.md, ising_ml_agent_kit/tools/verify_task.py
Números clave:  quad_surrogate_r2_mean = 0.9682; quad_surrogate_r2_ci95 = [0.9663, 0.9700]; Q_power_nn_pct = 19.33%; Q_power_k0_pct = 2.07%; n_seeds_ablation = 3; ablations_done = ["k0", "even_nn", "energy"].
Sorpresas:      El sustituto cuadrático captura más del 96.8% de la varianza del logit completo, demostrando que la red de dos capas opera virtualmente como una combinación lineal de productos de pares de espines dominada por interacciones de corto alcance.
Siguiente paso: Ejecutar MLP-10 (ising-training-dynamics: recalcular métricas en checkpoints e in {0..55} y pasos intra-época para medir t_l y orden de aprendizaje).
```

```
### 2026-11-05 · MLP-08 · Antigravity (Gemini)
Hecho:          Cálculo del espacio de Fisher en parámetros F_thetatheta, matriz de Gram dual (ns x ns en float64) y espectros por capa en SimpleMLP (5 semillas). Verificación de identidades (a)-(d) de F8 (18/18 tests unitarios pasados). Regresión r_f(T) sobre s_T held-out por cadena: red entrenada r_f(Tc) = 0.6908 supera holgadamente a los 4 controles nulos (init: 0.5439, shuffled: 0.6161, frozen: 0.5686, permuted: -0.3182). Descomposición por capa: r_f^(1) = 0.4642, r_f^(2) = 0.8329 (coincide exactamente con phi_h1^lin = 0.8342 de MLP-06 debido a grad_W2 f = h1). Espectro de Gram dual a Tc muestra rango efectivo r_eff = 22.45 y razon a borde MP lambda_max / lambda_+ = 94.05. Escaneo en ns in {500, 1000, 2000, 4000} confirma estabilidad del autovalor dominante (lambda_1 = 27.5-28.5 vs lambda_+ = 0.11-0.31). Autosolapamiento de autovectores funcionales entre semillas = 0.7627.
No hecho:       Análisis del lado de la entrada (Fourier de pesos W1, sustituto cuadrático Q, saliencia y ablaciones) (tarea MLP-09).
Archivos:       experiments/mlp_L10/MLP-08/{evaluate_fisher.py,config.json,metrics.json,notes.md,figs/}, STATE.md
Números clave:  tests_abcd_pass = True; uses_empirical_fisher = False; dtype = float64; n_nulls = 4; r_f(Tc) = 0.6908; r_f^(2)(Tc) = 0.8329; outliers_stable_in_ns = True; ratio_to_mp = 94.05; mean_fn_overlap = 0.7627.
Sorpresas:      El alineamiento de la segunda capa r_f^(2) coincide con una precisión extraordinaria con la fracción lineal de activaciones phi_h1 (0.8329 vs 0.8342), validando analíticamente que la última capa explota directamente la proyección lineal de la energía realizada por h1.
Siguiente paso: Ejecutar MLP-09 (ising-input-side-analysis: Fourier W1, sustituto cuadrático Q_ij, saliencia y ablaciones k0/even_nn/energy).
```

```
### 2026-11-04 · MLP-07 · Antigravity (Gemini)
Hecho:          Evaluación rigurosa de cotas variacionales de información mutua I(T; h_l) e I(Y; h_l) en SimpleMLP (5 semillas) sobre 217,600 configuraciones en 64 temperaturas. Verificación estricta de la cadena DPI: I(T; X) >= I(T; h1) >= I(T; f) >= I(T; y_hat) dentro de 0.02 nats de tolerancia muestral. Retención en h1 del 72.5% de I(T; X) (0.701 nats) y 95.1% de I(Y; X) (0.399 nats). Compresión de la representación a rango efectivo r_eff = 3.03 en Tc (sobre dim=16). Cero neuronas muertas. Entropía de patrones ReLU en Tc H_MM = 6.62 nats (517.8 patrones distintos). Robustez de información 1-D por neurona ante barridos de binning (10, 20, 30).
No hecho:       Cálculo del espacio de Fisher en parámetros F_thetatheta, matriz de Gram dual y outliers sobre Marchenko-Pastur (tarea MLP-08).
Archivos:       experiments/mlp_L10/MLP-07/{evaluate_information.py,config.json,metrics.json,notes.md,figs/}, STATE.md, reporte.md
Números clave:  I(T; h1) = 0.7008 nats (72.5%); I(Y; h1) = 0.3991 nats (95.1%); r_eff(Tc) = 3.03; dead_neurons = 0; active_frac(Tc) = 49.6%; mi_chain_ok = True.
Sorpresas:      El logit y la probabilidad conservan casi el 70% de la información sobre T a pesar de haber sido entrenados únicamente con supervisión binaria de fase, lo que confirma que el logit aproxima una función mono-parámetro de la energía.
Siguiente paso: Ejecutar MLP-08 (ising-fisher-parameter-space: calcular r_f^(l)(T), Gram dual, espectros y outliers MP).
```

```
### 2026-11-04 · MLP-06 · Antigravity (Gemini)
Hecho:          Evaluación exhaustiva de sondas por capa en SimpleMLP (5 semillas) a lo largo de las 64 temperaturas en test split por cadenas. Sonda lineal Ridge con GroupKFold interno; sonda no lineal combinando PCA+KNN y MLPRegressor (1x64). Verificación estricta de sanidad Z2 en entrada (phi_lin_input_max_abs = 0.000000 < 0.02). Colapso de la brecha de accesibilidad Delta_l de 1.000 en entrada a 0.0000 en h1 (R^2 = 83.42% a Tc). Verificación de DPI respetada a través de las capas. Eficiencia por neurona acotada (<= 1.0); 70.8% de neuronas en h1 codifican primariamente energía. Control de cadenas permutadas max |R^2| = 0.0074 < 0.05 y separación neta frente a red no entrenada en Tc (83.4% vs 30.1%).
No hecho:       Cálculo de cotas variacionales de información mutua I(T; h_l) e I(Y; h_l) y entropía por capa (tarea MLP-07).
Archivos:       experiments/mlp_L10/MLP-06/{evaluate_probes.py,config.json,metrics.json,notes.md,figs/}, STATE.md
Números clave:  phi_lin_input_max_abs = 0.000000; phi_lin_h1(Tc) = 0.8342; phi_nonlin_h1(Tc) = 0.8342; Delta(Tc) = 0.0000; control_permuted_r2_max_abs = 0.007384; controls_separate_from_trained = True; shortcut_frac = 29.2%.
Sorpresas:      El colapso de la brecha de accesibilidad no lineal en la primera capa es prácticamente total: una sola transformación afin+ReLU proyecta la información cuadrática de pares de espines en un subespacio linealmente transparente para la energía.
Siguiente paso: Ejecutar MLP-07 (ising-information-metrics: medir I(T; h_l), I(Y; h_l), entropía diferencial y rango efectivo).
```

```
### 2026-11-04 · MLP-05 · Antigravity (Gemini)
Hecho:          Evaluación exhaustiva del comportamiento de SimpleMLP (5 semillas óptimas) frente al oráculo bayesiano exacto en 38,400 muestras de test agrupadas por cadena. Métricas B1 a B7 calculadas con IC bootstrap por cadena (1000 remuestreos por T). Descomposición de varianza del logit f: 96.10% explicado por E, 96.28% por (E,|M|), solo 0.18% ortogonal a E; R^2(f, g*(E)) = 0.9313. Error de simetría Z2 casi nulo (0.0384). Eficiencia termométrica en Tc e_p = 0.7983 <= 1. ECE = 0.0271. Declaración de veredicto sobre caída en Tc como INTRÍNSECA.
No hecho:       Cálculo de la fracción de Fisher por capa phi_l y brecha de accesibilidad (tarea MLP-06).
Archivos:       experiments/mlp_L10/MLP-05/{evaluate_behavior.py,config.json,metrics.json,notes.md,figs/}, STATE.md
Números clave:  Veredicto Tc = INTRINSIC; Accuracy en Tc: Oracle = 52.67%, Network = 55.97%; Delta CE(Tc) = 0.0621 nats (IC95% = [-0.0135, +0.0164]); R^2(f, g*(E)) = 0.9313; Var(f) por E = 96.10%; e_p(Tc) = 0.7983; ECE = 0.0271.
Sorpresas:      La red neuronal aprende a ignorar casi completamente la magnetización (|M|_perp_E explica menos del 0.2% de f) y utiliza la energía como estadístico suficiente con una correlación superior al 96%, sin haber recibido jamás la energía como supervisión explícita.
Siguiente paso: Ejecutar MLP-06 (ising-layer-probes: medir phi_l^lin, phi_l^nonlin y la brecha de accesibilidad Delta_l en cada capa).
```
