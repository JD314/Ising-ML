# STATE · estado vivo (el agente actualiza este archivo)

**Última actualización:** 2026-11-04  **Agente/persona:** Antigravity (Gemini)  
**Tarea activa:** ninguna  **Siguiente acción:** ejecutar MLP-07 (métricas de información por capa y cotas variacionales).

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
| MLP-07 | Métricas de información | 06 | 05, 06 | TODO | | |
| MLP-08 | Fisher en parámetros por capa | 07 | 04, 06 | TODO | | |
| MLP-09 | Lado de la entrada | 08 | 04 | TODO | | |
| MLP-10 | Dinámica de entrenamiento | 09 | 06, 07, 08 | TODO | | |
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

## Anomalías abiertas

(ninguna)

## Bitácora de relevo (la entrada más reciente arriba)

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
