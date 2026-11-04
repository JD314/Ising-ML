# STATE · estado vivo (el agente actualiza este archivo)

**Última actualización:** 2026-11-04  **Agente/persona:** Antigravity (Gemini)  
**Tarea activa:** ninguna  **Siguiente acción:** ejecutar MLP-04 (entrenamiento del clasificador de referencia SimpleMLP con 5 semillas y checkpoints).

## Tablero

| ID | Tarea | Skill | Depende de | Estado | Artefactos | Hallazgo en una frase |
|---|---|---|---|---|---|---|
| MLP-00 | Auditoría del repo y entorno | 00 | — | DONE | `experiments/mlp_L10/MLP-00/{notes.md,metrics.json,config.json}` | 44 tests pasan; PBC y sweeps confirmados en `ising_fisher/`; R1/R2 de legacy mitigados en nuevo diseño. |
| MLP-01 | Oráculo exacto L=10 | 01 | 00 | DONE | `ising_fisher/exact_L10.py`, `tests/test_exact_L10.py`, `experiments/mlp_L10/MLP-01/{metrics.json,figs/F_TT_exact_L10.png,notes.md}` | Oráculo exacto de matriz de transferencia validado (<1e-10 en ln Z a L<=4); T_peak(F_TT)=2.279, I(T;X)=0.957 nats (prov). |
| MLP-02 | Datos y auditoría vs exacto | 02 | 01 | DONE | `data/L10_data_audit.h5`, `experiments/mlp_L10/MLP-02/{config.json,metrics.json,figs/,notes.md}` | 256k muestras en 64 T; balance Z2 exacto (0.500); 7.8% rechazos p<0.01 (Holm); I(T;X)=0.967 nats, I(Y;X)=0.420 nats. |
| MLP-03 | Baselines (E, \|M\|, lineal, RF) | 04 | 02 | DONE | `experiments/mlp_L10/MLP-03/{evaluate_baselines.py,config.json,metrics.json,figs/,notes.md}` | Logística en E alcanza CE a +2.16% del oráculo (Acc 87.9% vs 88.1%); lineal en s colapsa a CE=0.693 nats por Z2. |
| MLP-04 | Entrenamiento MLP con checkpoints | 03 | 02 | TODO | | |
| MLP-05 | Comportamiento vs oráculo | 04 | 03, 04 | TODO | | |
| MLP-06 | Sondas por capa | 05 | 04 | TODO | | |
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
| BC del generador | Periódicas (PBC toroidal exacta con `% L` en `ising_fisher/mc.py`) |
| Grilla de T final | 64 temperaturas: [0.1:0.1:4.0] + [2.00:0.02:2.60] sin duplicados |
| Muestras por T / cadenas | 20 cadenas independientes, 200 muestras/cadena (4,000 muestras/T; 256,000 en total). Split fijo 70/15/15 por cadena |
| T_peak(L=10) de F_TT exacto | 2.27900 (máx F_TT = 24.4844 total, f_TT = 0.2448/sitio; T_peak(C) = 2.34500) |
| Accuracy bayesiana en T_c | 51.75% a T=2.2600 (global del clasificador bayesiano óptimo en test: 88.12%) |
| I(T;X) exacto | 0.9671 nats (calculado sobre dataset completo auditado de 256k muestras; techo ln 64 = 4.1589 nats) |
| I(Y;X) exacto | 0.4197 nats (calculado sobre dataset completo auditado; techo H(Y) = 0.6927 nats) |
| Semillas de red | 0–4 |

## Decisiones (añadir al final)

| Fecha | ID | Decisión | Evidencia |
|---|---|---|---|
| 2026-11-03 | D0 | Enlaces simbólicos de conveniencia `STATE.md`, `AGENT_PROTOCOL.md` y `AGENT_CONTEXT.md` creados en la raíz apuntando a `ising_ml_agent_kit/` | Mantiene compatibilidad con las instrucciones que esperan `STATE.md` en la raíz. |
| 2026-11-03 | D6 | Se adopta `SimpleMLP` (`100 -> 16 -> 1`, ReLU, P=1633) existente en `ising_fisher/models/classifier.py` como arquitectura base de referencia | Verificado en `classifier.py` e integrado en `test_models.py` y `run_first_exploration.py`. |
| 2026-11-03 | D7 | Grilla de temperaturas fijada en 64 puntos ([0.1:0.1:4.0] más refinada [2.00:0.02:2.60]) | Especificada en `ROADMAP.md` y `THEORY.md` para capturar la vecindad de $T_c$ y $T_{\text{peak}}$. |
| 2026-11-03 | D8 | Esquema de split fijo de cadenas por temperatura: 14 cadenas train (70%), 3 val (15%), 3 test (15%) | Permite evaluación held-out sin filtración de autocorrelación entre train y test. |
| 2026-11-04 | D9 | Margen de exceso de CE de la logística sobre E documentado en +2.16% | El oráculo exacto P*(y=1\|E) tiene un logit con curvatura intrínseca en volumen finito; una sigmoide mono-parámetro alcanza CE=0.2741 vs CE=0.2683 del oráculo (exceso 2.16%, o +0.72% al incluir \|M\|). |

## Anomalías abiertas

(ninguna)

## Bitácora de relevo (la entrada más reciente arriba)

```
### 2026-11-04 · MLP-03 · Antigravity (Gemini)
Hecho:          Evaluación exhaustiva de 6 modelos baselines en el split de prueba (38,400 muestras) con intervalos de confianza bootstrap por cadena (1000 remuestreos por T). Comparación de accuracy, CE, H(p) y Delta CE frente al oráculo bayesiano óptimo P*(y=1|E). Generación de figuras comparativas y métricas completas.
No hecho:       Entrenamiento de la red neuronal SimpleMLP con checkpoints (tarea MLP-04).
Archivos:       experiments/mlp_L10/MLP-03/evaluate_baselines.py, experiments/mlp_L10/MLP-03/config.json, experiments/mlp_L10/MLP-03/metrics.json, experiments/mlp_L10/MLP-03/notes.md, experiments/mlp_L10/MLP-03/figs/baselines_accuracy_vs_T.png, experiments/mlp_L10/MLP-03/figs/delta_ce_vs_oracle.png, STATE.md
Números clave:  CE Oracle=0.268343, Acc Oracle=88.12%; CE Logistic(E)=0.274138, Acc Logistic(E)=87.94% (exceso CE = +2.16%); CE Linear(s)=0.692697, Acc Linear(s)=57.41%; CE Frozen MLP=0.528812, Acc Frozen MLP=75.34%.
Sorpresas:      El clasificador lineal sobre los espines s colapsa exactamente al nivel de entropía de azar puro (ln 2 = 0.6931 nats), confirmando empíricamente la nulidad de features lineales bajo simetría Z2 a h=0.
Siguiente paso: Ejecutar MLP-04 (entrenamiento de SimpleMLP con 5 semillas y guardado de checkpoints fijos).
```
