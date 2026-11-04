# STATE · estado vivo (el agente actualiza este archivo)

**Última actualización:** 2026-11-03  **Agente/persona:** Antigravity (Gemini)  
**Tarea activa:** ninguna  **Siguiente acción:** ejecutar MLP-03 (baselines de clasificación: E, |M|, lineal, RF) o MLP-04 (entrenamiento de SimpleMLP con checkpoints).

## Tablero

| ID | Tarea | Skill | Depende de | Estado | Artefactos | Hallazgo en una frase |
|---|---|---|---|---|---|---|
| MLP-00 | Auditoría del repo y entorno | 00 | — | DONE | `experiments/mlp_L10/MLP-00/{notes.md,metrics.json,config.json}` | 44 tests pasan; PBC y sweeps confirmados en `ising_fisher/`; R1/R2 de legacy mitigados en nuevo diseño. |
| MLP-01 | Oráculo exacto L=10 | 01 | 00 | DONE | `ising_fisher/exact_L10.py`, `tests/test_exact_L10.py`, `experiments/mlp_L10/MLP-01/{metrics.json,figs/F_TT_exact_L10.png,notes.md}` | Oráculo exacto de matriz de transferencia validado (<1e-10 en ln Z a L<=4); T_peak(F_TT)=2.279, I(T;X)=0.957 nats (prov). |
| MLP-02 | Datos y auditoría vs exacto | 02 | 01 | DONE | `data/L10_data_audit.h5`, `experiments/mlp_L10/MLP-02/{config.json,metrics.json,figs/,notes.md}` | 256k muestras en 64 T; balance Z2 exacto (0.500); 7.8% rechazos p<0.01 (Holm); I(T;X)=0.967 nats, I(Y;X)=0.420 nats. |
| MLP-03 | Baselines (E, \|M\|, lineal, RF) | 04 | 02 | TODO | | |
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
| Accuracy bayesiana en T_c | 51.75% a T=2.2600 (global del clasificador bayesiano óptimo: 88.04%) |
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

## Anomalías abiertas

(ninguna)

## Bitácora de relevo (la entrada más reciente arriba)

```
### 2026-11-03 · MLP-02 · Antigravity (Gemini)
Hecho:          Generación de dataset de 256,000 configuraciones L=10 con mc.py en 64 temperaturas (20 cadenas/T, termalización 2000 barridos, thin=20 >= 2 tau_int). Aleatorización global de paridad Z2 preservando signos originales. Almacenamiento en data/L10_data_audit.h5 con partición agrupada por cadena 70/15/15 fija. Auditoría contra oráculo exacto de matriz de transferencia: 7.8% de temperaturas con p < 0.01 bajo Holm-Bonferroni (umbral aceptación <= 10%), balance de signo 0.4999 (dentro de 0.50 +- 0.02), crecimiento consistente de S(0) hacia Tc. Evaluación del oráculo bayesiano sobre datos reales: accuracy global 88.04%, en Tc 51.75%, I(T;X)=0.9671 nats, I(Y;X)=0.4197 nats.
No hecho:       Entrenamiento de redes ni cálculo de baselines (pertenece a MLP-03 y MLP-04).
Archivos:       scripts/generate_and_audit_dataset.py, data/L10_data_audit.h5, experiments/mlp_L10/MLP-02/{config.json,metrics.json,figs/,notes.md}, STATE.md
Números clave:  256,000 muestras totales en 64 T; accuracy bayesiana en Tc = 51.75%; I(T;X)=0.9671 nats; I(Y;X)=0.4197 nats; balance signo = 0.4999.
Sorpresas:      La accuracy bayesiana óptima cae exactamente a ~50% (51.75%) en Tc en volumen finito L=10, marcando el techo teórico irrebasable para cualquier clasificador en la vecindad crítica.
Siguiente paso: Ejecutar MLP-03 (baselines de referencia) o MLP-04 (entrenamiento del clasificador SimpleMLP con 5 semillas y checkpoints).
```
