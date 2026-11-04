# STATE · estado vivo (el agente actualiza este archivo)

**Última actualización:** 2026-11-04  **Agente/persona:** Antigravity (Gemini)  
**Tarea activa:** ninguna  **Siguiente acción:** ejecutar MLP-05 (comportamiento de la red frente al oráculo bayesiano).

## Tablero

| ID | Tarea | Skill | Depende de | Estado | Artefactos | Hallazgo en una frase |
|---|---|---|---|---|---|---|
| MLP-00 | Auditoría del repo y entorno | 00 | — | DONE | `experiments/mlp_L10/MLP-00/{notes.md,metrics.json,config.json}` | 44 tests pasan; PBC y sweeps confirmados en `ising_fisher/`; R1/R2 de legacy mitigados en nuevo diseño. |
| MLP-01 | Oráculo exacto L=10 | 01 | 00 | DONE | `ising_fisher/exact_L10.py`, `tests/test_exact_L10.py`, `experiments/mlp_L10/MLP-01/{metrics.json,figs/F_TT_exact_L10.png,notes.md}` | Oráculo exacto de matriz de transferencia validado (<1e-10 en ln Z a L<=4); T_peak(F_TT)=2.279, I(T;X)=0.957 nats (prov). |
| MLP-02 | Datos y auditoría vs exacto | 02 | 01 | DONE | `data/L10_data_audit.h5`, `experiments/mlp_L10/MLP-02/{config.json,metrics.json,figs/,notes.md}` | 256k muestras en 64 T; balance Z2 exacto (0.500); 7.8% rechazos p<0.01 (Holm); I(T;X)=0.967 nats, I(Y;X)=0.420 nats. |
| MLP-03 | Baselines (E, \|M\|, lineal, RF) | 04 | 02 | DONE | `experiments/mlp_L10/MLP-03/{evaluate_baselines.py,config.json,metrics.json,figs/,notes.md}` | Logística en E alcanza CE a +2.16% del oráculo (Acc 87.9% vs 88.1%); lineal en s colapsa a CE=0.693 nats por Z2. |
| MLP-04 | Entrenamiento MLP con checkpoints | 03 | 02 | DONE | `ckpt/seed{0..4}/*.pt`, `experiments/mlp_L10/MLP-04/{train_mlp.py,config.json,metrics.json,figs/,notes.md}` | 5 semillas convergen en 50-69 épocas; 4/5 semillas estables (<1% var); test acc media 86.32%; checkpoints guardados. |
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
| Épocas hasta convergencia por semilla | Seed 0: 69, Seed 1: 63, Seed 2: 63, Seed 3: 50, Seed 4: 54 (media 59.8 épocas) |
| BC del generador | Periódicas (PBC toroidal exacta con `% L` en `ising_fisher/mc.py`) |
| Grilla de T final | 64 temperaturas: [0.1:0.1:4.0] + [2.00:0.02:2.60] sin duplicados |
| Muestras por T / cadenas | 20 cadenas independientes, 200 muestras/cadena (4,000 muestras/T; 256,000 en total). Split fijo 70/15/15 por cadena |
| T_peak(L=10) de F_TT exacto | 2.27900 (máx F_TT = 24.4844 total, f_TT = 0.2448/sitio; T_peak(C) = 2.34500) |
| Accuracy bayesiana en T_c | 51.75% a T=2.2600 (global del clasificador bayesiano óptimo en test: 88.12%) |
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

## Anomalías abiertas

(ninguna)

## Bitácora de relevo (la entrada más reciente arriba)

```
### 2026-11-04 · MLP-04 · Antigravity (Gemini)
Hecho:          Entrenamiento completo de SimpleMLP (100->16->1, ReLU, 1633 parámetros) en las 5 semillas independientes de red (0 a 4) sobre 179,200 muestras de entrenamiento agrupadas por cadena. Early stopping con paciencia 10 (convergencia alcanzada entre 50 y 69 épocas). 4/5 semillas satisfacen <1% de variación de pérdida en últimas 5 épocas. Accuracy de test promedio 86.32%. Guardado de checkpoints fijos en ckpt/seed{s}/ (épocas fijas 0..55, pasos intra-época 1..200, best.pt y final.pt).
No hecho:       Análisis detallado de sondas por capa y comportamiento vs oráculo (pertenece a MLP-05 y MLP-06).
Archivos:       experiments/mlp_L10/MLP-04/{train_mlp.py,config.json,metrics.json,notes.md,figs/}, ckpt/seed{0..4}/*.pt, STATE.md
Números clave:  Best epochs: [69, 63, 63, 50, 54]; Test accuracy media = 86.32% (+-0.45%); Stable seeds = 4/5; 347.8s tiempo total de cómputo.
Sorpresas:      El entrenamiento de 5 semillas en CPU con PyTorch tomó sólo ~5.8 minutos gracias a la arquitectura compacta de 1,633 parámetros, asegurando reproducibilidad y convergencia sin sobreajuste.
Siguiente paso: Ejecutar MLP-05 (comportamiento de la red entrenada vs oráculo bayesiano exacto en modo red).
```
