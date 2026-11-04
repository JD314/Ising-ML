---
name: ising-resume-and-handoff
description: Use at the start and end of every session on the Ising-ML MLP campaign. Retoma el trabajo sin releer todo el repo, selecciona la siguiente tarea pendiente y deja el relevo escrito. También ejecuta la auditoría MLP-00 del repositorio.
---

# Skill 00 · Retomar y relevar (incluye MLP-00)

## Al empezar (máx. 5 minutos de lectura)
1. Lee `AGENT_PROTOCOL.md` y `STATE.md`. Nada más.
2. En el tablero de `STATE.md`, elige la primera tarea con estado `TODO` o `IN_PROGRESS` cuyas dependencias estén en `DONE`/`DONE_NEGATIVE`.
3. Abre solo la skill de esa tarea (columna *Skill*).
4. Si `STATE.md` y la realidad (archivos en `experiments/mlp_L10/`) no coinciden, registra la anomalía y repara el estado antes de seguir.

## Al terminar
1. Marca el estado, añade artefactos y el hallazgo en una frase en el tablero.
2. Escribe la entrada de la bitácora (plantilla en `STATE.md`).
3. Si hay decisión nueva, regístrala en *Decisiones*.
4. Detente. No inicies la siguiente tarea.

## MLP-00 · Auditoría del repositorio
Responde con evidencia (archivo y línea) cada pregunta; guarda en `experiments/mlp_L10/MLP-00/notes.md`.

| Pregunta | Dónde mirar |
|---|---|
| ¿BC periódicas u abiertas? ¿vecinos con `% L`? | `mc.py` |
| ¿Se aleatoriza el signo global? ¿estado inicial? | `mc.py`, `scripts/generate_mc.py` |
| ¿Esquema de recocido, termalización, espaciado? | `mc.py` |
| ¿Formato HDF5 y split por cadena? | `dataset.py` |
| ¿Arquitectura exacta de `SimpleMLP` (capas, activación, sesgos, init)? | `models/classifier.py` |
| ¿Cómo se calcula `r_f` (features, ridge, held-out)? | `fisher.py` |
| ¿Se usa Fisher con etiquetas del modelo? | `fisher.py` |
| ¿Qué cubren los 44 tests? | `tests/` |
| ¿Qué dice la auditoría de código? | `CONTEXT_ISING_ML.md` |

Comandos: `python -m pytest -v` y `python scripts/run_first_exploration.py --L 8 --nT 8` (humo).

**Aceptación:** tests ejecutados, tabla de hallazgos completa, "Parámetros fijados" de `STATE.md` rellenos, riesgos R1/R2 confirmados o descartados.
