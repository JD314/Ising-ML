# STATE · estado vivo (el agente actualiza este archivo)

**Última actualización:** (pendiente)  **Agente/persona:** (pendiente)
**Tarea activa:** ninguna  **Siguiente acción:** ejecutar `skills/00-resume-and-handoff` y luego MLP-00.

## Tablero

| ID | Tarea | Skill | Depende de | Estado | Artefactos | Hallazgo en una frase |
|---|---|---|---|---|---|---|
| MLP-00 | Auditoría del repo y entorno | 00 | — | TODO | | |
| MLP-01 | Oráculo exacto L=10 | 01 | 00 | TODO | | |
| MLP-02 | Datos y auditoría vs exacto | 02 | 01 | TODO | | |
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
| Arquitectura MLP de referencia | (de MLP-00; si no existe: 100→128→64→1, ReLU) |
| BC del generador | (MLP-00/02) |
| Grilla de T final | (MLP-02) |
| Muestras por T / cadenas | (MLP-02) |
| T_peak(L=10) de F_TT exacto | (MLP-01) |
| Accuracy bayesiana en T_c | (MLP-01/03) |
| I(T;X) exacto | (MLP-01) |
| Semillas de red | 0–4 |

## Decisiones (añadir al final)

| Fecha | ID | Decisión | Evidencia |
|---|---|---|---|

## Anomalías abiertas

(ninguna)

## Bitácora de relevo (la entrada más reciente arriba)

```
### AAAA-MM-DD · <ID tarea> · <agente>
Hecho:          ...
No hecho:       ...
Archivos:       ...
Números clave:  ...
Sorpresas:      ...
Siguiente paso: ...
```
