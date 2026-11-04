# Ising-ML · Kit de agente: ¿dónde y cómo aprende patrones la red? (MLP, L = 10)

Este kit organiza una campaña de experimentos para medir **información, entropía y geometría de Fisher** en cada capa, neurona, entrada y salida de un clasificador MLP entrenado sobre configuraciones del modelo de Ising 2D (10×10) a través de la transición de fase. Está diseñado para que un agente (o una persona) ejecute **una tarea a la vez** y pueda retomar el trabajo sin releer todo.

> Base teórica: la charla *Fisher Information Geometry in Parameter Space: State of the Art 2026* (plan SYNTHESIS) y el repositorio `JD314/Ising-ML` (paquete `ising_fisher/`). Base empírica: la presentación *Ising ML* (MLP, CNN, ViT, U-Net sobre L = 10, 20, 30).

## Idea central en tres líneas

1. A h = 0 la energía E es **estadístico suficiente** de T, y la información de Fisher física es `F_TT = Var(E)/T⁴`. Cualquier función `g(x)` de la configuración cumple `(∂_T E[g])² ≤ Var(g)·F_TT`, con igualdad si y solo si `g` es afín en E. Por tanto, **la eficiencia termométrica de una neurona a T fijo es ρ²(a_j, E | T)**.
2. Por simetría Z₂, `Cov(s_i, E) = 0`: una lectura *lineal* de la entrada no contiene información sobre T. La red tiene que **fabricar rasgos pares** (energía, |M|). La pregunta experimental es *en qué capa y en qué momento del entrenamiento* ocurre.
3. En L = 10 todo lo físico es **exacto** (matriz de transferencia): `ln p(x|T)`, `F_TT`, el clasificador bayesiano óptimo y la información mutua `I(T;X)`. La red se mide contra ese oráculo, no contra intuiciones.

## Mapa de archivos

| Archivo | Para qué sirve |
|---|---|
| `AGENT_PROTOCOL.md` | Reglas de trabajo: una tarea a la vez, definición de "hecho", prohibiciones |
| `AGENT_CONTEXT.md` | Contexto estable: proyecto, convenciones, decisiones, riesgos |
| `STATE.md` | Estado mutable: qué está hecho, qué sigue, bitácora de relevo |
| `THEORY.md` | Definiciones y garantías teóricas con las que debe alinearse todo |
| `METRICS.md` | Catálogo de métricas por capa, neurona, entrada y salida |
| `ROADMAP.md` | Plan de experimentos MLP-00 … MLP-11 con criterios go/no-go |
| `skills/NN-*/SKILL.md` | Una skill por tarea, ejecutable de forma aislada |

## Cómo lanzar un agente

Copia este prompt (ajusta solo la ruta):

```
Trabaja en el repositorio Ising-ML siguiendo ising_ml_agent_kit/.
1) Lee AGENT_PROTOCOL.md y STATE.md. No leas nada más todavía.
2) Identifica la siguiente tarea pendiente en STATE.md y abre SOLO su SKILL.md.
3) Si la skill lo indica, consulta la sección puntual de THEORY.md o METRICS.md.
4) Ejecuta esa única tarea, valida con sus criterios de aceptación y actualiza STATE.md.
5) Detente y reporta. No empieces la siguiente tarea.
```

## Orden de ejecución

```
MLP-00 auditoría repo ─► MLP-01 oráculo exacto ─► MLP-02 datos ─► MLP-03 baselines
        └─► MLP-04 entrenamiento ─► MLP-05 comportamiento ─► MLP-06 sondas por capa
              └─► MLP-07 métricas de información ─► MLP-08 Fisher en parámetros
                    └─► MLP-09 lado de la entrada ─► MLP-10 dinámica ─► MLP-11 controles ─► síntesis
```

Las dependencias están en `ROADMAP.md`; ninguna tarea se salta una dependencia.

## Alcance y honestidad

- Este kit fue escrito leyendo el README del repo, la presentación y la charla de revisión. **No se inspeccionó el código fuente** (`fisher.py`, `train.py`, `mc.py`, modelos). La tarea MLP-00 existe precisamente para auditarlo antes de asumir nada.
- Las hipótesis en `ROADMAP.md` son predicciones falsables, no resultados. Un resultado negativo se registra igual que uno positivo.
