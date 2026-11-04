---
name: ising-controls-nulls
description: Use for task MLP-11. Repite las metricas centrales del MLP de Ising bajo controles (umbral de etiqueta desplazado, regresor de T, etiquetas barajadas, signo fijo vs aleatorizado, red sin entrenar, ancho y profundidad, augmentacion Z2) para decidir que hallazgos sobreviven.
---

# Skill 10 · Controles y nulos

Teoría: T3, T5 (punto 5), T7. Requiere MLP-05 a MLP-10.

## Controles
| Control | Cambio | Qué se pregunta |
|---|---|---|
| C-thr | Umbral de etiqueta en `T=1.5` y `T=3.0` (en vez de T_c) | ¿Los picos de incertidumbre de salida siguen al umbral (C8)? ¿`φ^nonlin` de capas intermedias sigue mostrando la física en T_c / T_peak? |
| C-reg | Regresor de T (MSE) o clasificador de K bandas, sin umbral | ¿Dónde resuelve la red T mejor? Compárese `F_out(T)/F_TT` con `√F_TT` |
| C-shuf | Etiquetas barajadas | Línea base de todas las métricas (debe colapsar) |
| C-sign | Datos con signo fijo vs. aleatorizado | ¿Aparece dependencia impar de M? ¿Cambian `φ_input^lin` y `Q`? |
| C-init | Red sin entrenar | Contribución de la arquitectura |
| C-size | Ancho `{64,128,256}`, profundidad `{1,2,3}` | ¿El hallazgo depende de la capacidad? (mantener `ϵ_lr/|B|`) |
| C-z2 | Augmentación `x → −x` durante el entrenamiento | ¿Aprende la simetría? Efecto sobre B6 y C7 |

## Procedimiento
1. Para cada control, entrena 3 semillas (5 si el costo lo permite) con la misma regla de parada.
2. Recalcula **solo las métricas P0**: B2, B5, D1–D4, E1–E2, F1–F2, G3.
3. Resume en una tabla `hallazgo × control` con tres estados: `sobrevive`, `se desplaza` (indica a dónde), `desaparece`.
4. Revisa explícitamente los riesgos R2 y R5 de `AGENT_CONTEXT.md` y márcalos como confirmados o descartados.

## Entregables
Tabla de supervivencia de hallazgos; figuras comparativas (control vs. referencia) para cada métrica P0.

## Aceptación
Para cada afirmación central candidata se declara su estado bajo cada control. Una afirmación que no sobrevive a C-shuf o C-init **no** entra al reporte como resultado.
