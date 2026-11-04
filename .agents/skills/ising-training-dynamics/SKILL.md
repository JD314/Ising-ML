---
name: ising-training-dynamics
description: Use for task MLP-10. Recalcula las metricas de sondas, informacion, Fisher y entrada en cada checkpoint del entrenamiento del MLP de Ising para medir cuando y en que orden aprende cada capa.
---

# Skill 09 · Dinámica del aprendizaje (H1–H5)

Requiere MLP-06, MLP-07 y MLP-08 (los pipelines de medida ya existen y validados).

## Procedimiento
1. Para cada semilla y checkpoint `e ∈ {0,1,2,3,5,8,13,21,34,55}` y pasos intra-época `{1,5,20,50,200}`, ejecuta **los mismos scripts** de MLP-06/07/08 sobre el **mismo** conjunto de sondas/test. No cambies hiperparámetros de sondas entre checkpoints.
2. Para limitar costo: usa solo T ∈ `{1.5, 2.0, 2.2, 2.3, 2.5, 3.0}` más T_peak; todas las T solo en inicio y final.
3. **H1 tiempo de emergencia:** `t_l` = primera época con `φ_l^lin(T_c) > 0.5·φ_l^lin,final`. Reporta por capa, media e IC entre semillas.
4. **H2 orden:** compara `t_1`, `t_2`, `t_logit`; reporta si el orden coincide en ≥ 4/5 semillas.
5. **H3:** trayectoria de `Tr F_θ^{(l)}`, rango efectivo y número de outliers (con ns fijo).
6. **H4:** `‖θ_t − θ_0‖/‖θ_0‖` por capa; **H5** (opcional): cambio relativo del kernel `∇f∇fᵀ` (régimen lazy vs. feature learning).
7. Curvas de pérdida de train/val superpuestas a las trayectorias para fijar el momento de "fase rápida".

## Entregables
Mapas capa × época de `φ^lin`, `φ^nonlin`, `I(T;h_l)`, `r_f^{(l)}`; tabla de `t_l`; figura de orden de aprendizaje.

## Aceptación
`t_l` definido en ≥ 4/5 semillas; curvas con IC; el orden de aprendizaje se declara como **medido** (con su incertidumbre), no supuesto.

## Cuidado
Mismo `ϵ_lr/|B|` y misma regla de parada que MLP-04. Un cambio de tasa de aprendizaje cambia los tiempos y no se puede comparar (K7).
