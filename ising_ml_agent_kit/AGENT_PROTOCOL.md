# Protocolo del agente

## Reglas de oro

1. **Una tarea a la vez.** Cada tarea corresponde a una skill y a un ID del `ROADMAP.md`. Al terminar, se detiene.
2. **Lee lo mínimo.** Orden de lectura: `AGENT_PROTOCOL.md` → `STATE.md` → `SKILL.md` de la tarea. Solo entonces las secciones citadas de `THEORY.md` / `METRICS.md`. No releas todo el repo.
3. **Declara el chart y la normalización** en cada resultado: variable de control (T, β, K), total vs. por sitio (N = 100), unidades (J = 1, k_B = 1).
4. **Nada sin held-out.** Todo ajuste (sondas, regresiones de score, r_f) se evalúa en cadenas MC que no participaron en el entrenamiento de ese ajuste (split agrupado por cadena).
5. **Nunca Fisher empírica.** Las etiquetas del Fisher de parámetros se muestrean del modelo (`y ~ p_θ(y|x)`) o se usa la forma cerrada `p(1−p)∇f∇fᵀ`. Usar etiquetas de los datos es un error metodológico (Sun 2026: RelMAE hasta 56.8).
6. **Todo número lleva incertidumbre**: bootstrap por cadena (≥ 1000 remuestreos) y ≥ 5 semillas de red. Sin barra de error no entra al reporte.
7. **Reproducibilidad.** Cada corrida guarda `config.json` (semilla, hash de git, hiperparámetros, versión de librerías), `metrics.json` y las figuras. Sin eso la tarea no está terminada.
8. **No modifiques `legacy_baseline/`.** Es la referencia de la presentación. Extiende `ising_fisher/` o crea código en `experiments/mlp_L10/`.
9. **Cambios de definición = decisión registrada.** Si necesitas cambiar una fórmula de `THEORY.md`, añade una entrada en `STATE.md → Decisiones` con motivo y evidencia, y detente a confirmar.
10. **Los resultados negativos se registran.** Un efecto que no aparece, o que desaparece bajo un control, se documenta con la misma prioridad.

## Definición de "hecho" (checklist por tarea)

- [ ] Se cumplieron los criterios de aceptación de la skill (números concretos).
- [ ] Los tests relevantes pasan (`python -m pytest -v`) y se añadieron tests nuevos para código nuevo.
- [ ] `metrics.json` valida contra el esquema de abajo.
- [ ] Figuras guardadas con ejes rotulados, T_c(L) marcada, unidades y N en el título.
- [ ] `STATE.md` actualizado: estado, artefactos, hallazgos en una frase, próxima acción.
- [ ] Una entrada en la bitácora de relevo.

## Estructura de salida

```
experiments/mlp_L10/<ID>/
├── config.json
├── metrics.json
├── figs/*.png
├── notes.md          # hallazgos, anomalías, decisiones locales
└── logs/
```

### Esquema mínimo de `metrics.json`

```json
{
  "task_id": "MLP-06",
  "git_hash": "...",
  "L": 10, "N": 100, "J": 1.0, "kB": 1.0,
  "chart": "T", "normalization": "total|per_site",
  "seeds": [0,1,2,3,4],
  "T_grid": [0.1, 0.2],
  "layer": "hidden1|hidden2|logit|output|input",
  "metric": "r2_energy_given_T",
  "value": {"mean": [], "ci95_lo": [], "ci95_hi": []},
  "estimator": "ridge held-out, alpha by CV",
  "n_samples_per_T": 4000,
  "split": "chain-grouped"
}
```

## Errores que este proyecto ya identificó (no repetir)

| Error | Consecuencia | Qué hacer |
|---|---|---|
| Mezclar `F` total y `f = F/N` | Exponentes distintos en d | Declarar siempre la normalización |
| Comparar picos entre charts (T vs β) | Picos desplazados, "fallos" falsos | Comparar en un solo chart o usar invariantes |
| Interpretar autovalores de `F_θ` como exponentes | No son invariantes bajo reparametrización | Solo rango y razones invariantes |
| Contar `nθ ≥ ns` sin held-out | R² = 1 trivial | Held-out + ridge + nulos |
| Pico de `p(1−p)` en el umbral del clasificador | Está construido por la etiqueta | Control con umbral desplazado |
| Atribuir al modelo lo que es covarianza de la entrada | Confusión con `S(0)` | Baselines lineales y de features aleatorias |
| Información mutua por binning en redes deterministas | Artefacto del binning | Sondas variacionales; binning solo con análisis de sensibilidad |
