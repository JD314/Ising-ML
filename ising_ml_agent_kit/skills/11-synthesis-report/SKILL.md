---
name: ising-synthesis-report
description: Use for the final task SYN. Consolida los resultados MLP-00 a MLP-11 en un informe con afirmaciones graduadas por nivel de evidencia, figuras principales, anomalías y una decision go/no-go hacia CNN y escalado en L.
---

# Skill 11 · Síntesis y decisión

Requiere todas las tareas anteriores en `DONE` o `DONE_NEGATIVE`.

## Contenido del informe (`experiments/mlp_L10/SYN/report.md`)
1. **Resumen en 10 líneas:** dónde y cuándo aprende el MLP, medido contra el oráculo.
2. **Tabla de hipótesis** H0-a … H11-b: `confirmada`, `refutada` o `no concluyente`, con la figura y el número que la sostienen.
3. **Afirmaciones graduadas:** cada afirmación con nivel de evidencia:
   - **A** sobrevive a controles de MLP-11, ≥ 5 semillas, IC claros;
   - **B** sobrevive a la mayoría de controles, IC moderados;
   - **C** sugerente, falta control o semillas.
4. **Figuras principales (máx. 8):** accuracy vs. oráculo; `φ_l(T)` lineal/no lineal; mapa de `e_j`; `I(T;h_l)` y fracción retenida; `r_f^{(l)}(T)`; perfil de `Q`; orden de aprendizaje; tabla de supervivencia.
5. **Anomalías abiertas y limitaciones:** un solo L, fronteras, número de semillas, estimadores que son cotas inferiores.
6. **Qué no se concluye:** exponentes críticos (T9), universalidad, comparaciones entre charts.

## Decisión go/no-go
- **Go a CNN (`SimpleCNN` circular) con la misma batería** si el oráculo coincide con los datos y las métricas D/E/F son estables entre semillas.
- **Go a escalado en L ∈ {8, 12, 16, 24, 32}** si al menos una afirmación de nivel A existe.
- **No-go y revisión teórica** si la DPI falla de forma sistemática o el oráculo no explica el comportamiento del clasificador.

## Cierre
Actualiza `STATE.md` (todas las filas), `AGENT_CONTEXT.md` (decisiones nuevas) y crea `NEXT_PHASE.md` con el plan de la fase siguiente reutilizando este kit (skills 05–10 son independientes del modelo).
