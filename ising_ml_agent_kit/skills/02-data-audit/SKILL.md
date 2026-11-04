---
name: ising-data-audit
description: Use for task MLP-02. Genera o carga datos Monte Carlo 10x10 y los audita contra el oráculo exacto (histograma de E, balance de signo, autocorrelación), con splits agrupados por cadena y grilla refinada cerca de Tc.
---

# Skill 02 · Datos y auditoría

Teoría: T2, T3. Métricas: A2, A3, A4, A7. Requiere MLP-01.

## Generación (`scripts/generate_mc.py` o extensión)
- Grilla: `T ∈ {0.1, 0.2, …, 4.0}` **más** `{2.00, 2.02, …, 2.60}`; sin duplicados. Documenta la grilla final.
- ≥ 20 cadenas independientes por T, termalización ≥ 2000 barridos (comprueba con τ_int), ≥ 200 muestras por cadena espaciadas ≥ 2·τ_int(E).
- Arranque: configuración aleatoria o recocido desde T alto. **Aleatoriza el signo global** de cada muestra con ±1 equiprobable (exacto por Z₂ a h = 0); guarda también `sign_original`.
- Guarda `chain_id`, `T`, `E`, `M`, semilla. Splits 70/15/15 **por `chain_id`**, idénticos para toda la campaña.

## Auditoría
1. **E vs exacto:** compara la distribución empírica de E con la exacta por T. Calcula `p(E|T)` exacta por la matriz de transferencia con variable auxiliar para E (polinomio en `x = e^{-4β}`), o por reponderación de histogramas anchos desde T cercana. KS y χ² por T; corrige múltiples tests (Holm). Compara también `⟨E⟩`, `Var E` con A1.
2. **Signo:** fracción de `M>0` antes y después de aleatorizar. Reporta cuántas muestras originales eran todo-↑.
3. **Correlaciones:** `G(r)` y `S(k)` por T (A4); comprueba que `S(0)` crece hacia T_c.
4. **Autocorrelación:** τ_int(E) por T y tamaño efectivo de muestra.
5. **Suficiencia:** verifica que `P*` del oráculo calculada sobre estos datos tiene accuracy coherente con `I(Y;X)`.

## Aceptación
- ≤ 10 % de las T con p < 0.01 (Holm), sin sesgo sistemático en `⟨E⟩`.
- Signo 0.5 ± 0.02 tras aleatorizar. HDF5 válido y reproducible (semilla en `config.json`).
- Si falla, no continúes: documenta la discrepancia en `STATE.md → Anomalías` y corrige el generador.

## Entregables
`data/L10_*.h5`, `experiments/mlp_L10/MLP-02/{config.json,metrics.json,figs/}`, tabla de τ_int, mapa de calor p-valor(T).
