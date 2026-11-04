---
name: ising-fisher-parameter-space
description: Use for task MLP-08. Calcula r_f(T) global y por capa, el espectro de la matriz de Gram del Fisher de parámetros por capa, rango efectivo y outliers frente al borde Marchenko-Pastur, con etiquetas muestreadas del modelo y nulos obligatorios.
---

# Skill 07 · Fisher en el espacio de parámetros (F1–F8)

Teoría: **T7, T8**. Requiere MLP-04 y MLP-06. Reutiliza `ising_fisher/fisher.py`; no lo reescribas sin auditarlo (MLP-00).

## Reglas duras
- Etiquetas del **modelo** o forma cerrada `p(1−p)∇f∇fᵀ`. **Nunca** Fisher empírica (D4).
- `float64`. Gram dual de `ns × ns`; nunca construyas `nθ × nθ`.
- Held-out por cadena para `r_f`; ridge con α por CV interna.

## Cálculos
1. **F8:** corre los tests (a)–(d) del repo sobre este MLP. Si fallan, detente.
2. **F1 `r_f(T)`:** regresión de `s_T = (E−⟨E⟩)/T²` sobre `∇_θ f(x)` por T. Reporta también `r_f` sobre todos los T mezclados.
3. **F2 `r_f^{(l)}(T)`:** igual con el bloque `θ_l` (pesos y sesgo de la capa l). Compara con `φ_l` de MLP-06: ¿coinciden las capas donde ambos suben?
4. **F3:** autovalores de `GGᵀ` por capa y T; rango efectivo, traza (F6) y razón al borde MP.
5. **F4:** número de outliers sobre MP escaneando `ns ∈ {500, 1000, 2000, 4000}`; un autovalor solo cuenta si su posición es estable al subir ns.
6. **F5:** autosolapamiento de autovectores entre semillas (calibración BBP); solo los outliers con solapamiento alto cuentan como estructura.
7. **F7 (opcional):** λ_max del Hessiano por Lanczos vs. λ_max(F_θ).

## Nulos obligatorios
| Nulo | Qué descarta |
|---|---|
| Red en inicialización | Que `r_f` sea geometría de la arquitectura |
| Pesos barajados por capa | Que dependa del valor de los pesos |
| Features aleatorias (capa oculta congelada) | Confusión con la covarianza de entrada `⟨ssᵀ⟩` (C5) |
| Etiquetas barajadas | Que dependa de la tarea |

## Entregables
Curvas `r_f(T)`, `r_f^{(l)}(T)` con bandas; espectros por capa; tabla de outliers vs. ns; reparto de la traza por capa.

## Aceptación
Tests pasan; `r_f(entrenado)` supera a todos los nulos cerca de T_c con IC no solapados (o se registra `DONE_NEGATIVE`); outliers contados solo si estables en ns.

## Interpretación permitida
Rangos, razones y outliers calibrados. **No** se interpretan autovalores absolutos ni se ajustan exponentes (T8, T9).
