---
name: ising-input-side-analysis
description: Use for task MLP-09. Analiza que rasgos de la entrada usa el MLP de Ising 10x10 mediante Fourier de los pesos de la primera capa, un sustituto cuadratico del logit, saliencia por sitio y ablaciones de entrada, comparandolo con el hamiltoniano.
---

# Skill 08 · Lado de la entrada (G1–G6)

Teoría: **T3**. Requiere MLP-04.

## Cálculos
1. **G2 Fourier de pesos.** Reordena cada fila de `W1` (100) como rejilla 10×10; FFT 2-D; espectro de potencia por neurona. Resume: fracción de potencia en `k=0` (detector de M, impar → solo útil vía ReLU), en `|k|` pequeño y en `k` alto. Comparación entre semillas y con la inicialización.
2. **G3 Sustituto cuadrático.** Ajusta `f(x) ≈ c + Σ_{i≤j} Q_ij s_i s_j` por ridge sobre un conjunto de test; reporta R² held-out global y a T fijo. Descompón `Q` por distancia toroidal `r`: potencia en `r=1` (vecinos próximos, estructura del hamiltoniano), `r>1` y diagonal. Compara el perfil con `J_ij` (1 en vecinos, 0 si no) mediante correlación.
3. **G4 Invarianza traslacional.** Para pares con la misma separación, mide la dispersión relativa de `Q_ij`; si es pequeña, la red aprendió la simetría pese a no imponerla.
4. **G1 Saliencia.** `E_x |∂f/∂s_i|` por sitio y T; esperado homogéneo si hay invarianza.
5. **G5 Ablaciones.** Reentrena (3 semillas) con entradas: (a) solo modo `k=0` (M), (b) solo rasgos pares `s_i s_j` de vecinos, (c) solo la energía E. Reporta accuracy(T) y `ΔCE` vs. oráculo.
6. **G6 (opcional).** `I(s_i;Y)` e `I(s_is_j;Y)` por conteo exacto.

## Entregables
Mapas 10×10 de saliencia y de potencia de pesos; espectro `k`; perfil de `Q` por distancia; tabla de ablaciones.

## Aceptación
R² del sustituto cuadrático con IC; resumen explícito del reparto: "% de potencia de `Q` en vecinos próximos" y "% en modo `k=0`".

## Lectura
Si `Q` se concentra en vecinos próximos, la red aprende (aprox.) la energía. Si domina `k=0`/|M|, usa el atajo de la magnetización. El reparto, no la anécdota, es el resultado.
