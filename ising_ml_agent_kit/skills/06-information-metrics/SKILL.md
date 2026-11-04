---
name: ising-information-metrics
description: Use for task MLP-07. Estima informacion mutua I(T;h_l) e I(Y;h_l) con sondas variacionales, entropias gaussianas, rango efectivo, correlacion total e informacion por neurona en cada capa del MLP de Ising, comparando con el techo exacto I(T;X).
---

# Skill 06 · Información y entropía por capa (E1–E10, C1–C3)

Teoría: **T5, T6**. Requiere MLP-05 y MLP-06.

## Información mutua con cotas variacionales
1. Discretiza T en las K temperaturas de la grilla (o K′ bandas). Entrena una **sonda softmax** `q(T|h_l)` (lineal y MLP 1×64) con el split de sondas; `I(T;h_l) ≥ ln K − CE_test`.
2. Igual para `Y` con sonda binaria: `I(Y;h_l) ≥ H(Y) − CE_test`.
3. Aplica a `l ∈ {0,1,2,logit}` y a `p`. Reporta también la **fracción retenida** `I(T;h_l)/I(T;X)` y `I(Y;h_l)/I(Y;X)` con el techo exacto de MLP-01.
4. Mínimo 5 semillas × 5 particiones de cadenas; IC por bootstrap de cadenas.
5. Comprueba la cadena DPI `I(T;X) ≥ I(T;h_1) ≥ … ≥ I(T;ŷ)` (las cotas son inferiores: una violación pequeña es ruido, una grande es error).

## Entropías y geometría
- **E4:** entropía gaussiana regularizada `½ ln det(2πe(Σ_l + εI))` por T; elige ε por validación (el valor debe ser estable en un rango de 2 órdenes de magnitud; si no, no reportes).
- **C3/E5:** rango efectivo `exp(H(λ̃))`, razón de participación y correlación total gaussiana por capa y T.
- **C1/C2:** fracción activa y neuronas muertas por capa.
- **E6:** entropía de patrones ReLU on/off solo en capas ≤ 20 unidades (plug-in + Miller–Madow); si la capa es más ancha, omite.
- **E7:** `I(a_j;T)` y `I(a_j;Y)` por neurona con KSG 1-D (k ∈ {3,5,10}, reporta sensibilidad) o histograma con barrido de bins.
- **E8/E9:** `I(T;ŷ)`, `I(T;p)` y `F_out(T)/F_TT` (= `e_p`).
- **E10:** `H(Y|x)` bayesiano vs. `H(p(x))` de la red.

## Prohibido
- Binning de activaciones de dimensión > 3 para estimar `I(X;h_l)` o `I(T;h_l)`.
- Interpretar caídas de `I(X;h_l)` binneado como "compresión" (Saxe 2018, verificar).

## Entregables
Curvas por capa de `I(T;h_l)`, `I(Y;h_l)`, fracciones retenidas, entropías gaussianas y rangos efectivos; mapa de información por neurona; tabla resumen por capa.

## Aceptación
- `I(T;h_l) ≤ I(T;X)` en todas las capas (con IC); DPI respetada.
- Estabilidad de entropías gaussianas frente a ε documentada.
- Ningún valor sin IC.

## Pregunta que responde
¿Qué parte de la información térmica de la configuración conserva cada capa, y cuánta de ella queda disponible para la etiqueta? Comparar con `φ_l` de MLP-06: `φ` mide información de Fisher local; `I` mide información global sobre T.
