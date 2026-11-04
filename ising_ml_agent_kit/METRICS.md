# Catálogo de métricas

Notación: `a_j^l` = activación de la neurona j de la capa l (post-ReLU salvo que se diga lo contrario); `h_l` = vector de la capa; `f` = logit; `p = σ(f)`; `E`, `M` de la configuración; "a T fijo" = condicionando sobre la temperatura de la grilla. **Unidad** indica sobre qué se mide: `IN` entrada · `NEU` neurona · `LAY` capa · `OUT` salida · `PAR` parámetros · `DAT` datos.

Prioridad: **P0** imprescindible · **P1** importante · **P2** exploratoria.

## A. Física exacta y datos (`DAT`)

| ID | Métrica | Estimador / fórmula | Pri | Control de calidad |
|---|---|---|---|---|
| A1 | `ln Z(T)`, `⟨E⟩`, `Var E`, `F_TT`, `C`, `T_peak` | Matriz de transferencia (THEORY T2) | P0 | Igual a enumeración L ≤ 4 (<1e-12) |
| A2 | Histograma de E de los datos vs exacto | `p(E\|T)` exacta por polinomio de transferencia o por reponderación; KS y χ² por T | P0 | p-valor no sistemáticamente bajo |
| A3 | Balance de signo de M | fracción `M>0` por T | P0 | ≈ 0.5 tras aleatorizar |
| A4 | Función de correlación `G(r)`, `S(k)` | promedio sobre sitios y cadenas | P1 | `G(r)` decae más lento cerca de T_c |
| A5 | `H(X\|T)` (entropía de Gibbs) | integración termodinámica de `C/T` | P1 | `S(∞) = N ln 2` |
| A6 | `I(T;X)`, `I(Y;X)` exactas | `ln K − E[H(T\|x)]` con posterior exacto | P0 | ≤ `ln K`, ≤ `H(Y)` |
| A7 | τ_int de E por T | autocorrelación por cadena | P1 | espaciado ≥ 2τ_int |

## B. Comportamiento (`OUT`)

| ID | Métrica | Pri |
|---|---|---|
| B1 | Accuracy(T), CE(T) de la red | P0 |
| B2 | Accuracy(T), CE(T) del **oráculo bayesiano** y brecha `ΔCE(T)=CE_red−CE_bayes` | P0 |
| B3 | Entropía predictiva `H(p(x))` media por T vs. entropía del posterior bayesiano | P0 |
| B4 | Calibración (ECE por T, curva de fiabilidad) | P1 |
| B5 | Logit vs `g*(E)` (logit óptimo, función de E): R² global y a T fijo; fracción de varianza del logit explicada por E, por `(E,|M|)` y por E⊥|M| | P0 |
| B6 | Simetría: `‖f(x) − f(−x)‖` medio por T | P0 |
| B7 | Sensibilidad térmica `∂_T E_T[p]` (diferencias finitas) y eficiencia `e_p(T)` | P0 |

## C. Geometría de la representación (`NEU`, `LAY`)

| ID | Métrica | Estimador | Pri |
|---|---|---|---|
| C1 | Media, varianza, fracción activa (ReLU > 0) por neurona y T | momentos | P0 |
| C2 | Neuronas muertas / siempre activas | frac. activa ∈ {0,1} sobre todo el conjunto | P0 |
| C3 | Rango efectivo `exp(H(λ̃))` y razón de participación `(Σλ)²/Σλ²` | espectro de `Cov(h_l)` por T | P0 |
| C4 | Dimensión intrínseca (TwoNN) por capa y T | vecinos más cercanos | P2 |
| C5 | Curvas de sintonía `μ_j(T)` y selectividad (pico, ancho) | medias por T | P1 |
| C6 | CKA entre capas, entre semillas y entre épocas | CKA lineal sobre set común | P1 |
| C7 | Error de equivarianza por capa `‖h_l(x) − h_l(−x)‖/‖h_l(x)‖` | paridad | P1 |

## D. Decodabilidad y Fisher por capa (`NEU`, `LAY`)  ← **núcleo del proyecto**

| ID | Métrica | Estimador | Pri |
|---|---|---|---|
| D1 | `φ_l^lin(T)` = R² held-out de `E ~ lineal(h_l)` a T fijo | ridge, α por CV, split por cadena | P0 |
| D2 | `φ_l^nonlin(T)` | sonda MLP (1×64) o medias condicionales; held-out | P0 |
| D3 | Brecha de accesibilidad `Δ_l = D2 − D1` | diferencia con IC bootstrap | P0 |
| D4 | Eficiencia por neurona `e_j(T) = ρ²(a_j, E \| T)` | correlación de Pearson a T fijo | P0 |
| D5 | Correlación parcial `ρ²(a_j, E \| \|M\|, T)` y `ρ²(a_j, \|M\| \| E, T)` | regresión parcial | P0 |
| D6 | Decodabilidad de observables: `E`, `\|M\|`, `M²`, energía local, tamaño del clúster mayor, `s_i s_j` por distancia | sondas lineal y no lineal por capa | P1 |
| D7 | Prueba de DPI: `φ_l^nonlin` no creciente en l (dentro de IC) | test emparejado | P0 |
| D8 | Lectura lineal de la entrada `φ_input^lin` (debe ser 0) | ridge sobre `s` | P0 |

## E. Información y entropía (`LAY`, `NEU`, `OUT`)

| ID | Métrica | Estimador | Pri |
|---|---|---|---|
| E1 | `I(T; h_l) ≥ ln K − CE_sonda` y fracción `I(T;h_l)/I(T;X)` | sonda softmax sobre bins de T, held-out | P0 |
| E2 | `I(Y; h_l)` análoga | sonda binaria | P0 |
| E3 | Información ganada por capa `ΔI_l = I(T;h_l) − I(T;h_{l−1})` | diferencia | P1 |
| E4 | Entropía gaussiana regularizada de `h_l` por T | `½ ln det(2πe(Σ+εI))`, ε por validación | P1 |
| E5 | Correlación total gaussiana `−½ ln det(corr(h_l))` | correlación regularizada | P1 |
| E6 | Entropía de patrones ReLU (capas ≤ 20 unidades) | plug-in + Miller–Madow | P2 |
| E7 | Información 1-D por neurona `I(a_j; T)`, `I(a_j; Y)` | KSG 1-D o histograma con barrido de bins | P1 |
| E8 | Información de salida `I(T; ŷ)` y `I(T; p)` | cota variacional | P0 |
| E9 | Fisher de salida `F_out(T) = (∂_T E_T[p])²/(Var_T(p))` aprox. gaussiana y razón a `F_TT` | = D4 aplicado a p | P0 |
| E10 | `H(Y\|x)` bayesiano vs `H(p(x))` de la red | promedio sobre x | P0 |

## F. Fisher en parámetros (`PAR`)

| ID | Métrica | Estimador | Pri |
|---|---|---|---|
| F1 | `r_f(T)` global | R² held-out de `s_T ~ ∇θ f`, ridge | P0 |
| F2 | `r_f^{(l)}(T)` por capa (bloque `θ_l`) | igual, restringido | P0 |
| F3 | Espectro de la Gram por capa `GGᵀ` (ns × ns), rango efectivo, razón al borde MP | float64, ns escaneado | P0 |
| F4 | Número de outliers sobre MP vs. ns y vs. T | escaneo ns ∈ {500, 1000, 2000, 4000} | P1 |
| F5 | Autosolapamiento de autovectores entre semillas (calibración BBP) | `\|⟨v_a,v_b⟩\|²` tras alineación | P1 |
| F6 | Traza `Tr F_θ^{(l)}` y reparto por capa | suma de autovalores | P1 |
| F7 | Hessiano de la pérdida: λ_max por Lanczos vs. `λ_max(F_θ)` | productos Hessiano-vector | P2 |
| F8 | Identidades (a)–(d) del repo reevaluadas en el MLP | tests existentes | P0 |

## G. Lado de la entrada (`IN`)

| ID | Métrica | Estimador | Pri |
|---|---|---|---|
| G1 | Saliencia media `E_x\|∂f/∂s_i\|` por sitio y T | autograd | P1 |
| G2 | Fourier 2-D de los pesos de la capa 1 (por neurona): espectro de potencia en `k = (k_x,k_y)` | FFT sobre la rejilla 10×10 | P0 |
| G3 | Sustituto cuadrático `f ≈ c + Σ_{ij} Q_ij s_i s_j`: R², estructura de `Q` por distancia (vecinos próximos vs. resto) y comparación con `J_ij` del hamiltoniano | regresión ridge sobre productos | P0 |
| G4 | Invarianza traslacional aprendida: variación de `Q` a lo largo de órbitas de traslación | norma relativa | P1 |
| G5 | Ablaciones de entrada: proyectar a modo `k=0` (solo M), a rasgos pares, a energía; accuracy y `ΔCE` | reentrenar y evaluar | P1 |
| G6 | Información por sitio y por par: `I(s_i; Y)`, `I(s_i s_j; Y)` | exacta o por conteo | P2 |

## H. Dinámica (`todas`)

Mismas métricas D, E, F, G calculadas en checkpoints `e ∈ {0, 1, 2, 3, 5, 8, 13, 21, 34, 55}` (más pasos intra-época 1–200 si hace falta):

| ID | Métrica | Pri |
|---|---|---|
| H1 | Tiempo de emergencia por capa `t_l`: primera época con `φ_l^lin(T_c) > 0.5·φ_final` | P0 |
| H2 | Orden de aprendizaje (¿entrada→salida o salida→entrada?) | P0 |
| H3 | Evolución de `Tr F_θ^{(l)}`, rango efectivo y outliers | P1 |
| H4 | Norma de pesos y movimiento relativo `‖θ_t−θ_0‖/‖θ_0‖` por capa | P1 |
| H5 | Regimen lazy vs. feature learning: cambio relativo del kernel `∇f∇fᵀ` | P2 |

## Reglas transversales

- Cada métrica se calcula **por T de la grilla** y se resume en una curva `métrica(T)` con IC.
- Toda métrica con ajuste usa **split por cadena**; el conjunto de sondas es independiente del de entrenamiento de la red.
- Se reporta siempre el valor en **red inicializada** (época 0) y en **red con etiquetas barajadas** como línea base.
- Para cada métrica de capa, mostrar la **entrada** (`l=0`) y la **salida** como extremos de referencia.
