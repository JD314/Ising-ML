# Teoría de referencia (apego obligatorio)

Todo lo que se mide debe poder rastrearse a una de estas definiciones. Las referencias `[Plan §x]` son las de la charla de revisión 2026 (SYNTHESIS); las marcadas **(verificar)** no están en las fichas del proyecto y deben contrastarse antes de citarlas.

---

## T1 · Fisher física a h = 0

Para `q(x;T) = e^{−E(x)/T}/Z(T)`, con `E = −Σ_{⟨ij⟩} s_i s_j`:

```
score        u_T(x) = (E(x) − ⟨E⟩_T) / T²
F_TT         = Var_T(E) / T⁴ = C / T²          (C = Var(E)/T²)
f_TT         = F_TT / N                         (por sitio)
chart β      F_ββ = Var(E)                      (¡pico en otra T!)
Cramér–Rao   δT ≥ 1 / √(n · F_TT)               (n configuraciones independientes)
```

- `F_Th = 0` a h = 0 (simetría Z₂).
- `F_TT` es un **tensor**: cambia con el chart; los exponentes críticos no, pero la posición del pico sí. [Plan §1.3]
- Con N = 100, `F_TT` es total por configuración. No compares con valores por sitio sin dividir por N.

## T2 · Oráculo exacto en L = 10 (toro periódico)

`ln Z(β)` se obtiene de la matriz de transferencia fila a fila (1024 × 1024):

```
V[s, s'] = exp( β·[ ½ Σ_i s_i s_{i+1} + ½ Σ_i s'_i s'_{i+1} + Σ_i s_i s'_i ] ),   i mod L
Z(β)     = Tr( V^L ) = Σ_k λ_k^L
```

Consecuencias, todas exactas:

- `ln p(x|T) = −E(x)/T − ln Z(T)` para **cada** configuración muestreada.
- `⟨E⟩ = −∂_β ln Z`, `Var(E) = ∂²_β ln Z` → `F_TT`, `C`, `T_peak(L=10)`.
- **Posterior de T** con prior uniforme sobre la grilla `{T_k}` y igual número de muestras por T:
  `P*(T_k | x) ∝ exp(−E(x)/T_k − ln Z(T_k))`.
- **Clasificador bayesiano óptimo**: `P*(y=1|x) = Σ_{T_k > T_c} P*(T_k|x)`. Depende de x solo a través de E.
- **Información mutua exacta (estimada por MC sobre x)**:
  `I(T;X) = ln K − E_x[ H(T|x) ]`,  `I(Y;X) = H(Y) − E_x[ H(Y|x) ]`.
  Cotas: `I(T;X) ≤ ln K`, `I(Y;X) ≤ H(Y) ≤ ln 2`.
- **Entropía de Gibbs de los datos**: `H(X|T) = S(T)/k_B`, con `S(T) = N ln 2 − ∫_T^∞ C(T')/T' dT'`.
- Validación obligatoria: reproducir enumeración exacta de L ≤ 4 del repo (`observables.py`) a precisión de máquina.

Esto exige que los datos sean equilibrio exacto del mismo hamiltoniano y **las mismas condiciones de frontera**. Si no, el oráculo no aplica (R1 en `AGENT_CONTEXT.md`).

## T3 · Simetría Z₂ y suficiencia

- `q(−x|T) = q(x|T)` a h = 0 ⇒ `Cov(s_i, E) = 0` y, en general, **toda función impar de x tiene covarianza nula con E**.
- Corolario: una lectura **lineal** de la entrada (`wᵀs`) no tiene información de Fisher sobre T: `R²(E | T; lineal en s) = 0` exactamente. Rasgos pares (`s_i s_j`, `|M|`, `M²`) sí la tienen.
- E es estadístico **suficiente** de T a h = 0 ⇒ el óptimo bayesiano de cualquier tarea sobre T es función de E.
- |M| **no** es suficiente, pero está correlacionado con E a T fijo. Una red puede usarlo como atajo. Se mide con correlación parcial (METRICS D5).

## T4 · Identidad de Cauchy–Schwarz: eficiencia termométrica

Para cualquier estadístico `g(x)` (neurona, capa, logit, probabilidad):

```
∂_T E_T[g] = Cov_T(g, u_T)                       (porque E[u_T] = 0)
(∂_T E_T[g])² ≤ Var_T(g) · F_TT
e_g(T) ≡ (∂_T E_T[g])² / (Var_T(g) F_TT) = ρ²_T(g, E)   a h = 0
```

`e_g ∈ [0,1]`, con igualdad si y solo si `g` es afín en E. Es la versión exacta de que los indicadores ML están acotados por `√F` (Arnold 2023). [Plan §1.5, L1]

## T5 · Fracción de Fisher por capa (la métrica central)

Sea `h_l = g_l(x)` la activación de la capa l (determinista). La información de Fisher sobre T contenida en `h_l` es exactamente

```
F_{h_l}(T) = Var_T( E[ u_T(x) | h_l ] )
φ_l(T)     = F_{h_l}(T) / F_TT = Var( E[E | h_l] | T ) / Var(E | T)  ∈ [0, 1]
```

(el score de un estadístico es la esperanza condicional del score completo). Propiedades:

1. **DPI (exacta):** `1 = φ_input ≥ φ_1 ≥ φ_2 ≥ … ≥ φ_logit`. Como `p = σ(logit)` es inyectiva, `φ_p = φ_logit`.
2. **Versión lineal:** `φ_l^lin(T) = R²(u_T ~ lineal en h_l)` (con constante), que coincide con `∂μᵀΣ⁻¹∂μ / F_TT`. Es una cota inferior de `φ_l` y **no** tiene que ser monótona en la profundidad.
3. **Brecha de accesibilidad** `Δ_l(T) = φ_l^nonlin − φ_l^lin ≥ 0`: información presente pero no linealmente legible. La capa donde `Δ` colapsa y `φ^lin` sube es donde la red **hace accesible** la energía. Esa es la operacionalización de "dónde aprende".
4. Estimación: `φ^nonlin` con sondas flexibles (MLP pequeño, kNN, o medias condicionales por bins sobre una proyección) evaluadas en held-out; siempre son cotas inferiores. **Test de cordura:** si `φ_l^est > 1` o viola DPI por más que el intervalo de confianza, hay un error de estimación o de datos.
5. Nota: para T lejos del umbral la red no necesita conservar información térmica (solo el signo de la etiqueta). Es esperable que `φ_logit(T)` sea pequeño allí y grande cerca del umbral. Eso no es un fallo; se contrasta con el control de umbral desplazado (C8, R5).

## T6 · Información mutua en capas

- Con T discreta (K valores): `I(T; h_l) ≥ ln K − CE_sonda(T | h_l)` (cota variacional, held-out). Análogo para `I(Y; h_l)`.
- DPI: `I(T;X) ≥ I(T;h_1) ≥ … ≥ I(T;ŷ)`. Con `I(T;X)` exacto de T2 hay un techo y se puede reportar la **fracción de información retenida** `I(T;h_l)/I(T;X)`.
- **No usar binning de activaciones de dimensión alta** para estimar `I(X;h_l)`: en redes deterministas continuas es infinita o un artefacto de la discretización (Saxe et al. 2018; Goldfeld et al. 2019, **verificar**). Si se reporta binning, es solo en 1-D (una neurona) con barrido del ancho de bin.
- Entropías de activación: entropía gaussiana regularizada `½ ln det(2πe Σ_l)`, rango efectivo `exp(H(λ̃))`, correlación total gaussiana, entropía de patrones ReLU on/off en capas ≤ 20 unidades (plug-in + Miller–Madow).

## T7 · Fisher en parámetros del clasificador

Para un logit `f_θ(x)` y `p = σ(f)`:

```
u_θ(x)   = ∇_θ log p_θ(y|x),   y ~ p_θ(·|x)           (etiquetas del MODELO)
F_θ      = E_x[ p(1−p) ∇_θ f ∇_θ fᵀ ] = Jᵀ I(z) J      (I = p(1−p), pullback del logit)
Gram     G ∈ R^{ns×nθ}, filas √(p(1−p)) ∇_θ f(x_a)ᵀ/√ns ;  F̂_θ = GᵀG ; nonzero spec igual que GGᵀ
```

- **Identidades verificadas en el repo** (`tests/test_fisher.py`): (a) `r_c` invariante bajo `θ→Aθ` a <1e-10; (b) para clasificadores `G_T ≡ 0`, `r_T ≡ 0` con etiquetas del modelo; (c) `r_h ≡ 0` con equivarianza Z₂ a h = 0; (d) `r_f(T)`.
- **`r_f(T)`** = R² held-out de regresar `s_T = (E − ⟨E⟩)/T²` sobre `∇_θ f(x)`. Mide si el **espacio tangente** de la red contiene la energía; no es lo mismo que `φ_l` (que mide la información que *transmiten las activaciones*). Se reportan ambos y se estudia su relación.
- **Por capa:** restringir `∇_θ` al bloque `θ_l` (pesos y sesgo de la capa l). Es el análogo del flujo de información de Fisher por capas (Weimar 2025). [Plan E4]
- **Invariancia:** `r_f` y rangos son invariantes bajo reparametrización; los **autovalores y sus exponentes no lo son** (C4). Solo se interpretan rango efectivo, razones y outliers frente al borde MP.
- **Aspecto `nθ/ns` grande** (MLP de referencia: nθ ≈ 2·10⁴, ns ≈ 4·10³ por T): sin held-out, `R² = 1` trivial. Obligatorio: held-out por cadena, ridge con α por validación cruzada, y nulos (red en inicialización, pesos barajados, features aleatorias). [Plan §1.11, §1.29–1.32]
- **Outliers:** un autovalor solo "existe" si supera el borde de Marchenko–Pastur calibrado con escaneo en ns y con autosolapamiento entre semillas (K8, K12).
- **Confusión de entrada:** parte del espectro del primer bloque refleja `⟨ssᵀ⟩` (covarianza de la entrada, ∝ S(0)), no la física (C5). El baseline de features aleatorias lo descuenta.

## T8 · Lo que es y no es invariante

| Invariante | No invariante |
|---|---|
| `φ_l`, `I(T;h_l)`, `e_g` (son funcionales de distribuciones sobre x) | Autovalores de `F_θ`, pesos individuales, normas |
| `r_f`, rango de `F_θ`, razones de autovalores generalizados | Posición del pico en un chart distinto de T |
| Exponentes bajo cambios de chart regulares y L-independientes | Exponentes de autovalores bajo reparametrizaciones de pesos |

## T9 · Alcance finito en L = 10

- Un solo tamaño **no permite extraer exponentes críticos**. Esta fase mide *estructura* (dónde, cuándo, con qué rasgos), no `ν`, `y_t`, `y_h`.
- Para cualquier afirmación de escalado: ≥ 4 tamaños, ajuste MLE/KS y calibración sobre un exponente conocido (K15). Reservado para fases posteriores (L ∈ {8, 12, 16, 24, 32}).
- Usar `T_peak(10)` del máximo de `F_TT` exacto además de `T_c`; en L = 10 difieren.
