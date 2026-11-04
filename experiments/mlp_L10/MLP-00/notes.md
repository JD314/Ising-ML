# MLP-00: Auditoría del Repositorio y Entorno

**Fecha:** 2026-11-03  
**Commit:** `db9fad1b5fd153a5534ec38f661edc3c296746c4`  
**Entorno:** Python 3.12.13, PyTorch 2.6.0+cu124 (CUDA disponible: True, device: NVIDIA GeForce RTX 3070 Ti Laptop GPU), pytest 9.1.1.

---

## 1. Tabla de Auditoría de Código y Configuración

| Pregunta | Dónde mirar | Evidencia y Hallazgo |
|---|---|---|
| **¿BC periódicas u abiertas? ¿vecinos con `% L`?** | `ising_fisher/mc.py` | **Periódicas (PBC)** mediante `% L`. En [`local_delta_E`](file:///home/jd/Projects/Ising-ML/ising_fisher/mc.py#L35-L45) (líneas 39-44) los vecinos se indexan como `(i + 1) % L`, `(i - 1) % L`, `(j + 1) % L`, `(j - 1) % L`. Igual en [`interaction_energy`](file:///home/jd/Projects/Ising-ML/ising_fisher/mc.py#L48-L58) (líneas 56-57). |
| **¿Se aleatoriza el signo global? ¿estado inicial?** | `ising_fisher/mc.py`, `scripts/generate_mc.py` | En [`mc.py:85-97`](file:///home/jd/Projects/Ising-ML/ising_fisher/mc.py#L85-L97), `_run_chain` soporta `init` in `{"random": 0, "up": 1, "down": -1}`. Por defecto en [`mc.py:116`](file:///home/jd/Projects/Ising-ML/ising_fisher/mc.py#L116) y [`mc.py:141`](file:///home/jd/Projects/Ising-ML/ising_fisher/mc.py#L141) es `init="random"` (hot start, `p=0.5`). **No se aleatoriza el signo global $\mathbf{S} \to -\mathbf{S}$ por muestra**. En `dataset.py:14` explícitamente se declara: *"No S -> -S augmentation is stored: the odd (magnetisation) channel must stay informative"*. El código legacy en cambio usaba `invert=True` provocando data leakage (verificado en `CONTEXT_ISING_ML.md:212-215`). |
| **¿Esquema de recocido, termalización, espaciado?** | `ising_fisher/mc.py` | **Sin recocido cruzado** ([`mc.py:19-20`](file:///home/jd/Projects/Ising-ML/ising_fisher/mc.py#L19-L20)): cada cadena corre a $(T, h)$ fijo independiente. **Unidades:** El tiempo se mide en **sweeps** ($L^2$ intentos de flip de espín individual, [`mc.py:17-18`](file:///home/jd/Projects/Ising-ML/ising_fisher/mc.py#L17-L18), [`mc.py:73-82`](file:///home/jd/Projects/Ising-ML/ising_fisher/mc.py#L73-L82)). Por defecto: `n_burn=1000` sweeps, `thin=10` sweeps entre muestras ([`mc.py:116`](file:///home/jd/Projects/Ising-ML/ising_fisher/mc.py#L116)). |
| **¿Formato HDF5 y split por cadena?** | `ising_fisher/dataset.py` | Formato jerárquico HDF5 en [`dataset.py:4-12`](file:///home/jd/Projects/Ising-ML/ising_fisher/dataset.py#L4-L12): `/L{L:03d}/T{T:.6f}/h{h:+.6f}/` conteniendo datasets `spins` (n, L, L, int8 gzip), `E_J` (float64), `M` (int32), `chain_id` (int32). Split por cadena implementado en [`dataset.py:79-101`](file:///home/jd/Projects/Ising-ML/ising_fisher/dataset.py#L79-L101) (`split_by_chain`), garantizando que cadenas enteras van a train o test. |
| **¿Arquitectura exacta de `SimpleMLP` (capas, activación, sesgos, init)?** | `ising_fisher/models/classifier.py` | [`SimpleMLP`](file:///home/jd/Projects/Ising-ML/ising_fisher/models/classifier.py#L21-L36): `nn.Sequential(nn.Flatten(), nn.Linear(L*L, 16), nn.ReLU(), nn.Linear(16, 1))`. Tiene 1 capa oculta de 16 neuronas, activación ReLU, salida 1 logit lineal (con sesgos por defecto de PyTorch, inicialización Kaiming uniforme por defecto de `nn.Linear`). Total parámetros para $L=10$: $(100 \times 16 + 16) + (16 \times 1 + 1) = 1616 + 17 = 1633$ parámetros. |
| **¿Cómo se calcula `r_f` (features, ridge, held-out)?** | `ising_fisher/fisher.py` | En [`compute_classifier_rf`](file:///home/jd/Projects/Ising-ML/ising_fisher/fisher.py#L492-L505) y [`rf_from_features`](file:///home/jd/Projects/Ising-ML/ising_fisher/fisher.py#L474-L490): Se calculan gradientes del logit por muestra $\nabla_\theta f(x)$ con `torch.func.vmap(grad_and_value)` ([`fisher.py:436-455`](file:///home/jd/Projects/Ising-ML/ising_fisher/fisher.py#L436-L455)). Se calcula el score térmico físico $s_T = (E_J - \langle E_J \rangle) / T^2$ ([`fisher.py:464-472`](file:///home/jd/Projects/Ising-ML/ising_fisher/fisher.py#L464-L472)). Se realiza WLS con SVD truncado (`rcond=1e-6` o `clf_rcond=1e-3`) en [`_wls_fit`](file:///home/jd/Projects/Ising-ML/ising_fisher/fisher.py#L315-L337). Con `split=True`, [`_heldout`](file:///home/jd/Projects/Ising-ML/ising_fisher/fisher.py#L366-L376) calcula $R^2$ fuera de muestra 2-fold (A $\to$ B y B $\to$ A) respetando grupos de cadena. |
| **¿Se usa Fisher con etiquetas del modelo?** | `ising_fisher/fisher.py` | Sí. En [`classifier_label_scores`](file:///home/jd/Projects/Ising-ML/ising_fisher/fisher.py#L458-L462): $u_\theta = (y - \sigma(f)) \nabla_\theta f(x)$. En test estructural [`test_identity_b_classifier_sampled_labels`](file:///home/jd/Projects/Ising-ML/tests/test_fisher.py#L202-L241) se verifica que con $y \sim \text{Bernoulli}(\sigma(f(x)))$, $r_T \approx 0$ (no hay fuga de información física espuria). En [`test_identity_b_classifier_analytic_label_expectation`](file:///home/jd/Projects/Ising-ML/tests/test_fisher.py#L242-L260) se evalúa exactamente con pesos $\sigma$ y $1-\sigma$. |
| **¿Qué cubren los 44 tests?** | `tests/` | 44 tests unitarios y de integración: <br>• `test_fisher.py` (18 tests): consistencia de ordenamiento de microestados, cálculo de scores generativos autograd vs vmap, descarte de parámetros inactivos/máscaras, identidad condicional $T u_T = W_T \cdot u_W$, equivalencia SVD vs pinv, held-out Schur, las 4 identidades estructurales de CONTEXT §6.5 ((a) invarianza a reparametrización, (b) classifier con labels muestreados da $r_T=0$, (c) paridad $\mathbb{Z}_2$ da $r_h=0$ y $F_{Th}=0$, (d) $r_f \in [0, 1]$ e invarianza lineal de $r_f$), y score térmico con campo. <br>• `test_mc.py` (4 tests): $\Delta E$ local vs diferencia total de Hamiltoniano para $J=1.0, 0.7$; kernels de $E$ y $M$; convergencia de $F_{TT}, F_{hh}$ muestral a $L=4$ vs enumeración exacta en $T_c$. <br>• `test_models.py` (22 tests): normalización de `SimpleGenCNN` a $L=4$, equivarianza exacta $\mathbb{Z}_2$, ruptura en modelo no equivariante, override flag, enmascaramiento autorregresivo estricto, canales de condición libres, conteo de parámetros, consistencia sampling vs log_prob, separación de fases con `SimpleMLP` y `SimpleCNN`, reducción de NLL hacia entropía teórica. |
| **¿Qué dice la auditoría de código?** | `CONTEXT_ISING_ML.md` | Documenta las correcciones a la base legacy: <br>1. Bug de signo en término de campo $\Delta E_{\text{legacy}} = 2J S \sum S_{\text{nn}} - 2H S$ corregido a $+2H S$ en `ising_fisher/mc.py`. <br>2. Confusión pasos vs sweeps: legacy usaba pasos individuales ($5000$ pasos = 50 sweeps en $L=10$, pero sólo 5.6 sweeps en $L=30$), corregido a sweeps explícitos en `mc.py`. <br>3. `invert=True` legacy introducía data leakage antes del split y mataba el canal impar ($M$); corregido en `dataset.py` prohibiendo inversión global por defecto y aplicando splits por cadenas independientes. <br>4. Especificación rigurosa de invariantes de Fisher y Schur en §6. |

---

## 2. Ejecución de Pruebas y Humo

1. **Pytest Suite (`python -m pytest -v`):**
   - **Resultado:** 44 passed, 1 warning (array de solo lectura en test de observables).
   - **Tiempo:** 51.48s.
2. **Smoke Test (`scripts/run_first_exploration.py`):**
   - Ejecutado con éxito: genera datos MC, entrena `SimpleMLP`, `SimpleCNN` y `SimpleGenCNN`, calcula métricas de Fisher/Schur ($r_f$, $r_T$, $F_{TT}$) y guarda figuras en `img_results/first_exploration.png` y `img_results/first_exploration.json`.

---

## 3. Estado de Riesgos R1 y R2

- **R1 (Condiciones de frontera y equilibrio): DESCARTADO / CONTROLADO.**
  - `mc.py` implementa periódicas (`% L`) exactas, verifica kernels con tests unitarios contra enumeración exacta (`test_local_delta_E_matches_total_energy_difference`, `test_fisher_matches_exact_enumeration_L4_Tc_h0`), y mide tiempo en sweeps reales.
- **R2 (Datos con signo fijo / rompimiento artificial de $\mathbb{Z}_2$): CONFIRMADO EN LEGACY, RESUELTO EN `ising_fisher`.**
  - El código de `legacy_baseline` inicializaba en `all up` y aplicaba `invert=True` mezclando train/test.
  - En `ising_fisher`, `mc.py` usa `init="random"` (hot start, simetría Z2 no sesgada) y `dataset.py` split por cadenas sin mezcla de duplicados invertidos. En MLP-04/MLP-11 se entrenará con control explícito.

---

## 4. Decisiones de Arquitectura para la Campaña

1. **Arquitectura SimpleMLP:**
   - La arquitectura existente en `ising_fisher/models/classifier.py` es `100 -> 16 -> 1` con ReLU (1633 parámetros).
   - La plantilla de `STATE.md` sugería como alternativa `100 -> 128 -> 64 -> 1`.
   - Se mantiene `SimpleMLP` (100 -> 16 -> 1, P=1633) como el clasificador lightweight de referencia, o se define el estándar según el plan de MLP-04.
