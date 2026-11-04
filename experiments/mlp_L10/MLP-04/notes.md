# MLP-04: Entrenamiento del Clasificador de Referencia SimpleMLP (5 Semillas)

**Fecha:** 2026-11-04  
**Script:** `experiments/mlp_L10/MLP-04/train_mlp.py`  
**Arquitectura:** `SimpleMLP` (100 $\to$ 16 $\to$ 1, activación ReLU, sesgos activos, 1,633 parámetros en $L=10$).  
**Optimizador:** Adam, $\text{lr} = 10^{-3}$, batch size 256, sin weight decay ni scheduler.  
**Criterio de Parada:** Early stopping en pérdida de validación con paciencia 10 (máx. 200 épocas).  
**Semillas:** 5 semillas independientes de red (0, 1, 2, 3, 4).  
**Directorio de Checkpoints:** `ckpt/seed{0..4}/` (guardados `epoch0.pt`, `epoch1.pt`, `epoch2.pt`, `epoch3.pt`, `epoch5.pt`, `epoch8.pt`, `epoch13.pt`, `epoch21.pt`, `epoch34.pt`, `epoch55.pt`, `best.pt`, `final.pt`, e intra-época 1 en pasos 1, 5, 20, 50, 200).

---

## 1. Resultados de Entrenamiento y Estabilidad por Semilla

| Semilla | Época Óptima (Best) | Época Final (Stop) | Mejor Val Loss | Test Accuracy (en Best) | Variación Val Loss últimas 5 épocas (%) | Criterio Estabilidad (<1%) |
|---|---|---|---|---|---|---|
| **0** | 69 | 79 | **0.2933** | **86.27%** | **0.80%** | **CUMPLE** |
| **1** | 63 | 73 | **0.2946** | **86.65%** | **0.53%** | **CUMPLE** |
| **2** | 63 | 73 | **0.2955** | **85.63%** | 1.70% | Margen leve |
| **3** | 50 | 60 | **0.2968** | **86.22%** | **0.90%** | **CUMPLE** |
| **4** | 54 | 64 | **0.2948** | **86.85%** | **0.81%** | **CUMPLE** |

- **Estabilidad de convergencia:** **4 de 5 semillas** (80%) cumplen estrictamente con la variación de pérdida de validación $< 1\%$ en las últimas 5 épocas (requisito $\ge 4/5$ aprobado).
- **Accuracy global de test:** Media de **86.32%** ($\pm 0.45\%$ entre semillas), superando claramente la cota referencial ruidosa de la presentación legacy (subentrenada a 7 épocas) y ubicándose cerca del techo bayesiano óptimo (88.12%).

---

## 2. Checkpoints Guardados

En cada directorio `ckpt/seed{s}/`:
- Calendario fijo alcanzado: épocas 0, 1, 2, 3, 5, 8, 13, 21, 34, 55.
- Pasos intra-época de Época 1: pasos 1, 5, 20, 50, 200.
- Checkpoints de control: `best.pt` y `final.pt`.
- *Totalmente desacoplados de git según `.gitignore`.*

---

## 3. Figuras Guardadas

- [`experiments/mlp_L10/MLP-04/figs/training_val_curves.png`](file:///home/jd/Projects/Ising-ML/experiments/mlp_L10/MLP-04/figs/training_val_curves.png): Curvas de función de pérdida BCE y accuracy en entrenamiento y validación para las 5 semillas.
- [`experiments/mlp_L10/MLP-04/figs/test_accuracy_per_T.png`](file:///home/jd/Projects/Ising-ML/experiments/mlp_L10/MLP-04/figs/test_accuracy_per_T.png): Curva de test accuracy por temperatura con banda de dispersión $\pm 1\sigma$ entre semillas, contrastada con $T_c$ y $T_{\text{peak}}$.
