---
name: ising-mlp-training
description: Use for task MLP-04. Entrena el MLP clasificador de referencia sobre Ising 10x10 con 5 semillas, split por cadena y checkpoints en un calendario fijo para análisis posterior de dinámica.
---

# Skill 03 · Entrenamiento con checkpoints

Requiere MLP-02 (y MLP-00 para la arquitectura).

## Configuración
- Arquitectura: la de `SimpleMLP` del repo. Si no existe documentada: `100 → 128 → 64 → 1` (logit), ReLU, sesgos activados, init por defecto de PyTorch. Anótala en `STATE.md`.
- Pérdida: BCE con logits. Optimizador: Adam, `lr = 1e-3`, batch 256, sin weight decay, sin scheduler. Fija `ϵ_lr/|B|` entre experimentos de capacidad (K7).
- Parada: pérdida de validación con paciencia 10, máx. 200 épocas. **No** limites a 7 épocas.
- Semillas de red 0–4, independientes de las del MC. `torch.use_deterministic_algorithms(True)` donde sea posible; float32 en entrenamiento.
- Entrada: spins ±1 aplanados. Variante B (control, MLP-11): igual con datos sin aleatorizar el signo.

## Checkpoints
Guarda `ckpt/seed{s}/epoch{e}.pt` para `e ∈ {0, 1, 2, 3, 5, 8, 13, 21, 34, 55}` y el mejor y el final. Guarda además pasos intra-época `{1, 5, 20, 50, 200}` de la época 1 (la mayor parte de la dinámica ocurre ahí).

## Registro
`config.json` (semilla, hash git, lr, batch, arquitectura, versión de PyTorch), curvas de pérdida y accuracy de train/val/test, accuracy por T en val.

## Aceptación
- Pérdida de validación estable (< 1 % de variación en las últimas 5 épocas) en ≥ 4/5 semillas.
- Accuracy global de test ≥ la de la presentación (L = 10) o explicación documentada.
- Pérdida y accuracy **por T** guardadas para MLP-05.

## Trampas
- Evaluar en datos de cadenas vistas en entrenamiento (fuga). Verifica el split antes de entrenar.
- Comparar semillas con distinto orden de datos sin fijarlo: documenta el generador de barajado.
