"""
Training loops for Stage 1 models.

* `train_classifier`: binary classifiers (SimpleMLP / SimpleCNN) with BCEWithLogitsLoss.
* `train_generator`: SimpleGenCNN by forward KL, i.e. maximum likelihood on MC data,
  minimising -mean(log_prob(x, T, h)).

Data are plain tensors / arrays: spins X of shape (N, L, L) in {-1, +1}.
Split by independent configurations / chains *before* calling these (see
`ising_fisher.dataset.split_by_chain`); no S -> -S augmentation is applied here.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _t(a, dtype=torch.float32) -> torch.Tensor:
    return torch.as_tensor(np.asarray(a) if not torch.is_tensor(a) else a).to(dtype)


def _broadcast(v, n: int) -> torch.Tensor:
    v = _t(v).reshape(-1)
    return v.expand(n).clone() if v.numel() == 1 else v


def _loader(tensors, batch_size, shuffle, seed) -> DataLoader:
    g = torch.Generator().manual_seed(seed)
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle, generator=g)


def train_classifier(model: nn.Module, X, y, *, X_val=None, y_val=None, epochs: int = 20,
                     batch_size: int = 128, lr: float = 1e-3, weight_decay: float = 0.0,
                     device: Optional[str] = None, seed: int = 0, verbose: bool = False) -> dict:
    """
    Train a binary classifier with BCEWithLogitsLoss.

    X: (N, L, L) spins in {-1, +1}; y: (N,) labels in {0, 1}.
    Returns a history dict with per-epoch 'train_loss', 'train_acc' and (if validation
    data are given) 'val_loss', 'val_acc'.
    """
    device = torch.device(device or "cpu")
    torch.manual_seed(seed)
    model.to(device)
    X, y = _t(X), _t(y)
    loader = _loader((X, y), batch_size, True, seed)
    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    hist = {"train_loss": [], "train_acc": []}
    if X_val is not None:
        hist.update(val_loss=[], val_acc=[])

    for epoch in range(epochs):
        model.train()
        tot_loss = tot_correct = n = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logit = model(xb).squeeze(-1)
            loss = loss_fn(logit, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_loss += loss.item() * len(yb)
            tot_correct += ((logit > 0).float() == yb).sum().item()
            n += len(yb)
        hist["train_loss"].append(tot_loss / n)
        hist["train_acc"].append(tot_correct / n)
        if X_val is not None:
            vl, va = evaluate_classifier(model, X_val, y_val, device=device)
            hist["val_loss"].append(vl)
            hist["val_acc"].append(va)
        if verbose:
            msg = f"epoch {epoch + 1:3d}  loss {hist['train_loss'][-1]:.4f}  acc {hist['train_acc'][-1]:.3f}"
            if X_val is not None:
                msg += f"  val_loss {hist['val_loss'][-1]:.4f}  val_acc {hist['val_acc'][-1]:.3f}"
            print(msg)
    return hist


@torch.no_grad()
def evaluate_classifier(model: nn.Module, X, y, device=None, batch_size: int = 1024):
    """Return (BCE loss, accuracy) on (X, y)."""
    device = torch.device(device or "cpu")
    model.eval()
    X, y = _t(X), _t(y)
    loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
    tot = correct = 0.0
    for i in range(0, len(y), batch_size):
        xb, yb = X[i:i + batch_size].to(device), y[i:i + batch_size].to(device)
        logit = model(xb).squeeze(-1)
        tot += loss_fn(logit, yb).item()
        correct += ((logit > 0).float() == yb).sum().item()
    return tot / len(y), correct / len(y)


def train_generator(model: nn.Module, X, T, h, *, X_val=None, T_val=None, h_val=None,
                    epochs: int = 20, batch_size: int = 128, lr: float = 1e-3,
                    weight_decay: float = 0.0, device: Optional[str] = None, seed: int = 0,
                    verbose: bool = False) -> dict:
    """
    Forward-KL / maximum-likelihood training of a SimpleGenCNN on MC data.

    Minimises the negative mean log-likelihood  -mean(model.log_prob(x, T, h))
    (honouring `model.z2_equivariant`). X: (N, L, L) in {-1, +1}; T, h: scalars or (N,).
    Returns a history dict with 'train_nll' (and 'val_nll'), in nats per configuration.
    """
    device = torch.device(device or "cpu")
    torch.manual_seed(seed)
    model.to(device)
    X = _t(X)
    N = len(X)
    Tt, ht = _broadcast(T, N), _broadcast(h, N)
    loader = _loader((X, Tt, ht), batch_size, True, seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    hist = {"train_nll": []}
    if X_val is not None:
        hist["val_nll"] = []

    for epoch in range(epochs):
        model.train()
        tot = n = 0
        for xb, Tb, hb in loader:
            xb, Tb, hb = xb.to(device), Tb.to(device), hb.to(device)
            loss = -model.log_prob(xb, Tb, hb).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
            n += len(xb)
        hist["train_nll"].append(tot / n)
        if X_val is not None:
            hist["val_nll"].append(evaluate_generator(model, X_val, T_val, h_val, device=device))
        if verbose:
            msg = f"epoch {epoch + 1:3d}  nll {hist['train_nll'][-1]:.4f}"
            if X_val is not None:
                msg += f"  val_nll {hist['val_nll'][-1]:.4f}"
            print(msg)
    return hist


@torch.no_grad()
def evaluate_generator(model: nn.Module, X, T, h, device=None, batch_size: int = 1024) -> float:
    """Mean negative log-likelihood (nats per configuration) on (X, T, h)."""
    device = torch.device(device or "cpu")
    model.eval()
    X = _t(X)
    N = len(X)
    Tt, ht = _broadcast(T, N), _broadcast(h, N)
    tot = 0.0
    for i in range(0, N, batch_size):
        sl = slice(i, i + batch_size)
        lp = model.log_prob(X[sl].to(device), Tt[sl].to(device), ht[sl].to(device))
        tot -= lp.sum().item()
    return tot / N
