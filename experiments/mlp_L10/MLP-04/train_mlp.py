"""
Training script for MLP-04: SimpleMLP reference model across 5 seeds (0-4).
Saves checkpoints on fixed schedule:
  - epochs e in {0, 1, 2, 3, 5, 8, 13, 21, 34, 55}
  - best model, final model
  - intra-epoch steps {1, 5, 20, 50, 200} of epoch 1
Tracks loss & accuracy for train, val, test, and per-T validation/test curves.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from typing import Dict, List

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ising_fisher import exact_L10 as ex
from ising_fisher.models import SimpleMLP

SCHEDULE_EPOCHS = {0, 1, 2, 3, 5, 8, 13, 21, 34, 55}
INTRA_STEPS = {1, 5, 20, 50, 200}


def get_git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def evaluate_model(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> tuple[float, float]:
    model.eval()
    tot_loss = 0.0
    tot_correct = 0
    tot_samples = 0
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb).squeeze(-1)
            loss = criterion(logits, yb)
            tot_loss += loss.item() * len(yb)
            preds = (logits >= 0.0).float()
            tot_correct += int((preds == yb).sum().item())
            tot_samples += len(yb)
    return float(tot_loss / tot_samples), float(tot_correct / tot_samples)


def evaluate_per_T(model: nn.Module, data_by_T: dict, criterion: nn.Module) -> tuple[list[float], list[float]]:
    model.eval()
    losses = []
    accs = []
    with torch.no_grad():
        for T in sorted(data_by_T.keys()):
            xb = torch.from_numpy(data_by_T[T]["spins"])
            yb = torch.from_numpy(data_by_T[T]["y"])
            logits = model(xb).squeeze(-1)
            loss = criterion(logits, yb)
            preds = (logits >= 0.0).float()
            acc = float((preds == yb).float().mean().item())
            losses.append(float(loss.item()))
            accs.append(acc)
    return losses, accs


def run_training_seed(
    seed: int,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    va_by_T: dict,
    te_by_T: dict,
    ckpt_root: str,
    batch_size: int = 256,
    lr: float = 1e-3,
    max_epochs: int = 200,
    patience: int = 10,
) -> dict:
    seed_dir = os.path.join(ckpt_root, f"seed{seed}")
    os.makedirs(seed_dir, exist_ok=True)

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Prepare datasets
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va)),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te)),
        batch_size=batch_size,
        shuffle=False,
    )

    model = SimpleMLP(L=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    history = {
        "train_loss": [],
        "val_loss": [],
        "test_loss": [],
        "train_acc": [],
        "val_acc": [],
        "test_acc": [],
    }

    # Epoch 0 evaluation (before any gradient updates)
    tr_loss0, tr_acc0 = evaluate_model(model, train_loader, criterion)
    va_loss0, va_acc0 = evaluate_model(model, val_loader, criterion)
    te_loss0, te_acc0 = evaluate_model(model, test_loader, criterion)

    history["train_loss"].append(tr_loss0)
    history["val_loss"].append(va_loss0)
    history["test_loss"].append(te_loss0)
    history["train_acc"].append(tr_acc0)
    history["val_acc"].append(va_acc0)
    history["test_acc"].append(te_acc0)

    # Checkpoint epoch 0
    torch.save(
        {
            "epoch": 0,
            "seed": seed,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        os.path.join(seed_dir, "epoch0.pt"),
    )

    best_val_loss = va_loss0
    best_epoch = 0
    patience_counter = 0

    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    step_count = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        tot_tr_loss = 0.0
        tot_tr_correct = 0
        tot_tr_samples = 0

        for xb, yb in train_loader:
            step_count += 1
            optimizer.zero_grad()
            logits = model(xb).squeeze(-1)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            tot_tr_loss += loss.item() * len(yb)
            preds = (logits >= 0.0).float()
            tot_tr_correct += int((preds == yb).sum().item())
            tot_tr_samples += len(yb)

            # Intra-epoch checkpoints during Epoch 1
            if epoch == 1 and step_count in INTRA_STEPS:
                torch.save(
                    {
                        "epoch": 1,
                        "step": step_count,
                        "seed": seed,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    os.path.join(seed_dir, f"epoch1_step{step_count}.pt"),
                )

        ep_tr_loss = float(tot_tr_loss / tot_tr_samples)
        ep_tr_acc = float(tot_tr_correct / tot_tr_samples)
        ep_va_loss, ep_va_acc = evaluate_model(model, val_loader, criterion)
        ep_te_loss, ep_te_acc = evaluate_model(model, test_loader, criterion)

        history["train_loss"].append(ep_tr_loss)
        history["val_loss"].append(ep_va_loss)
        history["test_loss"].append(ep_te_loss)
        history["train_acc"].append(ep_tr_acc)
        history["val_acc"].append(ep_va_acc)
        history["test_acc"].append(ep_te_acc)

        # Checkpoint schedule
        if epoch in SCHEDULE_EPOCHS:
            torch.save(
                {
                    "epoch": epoch,
                    "seed": seed,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                os.path.join(seed_dir, f"epoch{epoch}.pt"),
            )

        # Early stopping tracking
        if ep_va_loss < best_val_loss - 1e-4:
            best_val_loss = ep_va_loss
            best_epoch = epoch
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(
                {
                    "epoch": epoch,
                    "seed": seed,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                os.path.join(seed_dir, "best.pt"),
            )
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Seed {seed}: Early stopping triggered at epoch {epoch} (best epoch {best_epoch}, val_loss={best_val_loss:.4f})")
            break

    # Save final model checkpoint
    final_epoch = len(history["val_loss"]) - 1
    torch.save(
        {
            "epoch": final_epoch,
            "seed": seed,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        os.path.join(seed_dir, "final.pt"),
    )

    # Evaluate final and best on per-T validation and test
    model.load_state_dict(best_state)
    val_loss_per_T, val_acc_per_T = evaluate_per_T(model, va_by_T, criterion)
    test_loss_per_T, test_acc_per_T = evaluate_per_T(model, te_by_T, criterion)

    # Compute stability criterion: val_loss relative variation in last 5 epochs
    last_val_losses = history["val_loss"][-5:]
    val_loss_variation_pct = float((max(last_val_losses) - min(last_val_losses)) / np.mean(last_val_losses) * 100.0)

    final_test_acc = history["test_acc"][best_epoch]
    print(f"Seed {seed}: Finished in {final_epoch} epochs. Best epoch={best_epoch}, best val_loss={best_val_loss:.4f}, test_acc={final_test_acc:.4f}, val_loss_var={val_loss_variation_pct:.2f}%")

    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "final_epoch": final_epoch,
        "best_val_loss": best_val_loss,
        "best_val_acc": float(history["val_acc"][best_epoch]),
        "test_acc_at_best": float(history["test_acc"][best_epoch]),
        "test_loss_at_best": float(history["test_loss"][best_epoch]),
        "val_loss_variation_pct": val_loss_variation_pct,
        "history": history,
        "val_loss_per_T": val_loss_per_T,
        "val_acc_per_T": val_acc_per_T,
        "test_loss_per_T": test_loss_per_T,
        "test_acc_per_T": test_acc_per_T,
    }


def main():
    data_path = "data/L10_data_audit.h5"
    cfg_path = "experiments/mlp_L10/MLP-02/config.json"
    out_dir = "experiments/mlp_L10/MLP-04"
    ckpt_root = "ckpt"

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "figs"), exist_ok=True)
    os.makedirs(ckpt_root, exist_ok=True)

    with open(cfg_path) as f:
        cfg = json.load(f)

    splits = cfg["splits"]
    train_chains = set(splits["train"])
    val_chains = set(splits["val"])
    test_chains = set(splits["test"])

    grid_T = ex.default_temperature_grid()
    K = len(grid_T)

    print(f"MLP-04: Loading data from {data_path} across {K} temperatures...")
    X_tr, y_tr = [], []
    X_va, y_va = [], []
    X_te, y_te = [], []

    va_by_T = {}
    te_by_T = {}

    with h5py.File(data_path, "r") as f:
        for T in grid_T:
            grp = f[f"L010/T{T:.6f}/h+0.000000"]
            cid = grp["chain_id"][()]
            s = grp["spins"][()]
            y = np.full(len(s), int(T > ex.T_C), dtype=np.float32)

            m_tr = np.isin(cid, list(train_chains))
            m_va = np.isin(cid, list(val_chains))
            m_te = np.isin(cid, list(test_chains))

            X_tr.append(s[m_tr])
            y_tr.append(y[m_tr])
            X_va.append(s[m_va])
            y_va.append(y[m_va])
            X_te.append(s[m_te])
            y_te.append(y[m_te])

            va_by_T[T] = {"spins": s[m_va].astype(np.float32), "y": y[m_va]}
            te_by_T[T] = {"spins": s[m_te].astype(np.float32), "y": y[m_te]}

    X_tr = np.concatenate(X_tr).astype(np.float32)
    y_tr = np.concatenate(y_tr)
    X_va = np.concatenate(X_va).astype(np.float32)
    y_va = np.concatenate(y_va)
    X_te = np.concatenate(X_te).astype(np.float32)
    y_te = np.concatenate(y_te)

    print(f"Dataset loaded: Train={len(y_tr)}, Val={len(y_va)}, Test={len(y_te)}")

    seeds = [0, 1, 2, 3, 4]
    results_by_seed = {}

    t0_all = time.time()
    for s in seeds:
        print(f"\n--- Training Seed {s} ---")
        res = run_training_seed(
            seed=s,
            X_tr=X_tr,
            y_tr=y_tr,
            X_va=X_va,
            y_va=y_va,
            X_te=X_te,
            y_te=y_te,
            va_by_T=va_by_T,
            te_by_T=te_by_T,
            ckpt_root=ckpt_root,
            batch_size=256,
            lr=1e-3,
            max_epochs=200,
            patience=10,
        )
        results_by_seed[s] = res

    print(f"\nAll 5 seeds trained in {time.time() - t0_all:.1f}s.")

    # Quality criteria verification
    stable_seeds = [s for s in seeds if results_by_seed[s]["val_loss_variation_pct"] < 1.0]
    mean_test_acc = float(np.mean([results_by_seed[s]["test_acc_at_best"] for s in seeds]))

    print(f"Stable seeds (<1% val loss variation in last 5 epochs): {len(stable_seeds)}/5 ({stable_seeds})")
    print(f"Mean test accuracy across seeds: {mean_test_acc:.4f}")

    # Build metrics.json
    metrics = {
        "task_id": "MLP-04",
        "git_hash": get_git_hash(),
        "L": 10,
        "N": 100,
        "architecture": "SimpleMLP (Flatten -> Linear(100, 16) -> ReLU -> Linear(16, 1))",
        "parameters": 1633,
        "seeds": seeds,
        "epochs_to_convergence": {s: results_by_seed[s]["best_epoch"] for s in seeds},
        "final_epochs": {s: results_by_seed[s]["final_epoch"] for s in seeds},
        "best_val_loss": {s: results_by_seed[s]["best_val_loss"] for s in seeds},
        "test_acc_at_best": {s: results_by_seed[s]["test_acc_at_best"] for s in seeds},
        "mean_test_acc": mean_test_acc,
        "val_loss_variation_pct": {s: results_by_seed[s]["val_loss_variation_pct"] for s in seeds},
        "stable_seeds_count": len(stable_seeds),
        "meets_stability_criterion": bool(len(stable_seeds) >= 4),
        "grid_T": grid_T.tolist(),
        "val_acc_per_T_by_seed": {s: results_by_seed[s]["val_acc_per_T"] for s in seeds},
        "test_acc_per_T_by_seed": {s: results_by_seed[s]["test_acc_per_T"] for s in seeds},
        "test_loss_per_T_by_seed": {s: results_by_seed[s]["test_loss_per_T"] for s in seeds},
    }

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    config = {
        "task_id": "MLP-04",
        "git_hash": get_git_hash(),
        "architecture": "SimpleMLP (100 -> 16 -> 1, ReLU)",
        "batch_size": 256,
        "lr": 1e-3,
        "optimizer": "Adam",
        "loss": "BCEWithLogitsLoss",
        "patience": 10,
        "max_epochs": 200,
        "seeds": seeds,
        "checkpoint_schedule": list(SCHEDULE_EPOCHS),
        "intra_epoch_steps": list(INTRA_STEPS),
        "splits": splits,
        "train_samples": len(y_tr),
        "val_samples": len(y_va),
        "test_samples": len(y_te),
    }

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # ---------------- FIGURES ----------------
    # Fig 1: Training and Validation loss curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for s in seeds:
        h = results_by_seed[s]["history"]
        epochs_arr = np.arange(len(h["train_loss"]))
        ax1.plot(epochs_arr, h["train_loss"], "--", alpha=0.5, label=f"Train s{s}" if s == 0 else None)
        ax1.plot(epochs_arr, h["val_loss"], "-", label=f"Val s{s}")
        ax2.plot(epochs_arr, h["val_acc"], "-", label=f"Val Acc s{s}")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("BCE Loss")
    ax1.set_title("MLP-04 Training & Validation Loss Curves")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("MLP-04 Validation Accuracy Curves")
    ax2.grid(alpha=0.3)
    ax2.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "figs", "training_val_curves.png"), dpi=150)
    plt.close(fig)

    # Fig 2: Accuracy per T across seeds vs Tc and T_peak
    fig, ax = plt.subplots(figsize=(9, 5.5))
    acc_matrix = np.array([results_by_seed[s]["test_acc_per_T"] for s in seeds])  # (5, 64)
    acc_mean = np.mean(acc_matrix, axis=0)
    acc_std = np.std(acc_matrix, axis=0)

    ax.plot(grid_T, acc_mean, "o-", color="tab:blue", label="SimpleMLP (Mean across 5 seeds)")
    ax.fill_between(grid_T, acc_mean - acc_std, acc_mean + acc_std, color="tab:blue", alpha=0.2, label=r"$\pm 1\sigma$ Seed Spread")
    ax.axvline(ex.T_C, color="gray", linestyle=":", label=rf"$T_c \approx {ex.T_C:.4f}$")
    ax.axvline(2.279, color="tab:red", linestyle="--", label=r"$T_{\text{peak}} \approx 2.279$")
    ax.set_xlabel("Temperature $T$", fontsize=12)
    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_title(r"MLP-04: Test Accuracy per Temperature across 5 Seeds ($L=10$)", fontsize=13)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "figs", "test_accuracy_per_T.png"), dpi=150)
    plt.close(fig)

    print("All MLP-04 deliverables and figures saved.")


if __name__ == "__main__":
    main()
