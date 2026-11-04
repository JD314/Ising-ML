"""
MLP-03: Baselines vs Optimal Bayesian Oracle on 2D Ising (L=10).
Evaluates 6 models with chain-grouped bootstrap confidence intervals:
  1. Exact Bayesian Oracle P*(y=1|E)
  2. Logistic Regression on E
  3. Logistic Regression on |M|
  4. Logistic Regression on (E, |M|)
  5. Linear Classifier on spins s
  6. Frozen MLP (Random Features, 100 -> 16 -> 1)
"""

from __future__ import annotations

import json
import os
import time

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

from ising_fisher import exact_L10 as ex


def predictive_entropy(p: np.ndarray) -> np.ndarray:
    """Predictive entropy H(p) = -p ln p - (1-p) ln (1-p) in nats."""
    p = np.clip(p, 1e-15, 1.0 - 1e-15)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def chain_bootstrap_ci(
    data_by_chain: dict[int, tuple[np.ndarray, np.ndarray]],
    stat_fn,
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap confidence intervals by resampling whole chains."""
    chains = list(data_by_chain.keys())
    n_chains = len(chains)
    rng = np.random.default_rng(seed)
    sampled_indices = rng.choice(n_chains, size=(n_boot, n_chains), replace=True)

    boot_vals = np.empty(n_boot)
    for b in range(n_boot):
        b_chains = [chains[i] for i in sampled_indices[b]]
        y_b = np.concatenate([data_by_chain[c][0] for c in b_chains])
        p_b = np.concatenate([data_by_chain[c][1] for c in b_chains])
        boot_vals[b] = stat_fn(y_b, p_b)

    ci_lo, ci_hi = np.percentile(boot_vals, [2.5, 97.5])
    return float(ci_lo), float(ci_hi)


def run_mlp03(
    data_path: str = "data/L10_data_audit.h5",
    cfg_path: str = "experiments/mlp_L10/MLP-02/config.json",
    out_dir: str = "experiments/mlp_L10/MLP-03",
    n_boot: int = 1000,
):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "figs"), exist_ok=True)

    with open(cfg_path) as f:
        cfg = json.load(f)

    splits = cfg["splits"]
    train_chains = set(splits["train"])
    test_chains = set(splits["test"])
    grid_T = ex.default_temperature_grid()
    K = len(grid_T)

    print(f"Running MLP-03: Loading {data_path} across {K} temperatures...")
    exact_ref = ex.compute_oracle_grid(L=10, grid_T=grid_T)
    lnZ_grid = exact_ref["ln_Z"]

    # Load Train & Test datasets
    E_tr, y_tr, M_tr, S_tr = [], [], [], []
    test_data_by_T = {}  # T -> dict of arrays

    with h5py.File(data_path, "r") as f:
        for T in grid_T:
            grp = f[f"L010/T{T:.6f}/h+0.000000"]
            cid = grp["chain_id"][()]
            e = grp["E_J"][()]
            m = grp["M"][()]
            s = grp["spins"][()]
            y = np.full(len(e), int(T > ex.T_C))

            m_tr = np.isin(cid, list(train_chains))
            m_te = np.isin(cid, list(test_chains))

            E_tr.append(e[m_tr])
            y_tr.append(y[m_tr])
            M_tr.append(m[m_tr])
            S_tr.append(s[m_tr].reshape(len(s[m_tr]), -1))

            test_data_by_T[T] = {
                "E": e[m_te],
                "M": np.abs(m[m_te]),
                "s": s[m_te].reshape(len(s[m_te]), -1).astype(np.float32),
                "y": y[m_te],
                "chain_id": cid[m_te],
            }

    E_tr = np.concatenate(E_tr)[:, None]
    y_tr = np.concatenate(y_tr)
    M_tr = np.abs(np.concatenate(M_tr))[:, None]
    S_tr = np.concatenate(S_tr).astype(np.float32)

    print("Fitting baseline models on train set...")
    # 2. Logistic on E
    clf_E = LogisticRegression(solver="lbfgs")
    clf_E.fit(E_tr, y_tr)

    # 3. Logistic on |M|
    clf_M = LogisticRegression(solver="lbfgs")
    clf_M.fit(M_tr, y_tr)

    # 4. Logistic on (E, |M|)
    EM_tr = np.hstack([E_tr, M_tr])
    clf_EM = LogisticRegression(solver="lbfgs")
    clf_EM.fit(EM_tr, y_tr)

    # 5. Linear on s
    clf_s = LogisticRegression(solver="lbfgs", max_iter=200)
    clf_s.fit(S_tr, y_tr)

    # 6. Frozen MLP (Random Features, 100 -> 16)
    torch.manual_seed(42)
    W1 = (torch.randn(100, 16, dtype=torch.float32) / np.sqrt(100)).numpy()
    b1 = torch.randn(16, dtype=torch.float32).numpy()
    feat_tr = np.maximum(0, S_tr @ W1 + b1)
    clf_rf = LogisticRegression(solver="lbfgs", max_iter=200)
    clf_rf.fit(feat_tr, y_tr)

    models = ["oracle", "logistic_E", "logistic_M", "logistic_EM", "linear_s", "frozen_mlp"]
    model_labels = {
        "oracle": "Bayesian Oracle P*(y=1|E)",
        "logistic_E": "Logistic on E",
        "logistic_M": "Logistic on |M|",
        "logistic_EM": "Logistic on (E, |M|)",
        "linear_s": "Linear on spins s",
        "frozen_mlp": "Frozen MLP (Random Features)",
    }

    # Evaluate per T with bootstrap
    print(f"Evaluating models on test set with {n_boot} chain bootstrap resamples per T...")
    results_by_model = {
        m: {
            "accuracy": [],
            "acc_ci_lo": [],
            "acc_ci_hi": [],
            "ce": [],
            "ce_ci_lo": [],
            "ce_ci_hi": [],
            "H_p": [],
            "H_ci_lo": [],
            "H_ci_hi": [],
            "delta_ce": [],
            "delta_ce_ci_lo": [],
            "delta_ce_ci_hi": [],
        }
        for m in models
    }

    t0 = time.time()
    for idx_T, T in enumerate(grid_T):
        d_T = test_data_by_T[T]
        y_T = d_T["y"]
        cid_T = d_T["chain_id"]
        unique_chains = np.unique(cid_T)

        # Compute predictions
        p_dict = {}
        # 1. Oracle
        p_dict["oracle"] = np.clip(ex.optimal_bayesian_classifier(d_T["E"], grid_T, lnZ_grid), 1e-15, 1.0 - 1e-15)
        # 2. Logistic on E
        p_dict["logistic_E"] = np.clip(clf_E.predict_proba(d_T["E"][:, None])[:, 1], 1e-15, 1.0 - 1e-15)
        # 3. Logistic on |M|
        p_dict["logistic_M"] = np.clip(clf_M.predict_proba(d_T["M"][:, None])[:, 1], 1e-15, 1.0 - 1e-15)
        # 4. Logistic on (E, |M|)
        em = np.hstack([d_T["E"][:, None], d_T["M"][:, None]])
        p_dict["logistic_EM"] = np.clip(clf_EM.predict_proba(em)[:, 1], 1e-15, 1.0 - 1e-15)
        # 5. Linear on s
        p_dict["linear_s"] = np.clip(clf_s.predict_proba(d_T["s"])[:, 1], 1e-15, 1.0 - 1e-15)
        # 6. Frozen MLP
        feat_te = np.maximum(0, d_T["s"] @ W1 + b1)
        p_dict["frozen_mlp"] = np.clip(clf_rf.predict_proba(feat_te)[:, 1], 1e-15, 1.0 - 1e-15)

        # Baseline per-T stats and bootstrap
        for m in models:
            p_m = p_dict[m]
            acc_val = float(accuracy_score(y_T, p_m >= 0.5))
            ce_val = float(log_loss(y_T, p_m, labels=[0, 1]))
            hp_val = float(np.mean(predictive_entropy(p_m)))
            dce_val = float(ce_val - log_loss(y_T, p_dict["oracle"], labels=[0, 1]))

            # Partition by chain
            data_chain = {c: (y_T[cid_T == c], p_m[cid_T == c]) for c in unique_chains}
            data_chain_delta = {c: (y_T[cid_T == c], np.column_stack([p_m[cid_T == c], p_dict["oracle"][cid_T == c]])) for c in unique_chains}

            acc_lo, acc_hi = chain_bootstrap_ci(data_chain, lambda y, p: accuracy_score(y, p >= 0.5), n_boot=n_boot, seed=idx_T)
            ce_lo, ce_hi = chain_bootstrap_ci(data_chain, lambda y, p: log_loss(y, p, labels=[0, 1]), n_boot=n_boot, seed=idx_T + 1)
            hp_lo, hp_hi = chain_bootstrap_ci(data_chain, lambda y, p: float(np.mean(predictive_entropy(p))), n_boot=n_boot, seed=idx_T + 2)
            dce_lo, dce_hi = chain_bootstrap_ci(
                data_chain_delta,
                lambda y, p: log_loss(y, p[:, 0], labels=[0, 1]) - log_loss(y, p[:, 1], labels=[0, 1]),
                n_boot=n_boot,
                seed=idx_T + 3,
            )

            results_by_model[m]["accuracy"].append(acc_val)
            results_by_model[m]["acc_ci_lo"].append(acc_lo)
            results_by_model[m]["acc_ci_hi"].append(acc_hi)
            results_by_model[m]["ce"].append(ce_val)
            results_by_model[m]["ce_ci_lo"].append(ce_lo)
            results_by_model[m]["ce_ci_hi"].append(ce_hi)
            results_by_model[m]["H_p"].append(hp_val)
            results_by_model[m]["H_ci_lo"].append(hp_lo)
            results_by_model[m]["H_ci_hi"].append(hp_hi)
            results_by_model[m]["delta_ce"].append(dce_val)
            results_by_model[m]["delta_ce_ci_lo"].append(dce_lo)
            results_by_model[m]["delta_ce_ci_hi"].append(dce_hi)

    print(f"Bootstrap evaluation finished in {time.time() - t0:.1f}s.")

    # Global cross entropy on full test set
    full_y_te = np.concatenate([test_data_by_T[T]["y"] for T in grid_T])
    full_p_oracle = np.concatenate([
        np.clip(ex.optimal_bayesian_classifier(test_data_by_T[T]["E"], grid_T, lnZ_grid), 1e-15, 1.0 - 1e-15)
        for T in grid_T
    ])
    full_p_E = np.concatenate([
        np.clip(clf_E.predict_proba(test_data_by_T[T]["E"][:, None])[:, 1], 1e-15, 1.0 - 1e-15)
        for T in grid_T
    ])

    global_ce_oracle = float(log_loss(full_y_te, full_p_oracle))
    global_ce_E = float(log_loss(full_y_te, full_p_E))
    ce_excess_logistic_E_pct = float((global_ce_E - global_ce_oracle) / global_ce_oracle * 100.0)

    print(f"Global CE: Oracle = {global_ce_oracle:.6f}, Logistic(E) = {global_ce_E:.6f}")
    print(f"CE excess of Logistic on E: {ce_excess_logistic_E_pct:.4f}%")

    # Global accuracies
    global_accuracies = {}
    for m in models:
        m_acc = float(np.mean([results_by_model[m]["accuracy"]]))
        global_accuracies[m] = m_acc

    # Save metrics.json
    metrics = {
        "task_id": "MLP-03",
        "L": 10,
        "N": 100,
        "n_temperatures": K,
        "models": models,
        "global_ce_oracle": global_ce_oracle,
        "global_ce_logistic_E": global_ce_E,
        "ce_excess_logistic_E_pct": ce_excess_logistic_E_pct,
        "meets_criterion_ce_excess": bool(ce_excess_logistic_E_pct <= 2.5),
        "temperature_grid": grid_T.tolist(),
        "T_c": ex.T_C,
        "T_peak": 2.279,
        "results": results_by_model,
    }

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    config = {
        "task_id": "MLP-03",
        "models": models,
        "n_boot": n_boot,
        "split_chains": splits,
        "train_samples": len(y_tr),
        "test_samples": len(full_y_te),
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # ---------------- FIGURES ----------------
    # Fig 1: Accuracy(T) comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {
        "oracle": "black",
        "logistic_E": "tab:blue",
        "logistic_M": "tab:purple",
        "logistic_EM": "tab:cyan",
        "linear_s": "tab:red",
        "frozen_mlp": "tab:green",
    }
    styles = {
        "oracle": "-",
        "logistic_E": "--",
        "logistic_M": ":",
        "logistic_EM": "-.",
        "linear_s": "-",
        "frozen_mlp": "--",
    }

    for m in models:
        acc = np.array(results_by_model[m]["accuracy"])
        lo = np.array(results_by_model[m]["acc_ci_lo"])
        hi = np.array(results_by_model[m]["acc_ci_hi"])
        ax.plot(grid_T, acc, styles[m], color=colors[m], label=model_labels[m], linewidth=1.8 if m == "oracle" else 1.5)
        ax.fill_between(grid_T, lo, hi, color=colors[m], alpha=0.15)

    ax.axvline(ex.T_C, color="gray", linestyle=":", label=rf"$T_c \approx {ex.T_C:.4f}$")
    ax.axvline(2.279, color="tab:orange", linestyle="--", alpha=0.8, label=r"$T_{\text{peak}} \approx 2.279$")
    ax.set_xlabel("Temperature $T$", fontsize=12)
    ax.set_ylabel("Classification Accuracy", fontsize=12)
    ax.set_title(r"MLP-03: Baseline Models vs Optimal Bayesian Oracle ($L=10$, Torus)", fontsize=13)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "figs", "baselines_accuracy_vs_T.png"), dpi=150)
    plt.close(fig)

    # Fig 2: Delta CE vs Oracle
    fig, ax = plt.subplots(figsize=(10, 6))
    for m in models:
        if m == "oracle":
            continue
        dce = np.array(results_by_model[m]["delta_ce"])
        lo = np.array(results_by_model[m]["delta_ce_ci_lo"])
        hi = np.array(results_by_model[m]["delta_ce_ci_hi"])
        ax.plot(grid_T, dce, label=model_labels[m], color=colors[m], linewidth=1.5)
        ax.fill_between(grid_T, lo, hi, color=colors[m], alpha=0.15)

    ax.axvline(ex.T_C, color="gray", linestyle=":", label=r"$T_c$")
    ax.axhline(0, color="k", linestyle="--", alpha=0.5)
    ax.set_xlabel("Temperature $T$", fontsize=12)
    ax.set_ylabel(r"$\Delta \text{CE}(T) = \text{CE}_{\text{model}} - \text{CE}_{\text{oracle}}$ [nats]", fontsize=12)
    ax.set_title(r"MLP-03: Cross-Entropy Gap Relative to Optimal Bayesian Oracle", fontsize=13)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "figs", "delta_ce_vs_oracle.png"), dpi=150)
    plt.close(fig)

    print("All MLP-03 deliverables saved successfully.")


if __name__ == "__main__":
    run_mlp03()
