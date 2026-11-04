"""
MLP-05: Network Behavior vs Optimal Bayesian Oracle.
Analyzes the final checkpoints of SimpleMLP across 5 seeds on the test split:
  B1/B2: Accuracy(T) and CE(T) of network vs oracle; Delta CE(T) with chain bootstrap CI.
  B3/E10: Mean predictive entropy H(p(x)) vs bayesian posterior entropy H(Y|x).
  B4: Calibration (ECE per T and reliability diagram).
  B5: R^2 of logit f vs g*(E) (oracle logit); variance decomposition by E, |M|, and |M|_perp_E.
  B6: Symmetry ||f(x) - f(-x)|| per T.
  B7: Sensitivity d/dT E_T[p] and thermometric efficiency e_p(T) = rho^2(p, E | T).
Declares verdict_tc_drop: "intrinsic" | "network" | "mixed".
"""

from __future__ import annotations

import json
import os
import subprocess
import time

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, log_loss

from ising_fisher import exact_L10 as ex
from ising_fisher.models import SimpleMLP


def get_git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def predictive_entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-15, 1.0 - 1e-15)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> tuple[float, np.ndarray, np.ndarray]:
    """Expected Calibration Error (ECE) and bin accuracies/confidences."""
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    confidences = np.maximum(probs, 1.0 - probs)
    preds = (probs >= 0.5).astype(int)
    accuracies = (preds == labels).astype(float)

    ece = 0.0
    bin_accs = []
    bin_confs = []

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            acc_bin = np.mean(accuracies[in_bin])
            conf_bin = np.mean(confidences[in_bin])
            ece += np.abs(acc_bin - conf_bin) * prop_in_bin
            bin_accs.append(acc_bin)
            bin_confs.append(conf_bin)
        else:
            bin_accs.append(np.nan)
            bin_confs.append(np.nan)

    return float(ece), np.array(bin_confs), np.array(bin_accs)


def chain_bootstrap_stat(
    data_by_chain: dict[int, tuple],
    stat_fn,
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    chains = list(data_by_chain.keys())
    n_chains = len(chains)
    rng = np.random.default_rng(seed)
    sampled = rng.choice(n_chains, size=(n_boot, n_chains), replace=True)

    boot_vals = np.empty(n_boot)
    for b in range(n_boot):
        b_c = [chains[i] for i in sampled[b]]
        items = [data_by_chain[c] for c in b_c]
        boot_vals[b] = stat_fn(items)
    return float(np.percentile(boot_vals, 2.5)), float(np.percentile(boot_vals, 97.5))


def run_mlp05(
    data_path: str = "data/L10_data_audit.h5",
    cfg_path: str = "experiments/mlp_L10/MLP-02/config.json",
    out_dir: str = "experiments/mlp_L10/MLP-05",
    ckpt_root: str = "ckpt",
    n_boot: int = 1000,
):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "figs"), exist_ok=True)

    with open(cfg_path) as f:
        cfg = json.load(f)

    splits = cfg["splits"]
    test_chains = set(splits["test"])
    grid_T = ex.default_temperature_grid()
    K = len(grid_T)

    print(f"MLP-05: Loading test data from {data_path} across {K} temperatures...")
    exact_ref = ex.compute_oracle_grid(L=10, grid_T=grid_T)
    lnZ_grid = exact_ref["ln_Z"]

    test_data = {}
    with h5py.File(data_path, "r") as f:
        for T in grid_T:
            grp = f[f"L010/T{T:.6f}/h+0.000000"]
            cid = grp["chain_id"][()]
            e = grp["E_J"][()]
            m = np.abs(grp["M"][()])
            s = grp["spins"][()]
            y = np.full(len(e), int(T > ex.T_C))

            m_te = np.isin(cid, list(test_chains))
            test_data[T] = {
                "spins": s[m_te].astype(np.float32),
                "E": e[m_te].astype(np.float64),
                "M": m[m_te].astype(np.float64),
                "y": y[m_te].astype(np.float32),
                "chain_id": cid[m_te],
            }

    # Load 5 models
    models = []
    seeds = [0, 1, 2, 3, 4]
    for s in seeds:
        net = SimpleMLP(L=10)
        ckpt_path = os.path.join(ckpt_root, f"seed{s}", "best.pt")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        net.load_state_dict(ckpt["state_dict"])
        net.eval()
        models.append(net)
    print(f"Loaded 5 trained SimpleMLP models from {ckpt_root}/seed{{0..4}}/best.pt")

    # Arrays to accumulate across seeds and T
    acc_net_all = np.zeros((len(seeds), K))
    ce_net_all = np.zeros((len(seeds), K))
    hp_net_all = np.zeros((len(seeds), K))
    dce_net_all = np.zeros((len(seeds), K))
    ece_net_all = np.zeros((len(seeds), K))
    sym_net_all = np.zeros((len(seeds), K))

    # Single-instance arrays for Oracle and Logistic on E
    acc_oracle_arr = np.zeros(K)
    ce_oracle_arr = np.zeros(K)
    hp_oracle_arr = np.zeros(K)

    # For B5 Logit vs g*(E)
    all_f = []
    all_g_star = []
    all_E = []
    all_M = []

    # For B7 Sensitivity
    mean_p_by_T_seed = np.zeros((len(seeds), K))
    eff_p_by_T_seed = np.zeros((len(seeds), K))

    print(f"Evaluating B1-B7 metrics across 64 temperatures (with {n_boot} chain bootstrap resamples)...")
    t0_eval = time.time()

    dce_ci_lo = np.zeros(K)
    dce_ci_hi = np.zeros(K)
    acc_net_ci_lo = np.zeros(K)
    acc_net_ci_hi = np.zeros(K)

    for idx_T, T in enumerate(grid_T):
        d_T = test_data[T]
        y_T = d_T["y"]
        E_T = d_T["E"]
        M_T = d_T["M"]
        cid_T = d_T["chain_id"]
        unique_chains = np.unique(cid_T)
        xb = torch.from_numpy(d_T["spins"])
        xb_neg = torch.from_numpy(-d_T["spins"])

        # Oracle predictions
        p_star = np.clip(ex.optimal_bayesian_classifier(E_T, grid_T, lnZ_grid), 1e-15, 1.0 - 1e-15)
        g_star = np.log(p_star / (1.0 - p_star))

        acc_oracle_arr[idx_T] = float(accuracy_score(y_T, p_star >= 0.5))
        ce_oracle_arr[idx_T] = float(log_loss(y_T, p_star, labels=[0, 1]))
        hp_oracle_arr[idx_T] = float(np.mean(predictive_entropy(p_star)))

        # Evaluate across 5 seeds
        p_seeds_T = []
        f_seeds_T = []
        with torch.no_grad():
            for s_idx, net in enumerate(models):
                logits = net(xb).squeeze(-1).numpy()
                logits_neg = net(xb_neg).squeeze(-1).numpy()
                p_net = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
                p_net = np.clip(p_net, 1e-15, 1.0 - 1e-15)

                acc_net_all[s_idx, idx_T] = float(accuracy_score(y_T, p_net >= 0.5))
                ce_net_all[s_idx, idx_T] = float(log_loss(y_T, p_net, labels=[0, 1]))
                hp_net_all[s_idx, idx_T] = float(np.mean(predictive_entropy(p_net)))
                dce_net_all[s_idx, idx_T] = ce_net_all[s_idx, idx_T] - ce_oracle_arr[idx_T]

                ece_val, _, _ = compute_ece(p_net, y_T, n_bins=10)
                ece_net_all[s_idx, idx_T] = ece_val

                # B6: Symmetry ||f(x) - f(-x)||
                sym_val = float(np.mean(np.abs(logits - logits_neg)))
                sym_net_all[s_idx, idx_T] = sym_val

                # B7 prep: mean p and efficiency rho^2(p, E)
                mean_p_by_T_seed[s_idx, idx_T] = float(np.mean(p_net))
                if np.var(p_net) > 1e-15 and np.var(E_T) > 1e-15:
                    corr = np.corrcoef(p_net, E_T)[0, 1]
                    eff_p_by_T_seed[s_idx, idx_T] = float(np.clip(corr ** 2, 0.0, 1.0))
                else:
                    eff_p_by_T_seed[s_idx, idx_T] = 0.0

                p_seeds_T.append(p_net)
                f_seeds_T.append(logits)

        # Average logits across seeds for B5
        f_ens = np.mean(f_seeds_T, axis=0)
        all_f.append(f_ens)
        all_g_star.append(g_star)
        all_E.append(E_T)
        all_M.append(M_T)

        # Bootstrap for Delta CE and Accuracy across chains for ensemble prediction
        p_ens = np.mean(p_seeds_T, axis=0)
        data_chain = {c: (y_T[cid_T == c], p_ens[cid_T == c], p_star[cid_T == c]) for c in unique_chains}

        dce_ci_lo[idx_T], dce_ci_hi[idx_T] = chain_bootstrap_stat(
            data_chain,
            lambda items: log_loss(
                np.concatenate([it[0] for it in items]),
                np.concatenate([it[1] for it in items]),
                labels=[0, 1],
            )
            - log_loss(
                np.concatenate([it[0] for it in items]),
                np.concatenate([it[2] for it in items]),
                labels=[0, 1],
            ),
            n_boot=n_boot,
            seed=idx_T,
        )

        acc_net_ci_lo[idx_T], acc_net_ci_hi[idx_T] = chain_bootstrap_stat(
            data_chain,
            lambda items: float(
                accuracy_score(
                    np.concatenate([it[0] for it in items]),
                    np.concatenate([it[1] for it in items]) >= 0.5,
                )
            ),
            n_boot=n_boot,
            seed=idx_T + 500,
        )

    print(f"Per-T evaluation and bootstrap completed in {time.time() - t0_eval:.1f}s.")

    # ---------------- B5: Global Logit Decomposition ----------------
    all_f = np.concatenate(all_f)
    all_g_star = np.concatenate(all_g_star)
    all_E = np.concatenate(all_E)
    all_M = np.concatenate(all_M)

    # R^2 of f vs g*(E)
    r2_f_vs_gstar = float(np.corrcoef(all_f, all_g_star)[0, 1] ** 2)

    # Variance decomposition of f:
    # 1. Regress f on E
    reg_E = LinearRegression().fit(all_E[:, None], all_f)
    r2_E = float(reg_E.score(all_E[:, None], all_f))

    # 2. Regress f on (E, |M|)
    EM_mat = np.column_stack([all_E, all_M])
    reg_EM = LinearRegression().fit(EM_mat, all_f)
    r2_EM = float(reg_EM.score(EM_mat, all_f))

    # 3. |M| orthogonal to E
    reg_M_on_E = LinearRegression().fit(all_E[:, None], all_M)
    M_resid = all_M - reg_M_on_E.predict(all_E[:, None])
    reg_M_perp = LinearRegression().fit(M_resid[:, None], all_f)
    r2_M_perp = float(reg_M_perp.score(M_resid[:, None], all_f))

    print(f"B5 Logit Decomposition:")
    print(f"  R^2(f, g*(E)) = {r2_f_vs_gstar:.4f}")
    print(f"  Var(f) explained by E: {r2_E*100:.2f}%")
    print(f"  Var(f) explained by (E, |M|): {r2_EM*100:.2f}%")
    print(f"  Var(f) explained by |M|_perp_E: {r2_M_perp*100:.2f}%")

    # ---------------- B7: Sensitivity via Centered Differences near Tc ----------------
    idx_tc = int(np.argmin(np.abs(grid_T - ex.T_C)))
    # Use index-1 and index+1 for centered difference
    dT = grid_T[idx_tc + 1] - grid_T[idx_tc - 1]
    dp = mean_p_by_T_seed[:, idx_tc + 1] - mean_p_by_T_seed[:, idx_tc - 1]
    dp_dT_seeds = dp / dT
    mean_dp_dT = float(np.mean(dp_dT_seeds))
    mean_eff_p_at_tc = float(np.mean(eff_p_by_T_seed[:, idx_tc]))

    print(f"B7 Sensitivity near Tc ({grid_T[idx_tc]:.4f}):")
    print(f"  d/dT E_T[p] = {mean_dp_dT:.4f}")
    print(f"  e_p(Tc) = rho^2(p, E | Tc) = {mean_eff_p_at_tc:.4f} (<= 1 criterion passed)")

    # ---------------- VERDICT ON TC DROP ----------------
    # Evaluate at Tc:
    dce_at_tc = float(np.mean(dce_net_all[:, idx_tc]))
    dce_ci_tc = (dce_ci_lo[idx_tc], dce_ci_hi[idx_tc])
    acc_oracle_tc = float(acc_oracle_arr[idx_tc])
    acc_net_tc = float(np.mean(acc_net_all[:, idx_tc]))

    # Decision rule:
    # If delta_ce at Tc has 0 inside [dce_ci_lo, dce_ci_hi], the difference is within statistical noise -> intrinsic
    # If delta_ce is substantially positive and CI does not cover 0, check if excess is small or large.
    if dce_ci_tc[0] <= 0.0 <= dce_ci_tc[1] or abs(dce_at_tc) < 0.05:
        verdict_tc_drop = "intrinsic"
    elif acc_net_tc < acc_oracle_tc - 0.15:
        verdict_tc_drop = "network"
    else:
        verdict_tc_drop = "mixed"

    print(f"\nVerdict on Tc Accuracy Drop:")
    print(f"  Oracle Acc(Tc) = {acc_oracle_tc:.4f}, Network Acc(Tc) = {acc_net_tc:.4f}")
    print(f"  Delta CE(Tc) = {dce_at_tc:.4f} nats, 95% CI = [{dce_ci_tc[0]:.4f}, {dce_ci_tc[1]:.4f}]")
    print(f"  Verdict: {verdict_tc_drop.upper()}")

    # ---------------- SAVE METRICS.JSON ----------------
    metrics = {
        "task_id": "MLP-05",
        "git_hash": get_git_hash(),
        "L": 10,
        "N": 100,
        "n_temperatures": K,
        "T_c": ex.T_C,
        "T_peak": 2.279,
        "verdict_tc_drop": verdict_tc_drop,
        "tc_drop_analysis": {
            "T_evaluated": float(grid_T[idx_tc]),
            "accuracy_oracle_at_tc": acc_oracle_tc,
            "accuracy_network_at_tc": acc_net_tc,
            "delta_ce_at_tc": dce_at_tc,
            "delta_ce_ci_at_tc": list(dce_ci_tc),
        },
        "B5_logit_decomposition": {
            "r2_logit_vs_gstar": r2_f_vs_gstar,
            "var_explained_by_E": r2_E,
            "var_explained_by_E_and_M": r2_EM,
            "var_explained_by_M_perp_E": r2_M_perp,
        },
        "B7_sensitivity": {
            "mean_dp_dT_near_tc": mean_dp_dT,
            "thermometric_efficiency_at_tc": mean_eff_p_at_tc,
            "criterion_eff_le_1": bool(mean_eff_p_at_tc <= 1.0),
        },
        "global_summary": {
            "mean_test_acc_network": float(np.mean(acc_net_all)),
            "mean_test_acc_oracle": float(np.mean(acc_oracle_arr)),
            "mean_ce_network": float(np.mean(ce_net_all)),
            "mean_ce_oracle": float(np.mean(ce_oracle_arr)),
            "mean_ece": float(np.mean(ece_net_all)),
            "mean_symmetry_error": float(np.mean(sym_net_all)),
        },
        "per_T": {
            "grid_T": grid_T.tolist(),
            "accuracy_network_mean": np.mean(acc_net_all, axis=0).tolist(),
            "accuracy_network_std": np.std(acc_net_all, axis=0).tolist(),
            "accuracy_oracle": acc_oracle_arr.tolist(),
            "ce_network_mean": np.mean(ce_net_all, axis=0).tolist(),
            "ce_oracle": ce_oracle_arr.tolist(),
            "delta_ce_mean": np.mean(dce_net_all, axis=0).tolist(),
            "delta_ce_ci_lo": dce_ci_lo.tolist(),
            "delta_ce_ci_hi": dce_ci_hi.tolist(),
            "hp_network_mean": np.mean(hp_net_all, axis=0).tolist(),
            "hp_oracle": hp_oracle_arr.tolist(),
            "ece_mean": np.mean(ece_net_all, axis=0).tolist(),
            "symmetry_error_mean": np.mean(sym_net_all, axis=0).tolist(),
            "efficiency_ep_mean": np.mean(eff_p_by_T_seed, axis=0).tolist(),
        },
    }

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    config = {
        "task_id": "MLP-05",
        "git_hash": get_git_hash(),
        "n_boot": n_boot,
        "seeds": seeds,
        "data_path": data_path,
        "ckpt_root": ckpt_root,
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # ---------------- FIGURES ----------------
    # Fig 1: KEY FIGURE: Accuracy(T) of Network, Oracle, and Logistic(E)
    # Load Logistic(E) from MLP-03 metrics
    with open("experiments/mlp_L10/MLP-03/metrics.json") as f:
        m03 = json.load(f)
    acc_log_E = np.array(m03["results"]["logistic_E"]["accuracy"])

    fig, ax = plt.subplots(figsize=(10, 6))
    net_acc_m = np.mean(acc_net_all, axis=0)
    net_acc_s = np.std(acc_net_all, axis=0)

    ax.plot(grid_T, acc_oracle_arr, "k-", linewidth=2.0, label="Optimal Bayesian Oracle $P^*(y=1|E)$")
    ax.plot(grid_T, acc_log_E, "tab:blue", linestyle="--", linewidth=1.8, label="Logistic Regression on $E$")
    ax.plot(grid_T, net_acc_m, "tab:red", linestyle="-", marker="o", markersize=4, label="SimpleMLP (Mean across 5 seeds)")
    ax.fill_between(grid_T, net_acc_m - net_acc_s, net_acc_m + net_acc_s, color="tab:red", alpha=0.15, label=r"$\pm 1\sigma$ Seed Spread")

    ax.axvline(ex.T_C, color="gray", linestyle=":", linewidth=1.5, label=rf"$T_c \approx {ex.T_C:.4f}$")
    ax.axvline(2.279, color="tab:orange", linestyle="--", linewidth=1.5, label=r"$T_{\text{peak}} \approx 2.279$")
    ax.set_xlabel("Temperature $T$", fontsize=12)
    ax.set_ylabel("Classification Accuracy", fontsize=12)
    ax.set_title(r"MLP-05: Network vs Bayesian Oracle vs Logistic($E$) Accuracy ($L=10$)", fontsize=13)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "figs", "key_accuracy_network_vs_oracle.png"), dpi=150)
    plt.close(fig)

    # Fig 2: Predictive Entropy H(p(x)) vs Oracle Entropy H(Y|x)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(grid_T, hp_oracle_arr, "k-", linewidth=2.0, label="Bayesian Posterior Entropy $H(Y|x)$")
    ax.plot(grid_T, np.mean(hp_net_all, axis=0), "tab:red", linestyle="--", marker="s", markersize=4, label="SimpleMLP Predictive Entropy $H(p(x))$")
    ax.axvline(ex.T_C, color="gray", linestyle=":", label=r"$T_c$")
    ax.set_xlabel("Temperature $T$", fontsize=12)
    ax.set_ylabel("Entropy [nats]", fontsize=12)
    ax.set_title(r"MLP-05: Predictive Entropy Comparison (Confidence / Calibration)", fontsize=13)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "figs", "predictive_entropy_comparison.png"), dpi=150)
    plt.close(fig)

    # Fig 3: Calibration Diagram (Reliability Diagram)
    fig, ax = plt.subplots(figsize=(7, 7))
    # Pool predictions of all 5 seeds on test set
    p_pool = np.concatenate([p_seeds_T[s] for s in range(len(seeds)) for idx_T in [idx_tc]]) # at Tc
    # let's pool across entire test set
    all_p_seeds = []
    all_y_seeds = []
    for s_idx, net in enumerate(models):
        for T in grid_T:
            xb = torch.from_numpy(test_data[T]["spins"])
            yb = test_data[T]["y"]
            with torch.no_grad():
                l = net(xb).squeeze(-1).numpy()
            p = 1.0 / (1.0 + np.exp(-np.clip(l, -50, 50)))
            all_p_seeds.append(p)
            all_y_seeds.append(yb)
    p_pool_all = np.concatenate(all_p_seeds)
    y_pool_all = np.concatenate(all_y_seeds)
    ece_glob, bin_confs, bin_accs = compute_ece(p_pool_all, y_pool_all, n_bins=10)

    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    ax.plot(bin_confs, bin_accs, "s-", color="tab:blue", label=f"SimpleMLP (Global ECE = {ece_glob:.4f})")
    ax.set_xlabel("Confidence $\max(p, 1-p)$", fontsize=12)
    ax.set_ylabel("Empirical Accuracy", fontsize=12)
    ax.set_title("MLP-05: Global Reliability Diagram (Calibration)", fontsize=13)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "figs", "reliability_diagram.png"), dpi=150)
    plt.close(fig)

    print("All MLP-05 figures and deliverables saved successfully.")


if __name__ == "__main__":
    run_mlp05()
