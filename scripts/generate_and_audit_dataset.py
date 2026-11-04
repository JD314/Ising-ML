"""
Generation and audit of Monte Carlo dataset for 2D Ising (L=10) across phase transition.
Task MLP-02 for Ising-ML.
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
from scipy import stats

from ising_fisher import exact_L10 as ex
from ising_fisher import mc


def estimate_tau_int(x: np.ndarray, max_lag: int = 50) -> float:
    """Integrated autocorrelation time using automated Sokal windowing."""
    n = len(x)
    x = x - np.mean(x)
    var = np.var(x)
    if var == 0 or n <= 1:
        return 0.5
    max_lag = min(max_lag, n - 1)
    cf = np.correlate(x, x, mode="full")
    lags = np.arange(max_lag + 1)
    autocorr = cf[n - 1:n + max_lag] / (var * (n - lags))

    tau = 0.5
    for t in range(1, len(autocorr)):
        if autocorr[t] <= 0:
            break
        tau += autocorr[t]
        if t >= 6 * tau:
            break
    return float(max(0.5, tau))


def run_mlp02(
    L: int = 10,
    n_chains: int = 20,
    n_samples: int = 200,
    n_burn: int = 2000,
    thin: int = 20,
    master_seed: int = 42,
    out_h5: str = "data/L10_data_audit.h5",
    exp_dir: str = "experiments/mlp_L10/MLP-02",
):
    os.makedirs(os.path.dirname(os.path.abspath(out_h5)), exist_ok=True)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "figs"), exist_ok=True)

    grid_T = ex.default_temperature_grid()
    K = len(grid_T)
    print(f"Starting MLP-02 generation: L={L}, {K} temperatures, {n_chains} chains, {n_samples} samples/chain (thin={thin}, burn={n_burn})")

    # Fixed chain split across all temperatures: 14 train (70%), 3 val (15%), 3 test (15%)
    rng_split = np.random.default_rng(master_seed)
    chain_perm = rng_split.permutation(n_chains).tolist()
    split_chains = {
        "train": chain_perm[:14],
        "val": chain_perm[14:17],
        "test": chain_perm[17:20],
    }

    # Prepare exact reference
    exact_ref = ex.compute_oracle_grid(L=L, grid_T=grid_T)

    # Containers for metrics
    audit_results = {
        "T": [],
        "mean_E_mc": [],
        "mean_E_exact": [],
        "var_E_mc": [],
        "var_E_exact": [],
        "z_score_E": [],
        "p_val_z_E": [],
        "frac_M_pos_orig": [],
        "frac_M_pos_randomized": [],
        "all_up_orig_count": [],
        "tau_int_E": [],
        "S_0": [],
    }

    config = {
        "task_id": "MLP-02",
        "L": L,
        "N": L * L,
        "n_chains": n_chains,
        "n_samples_per_chain": n_samples,
        "n_samples_total_per_T": n_chains * n_samples,
        "n_burn_sweeps": n_burn,
        "thin_sweeps": thin,
        "master_seed": master_seed,
        "temperature_grid": grid_T.tolist(),
        "splits": split_chains,
        "h5_file": out_h5,
    }

    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Open HDF5
    h5_file = h5py.File(out_h5, "w")
    h5_file.attrs.update({
        "L": int(L),
        "master_seed": int(master_seed),
        "n_chains": int(n_chains),
        "n_samples": int(n_samples),
        "n_burn": int(n_burn),
        "thin": int(thin),
    })

    t0_all = time.time()
    for idx_T, T in enumerate(grid_T):
        t0 = time.time()
        # Seed per T
        seed_T = hash((master_seed, round(T, 6))) % (2 ** 32)
        spins_orig, E_orig, M_orig, cid = mc.run_chains(
            L, T, h=0.0, J=1.0, n_chains=n_chains, n_burn=n_burn, n_samples=n_samples, thin=thin, seed=seed_T
        )
        total_samples = len(E_orig)

        # Record original sign statistics
        frac_M_pos_orig = float(np.mean(M_orig > 0))
        all_up_count = int(np.sum(M_orig == (L * L)))

        # Z2 Randomization: global flip with p=0.5 per sample
        rng_z2 = np.random.default_rng(seed_T + 777)
        flip_signs = rng_z2.choice(np.array([-1, 1], dtype=np.int8), size=total_samples)

        # Applied to spins and M (E is even under Z2 so E remains unchanged)
        spins_rand = spins_orig * flip_signs[:, None, None]
        M_rand = M_orig * flip_signs.astype(np.int32)
        frac_M_pos_rand = float(np.mean(M_rand > 0))

        # Measure tau_int(E) across chains
        tau_list = []
        for c in range(n_chains):
            mask_c = (cid == c)
            tau_list.append(estimate_tau_int(E_orig[mask_c]))
        tau_int_mean = float(np.mean(tau_list))

        # Structure factor S(0) = < M^2 > / L^2
        S_0 = float(np.mean(M_rand.astype(np.float64) ** 2) / (L * L))

        # Exact comparison
        mE_exact = float(exact_ref["mean_E"][idx_T])
        varE_exact = float(exact_ref["var_E"][idx_T])
        mE_mc = float(np.mean(E_orig))
        varE_mc = float(np.var(E_orig))

        # Standard error of mean accounting for number of samples:
        if varE_exact > 1e-12:
            se = np.sqrt(varE_exact / total_samples)
            z_E = float((mE_mc - mE_exact) / se)
            p_val_z = float(2.0 * (1.0 - stats.norm.cdf(abs(z_E))))
        else:
            # Low T limit (frozen ground state)
            diff = abs(mE_mc - mE_exact)
            z_E = 0.0 if diff < 1e-6 else float(diff)
            p_val_z = 1.0 if diff < 1e-6 else 0.0

        audit_results["T"].append(float(T))
        audit_results["mean_E_mc"].append(mE_mc)
        audit_results["mean_E_exact"].append(mE_exact)
        audit_results["var_E_mc"].append(varE_mc)
        audit_results["var_E_exact"].append(varE_exact)
        audit_results["z_score_E"].append(z_E)
        audit_results["p_val_z_E"].append(p_val_z)
        audit_results["frac_M_pos_orig"].append(frac_M_pos_orig)
        audit_results["frac_M_pos_randomized"].append(frac_M_pos_rand)
        audit_results["all_up_orig_count"].append(all_up_count)
        audit_results["tau_int_E"].append(tau_int_mean)
        audit_results["S_0"].append(S_0)

        # Save to HDF5
        key = f"L{L:03d}/T{T:.6f}/h+0.000000"
        grp = h5_file.create_group(key)
        grp.create_dataset("spins", data=spins_rand, compression="gzip", chunks=True)
        grp.create_dataset("spins_orig", data=spins_orig, compression="gzip", chunks=True)
        grp.create_dataset("sign_original", data=flip_signs)
        grp.create_dataset("E_J", data=E_orig)
        grp.create_dataset("M", data=M_rand)
        grp.create_dataset("M_orig", data=M_orig)
        grp.create_dataset("chain_id", data=cid)
        grp.attrs.update({"L": int(L), "T": float(T), "h": 0.0, "J": 1.0, "seed": seed_T})

        dt = time.time() - t0
        if (idx_T + 1) % 16 == 0 or idx_T == K - 1:
            print(f"  [{idx_T+1}/{K}] T={T:.4f}  <E>_mc={mE_mc:.2f} (<E>_ex={mE_exact:.2f}, z={z_E:+.2f})  sign_rand={frac_M_pos_rand:.3f}  tau_int={tau_int_mean:.2f} ({dt:.1f}s)")

    h5_file.close()
    print(f"Data generation and saving completed in {time.time() - t0_all:.1f}s.")

    # Holm-Bonferroni correction on p-values
    p_vals = np.array(audit_results["p_val_z_E"])
    sorted_indices = np.argsort(p_vals)
    holm_rejected = np.zeros(K, dtype=bool)
    alpha = 0.01
    for rank, idx in enumerate(sorted_indices):
        p_thresh = alpha / (K - rank)
        if p_vals[idx] < p_thresh:
            holm_rejected[idx] = True
        else:
            break

    n_rejected = int(np.sum(holm_rejected))
    frac_rejected = float(n_rejected / K)
    print(f"Holm-Bonferroni p-value test: {n_rejected}/{K} rejected at alpha=0.01 ({frac_rejected*100:.1f}%)")

    # Bayesian oracle accuracy on generated data
    print("Evaluating optimal Bayesian oracle accuracy on full generated dataset...")
    with h5py.File(out_h5, "r") as f:
        E_all = []
        y_true_all = []
        for T in grid_T:
            key = f"L{L:03d}/T{T:.6f}/h+0.000000"
            e = f[key]["E_J"][()]
            E_all.append(e)
            y_true_all.append(np.full(len(e), int(T > ex.T_C)))
        E_all = np.concatenate(E_all)
        y_true_all = np.concatenate(y_true_all)

    p_star = ex.optimal_bayesian_classifier(E_all, grid_T, exact_ref["ln_Z"])
    y_pred = (p_star >= 0.5).astype(int)
    oracle_acc = float(np.mean(y_pred == y_true_all))
    print(f"Bayesian oracle global accuracy: {oracle_acc:.4f}")

    # Accuracy at Tc
    idx_tc = np.argmin(np.abs(grid_T - ex.T_C))
    key_tc = f"L{L:03d}/T{grid_T[idx_tc]:.6f}/h+0.000000"
    with h5py.File(out_h5, "r") as f:
        e_tc = f[key_tc]["E_J"][()]
    p_star_tc = ex.optimal_bayesian_classifier(e_tc, grid_T, exact_ref["ln_Z"])
    oracle_acc_tc = float(np.mean((p_star_tc >= 0.5) == int(grid_T[idx_tc] > ex.T_C)))
    print(f"Bayesian oracle accuracy at T={grid_T[idx_tc]:.4f} (~Tc): {oracle_acc_tc:.4f}")

    # Mutual information recalculated with actual dataset
    mi_actual = ex.compute_mutual_information(E_all, grid_T, exact_ref["ln_Z"])
    print(f"Actual dataset I(T;X)={mi_actual['I_T_X']:.4f}, I(Y;X)={mi_actual['I_Y_X']:.4f}")

    # Prepare metrics.json
    metrics = {
        "task_id": "MLP-02",
        "L": L,
        "N": L * L,
        "n_temperatures": K,
        "total_samples": K * n_chains * n_samples,
        "acceptance_criteria": {
            "max_frac_rejected_p_01": 0.10,
            "actual_frac_rejected": frac_rejected,
            "passed_energy_audit": bool(frac_rejected <= 0.10),
            "sign_randomized_mean": float(np.mean(audit_results["frac_M_pos_randomized"])),
            "passed_sign_balance": bool(abs(np.mean(audit_results["frac_M_pos_randomized"]) - 0.5) <= 0.02),
        },
        "oracle_metrics": {
            "global_bayesian_accuracy": oracle_acc,
            "bayesian_accuracy_at_Tc": oracle_acc_tc,
            "T_near_Tc": float(grid_T[idx_tc]),
            "I_T_X": mi_actual["I_T_X"],
            "I_Y_X": mi_actual["I_Y_X"],
        },
        "summary": audit_results,
    }

    with open(os.path.join(exp_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # ---------------- FIGURES ----------------
    # Fig 1: P-values and Z-scores per temperature
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax1.plot(grid_T, audit_results["z_score_E"], "o-", color="tab:blue", markersize=4)
    ax1.axhline(0, color="k", linestyle="--", alpha=0.5)
    ax1.axhline(2.576, color="tab:red", linestyle=":", label=r"$p = 0.01$ (unadjusted)")
    ax1.axhline(-2.576, color="tab:red", linestyle=":")
    ax1.axvline(ex.T_C, color="gray", linestyle=":", label=r"$T_c$")
    ax1.set_ylabel(r"Z-score $\frac{\langle E \rangle_{\text{mc}} - \langle E \rangle_{\text{exact}}}{\sigma / \sqrt{N}}$")
    ax1.set_title(r"MLP-02 Data Audit: Energy Consistency vs Exact Oracle ($L=10$)")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=9)

    ax2.semilogy(grid_T, np.clip(audit_results["p_val_z_E"], 1e-15, 1.0), "s-", color="tab:purple", markersize=4, label="p-value (Z-test)")
    ax2.axhline(0.01, color="tab:red", linestyle=":", label=r"$\alpha = 0.01$")
    ax2.axvline(ex.T_C, color="gray", linestyle=":")
    ax2.set_xlabel("Temperature $T$")
    ax2.set_ylabel("p-value")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(exp_dir, "figs", "p_values_energy_audit.png"), dpi=150)
    plt.close(fig)

    # Fig 2: Autocorrelation time and Structure factor S(0)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax1.plot(grid_T, audit_results["tau_int_E"], "o-", color="tab:green", markersize=4)
    ax1.axvline(ex.T_C, color="gray", linestyle=":", label=r"$T_c$")
    ax1.set_ylabel(r"$\tau_{\text{int}}(E)$ [sweeps]")
    ax1.set_title(r"Integrated Autocorrelation Time & Structure Factor $S(0)$")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.plot(grid_T, audit_results["S_0"], "^-", color="tab:orange", markersize=4, label=r"$S(0) = \langle M^2 \rangle / L^2$")
    ax2.axvline(ex.T_C, color="gray", linestyle=":", label=r"$T_c$")
    ax2.set_xlabel("Temperature $T$")
    ax2.set_ylabel(r"$S(0)$")
    ax2.grid(alpha=0.3)
    ax2.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(exp_dir, "figs", "tau_int_and_S0.png"), dpi=150)
    plt.close(fig)

    # Fig 3: Sign balance before and after randomization
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(grid_T, audit_results["frac_M_pos_orig"], "o--", color="tab:blue", alpha=0.6, label="Original MC (hot start)")
    ax.plot(grid_T, audit_results["frac_M_pos_randomized"], "s-", color="tab:red", label=r"Randomized $\mathbb{Z}_2$")
    ax.axhline(0.5, color="k", linestyle="--", label="Exact balance (0.5)")
    ax.axhspan(0.48, 0.52, color="gray", alpha=0.2, label=r"Tolerance $\pm 0.02$")
    ax.axvline(ex.T_C, color="gray", linestyle=":", label=r"$T_c$")
    ax.set_xlabel("Temperature $T$")
    ax.set_ylabel("Fraction $M > 0$")
    ax.set_title(r"MLP-02 Sign Balance Verification ($\mathbb{Z}_2$ Parity)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(exp_dir, "figs", "sign_balance.png"), dpi=150)
    plt.close(fig)

    print("All figures and metrics saved in", exp_dir)


if __name__ == "__main__":
    run_mlp02()
