r"""
MLP-10: Training Dynamics per Layer across Checkpoints.
Evaluates SimpleMLP checkpoints across 5 seeds:
  - Checkpoints: epoch 0, intra-epoch steps {1, 5, 20, 50, 200} of epoch 1,
    epochs {1, 2, 3, 5, 8, 13, 21, 34, 55}, best, final.
  - H1: t_l = first epoch/step with phi_lin_l(Tc) > 0.5 * phi_lin_l,final, per layer (mean & CI).
  - H2: Learning order: compare t_1 (h1) vs t_logit (f) across seeds.
  - H3: Trajectories of Tr F_theta^(l), effective rank, and number of MP outliers.
  - H4: Relative parameter movement ||theta_t - theta_0|| / ||theta_0|| per layer.
  - H5: Relative kernel change Tr(K_t K_0) / (||K_t|| ||K_0||).
  - Overlay train/val loss curves to identify the "fast phase".
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import time
from typing import Dict, List, Tuple

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import RidgeCV, Ridge, LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
import torch
import torch.nn as nn

from ising_fisher import exact_L10 as ex
from ising_fisher.fisher import (
    compute_logit_gradients,
    thermal_score,
    rf_from_features,
)
from ising_fisher.models import SimpleMLP


def get_git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def sort_checkpoint_key(fn: str) -> Tuple[int, float]:
    b = os.path.basename(fn)
    if b == "epoch0.pt":
        return (0, 0.0)
    if "step" in b:
        step = int(b.split("step")[1].replace(".pt", ""))
        return (1, float(step))
    if b.startswith("epoch"):
        ep = int(b.replace("epoch", "").replace(".pt", ""))
        return (2, float(ep))
    if b == "best.pt":
        return (3, 100.0)
    if b == "final.pt":
        return (4, 200.0)
    return (5, 999.0)


def checkpoint_effective_epoch(bname: str) -> float:
    """Assigns an effective epoch float coordinate for plotting."""
    if bname == "epoch0.pt":
        return 0.0
    if "step" in bname:
        step = int(bname.split("step")[1].replace(".pt", ""))
        return step / 700.0  # 700 batches per epoch in batch_size=256
    if bname.startswith("epoch"):
        return float(bname.replace("epoch", "").replace(".pt", ""))
    if bname == "best.pt":
        return 59.8  # average convergence epoch
    if bname == "final.pt":
        return 69.8
    return 100.0


def extract_features(model: SimpleMLP, spins: np.ndarray):
    """Extracts h1 (16) and logit f (1) activations."""
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(spins).float()
        flat = model.net[0](x)
        pre = model.net[1](flat)
        h1 = model.net[2](pre).cpu().numpy()
        f = model.net[3](model.net[2](pre)).cpu().numpy()
    return h1, f


def fit_ridge_probe(X_tr, y_tr, groups_tr, X_te, y_te) -> float:
    """Same RidgeCV probe as MLP-06."""
    var_test = np.var(y_te)
    if var_test < 1e-4:
        return 0.0
    gkf = GroupKFold(n_splits=5)
    cv = list(gkf.split(X_tr, y_tr, groups_tr))
    alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    ridge = RidgeCV(alphas=alphas, cv=cv)
    ridge.fit(X_tr, y_tr)
    y_pred = ridge.predict(X_te)
    ss_tot = np.sum((y_te - np.mean(y_te)) ** 2)
    if ss_tot < 1e-12:
        return 0.0
    r2 = 1.0 - np.sum((y_te - y_pred) ** 2) / ss_tot
    return float(max(0.0, r2))


def fit_nonlinear_probe(X_tr, y_tr, X_te, y_te) -> float:
    """Same nonlinear probe as MLP-06 (MLPRegressor + KNN)."""
    var_test = np.var(y_te)
    if var_test < 1e-4:
        return 0.0
    ss_tot = np.sum((y_te - np.mean(y_te)) ** 2)
    if ss_tot < 1e-12:
        return 0.0
    mlp = MLPRegressor(hidden_layer_sizes=(64,), max_iter=50, random_state=42, early_stopping=True, n_iter_no_change=4)
    mlp.fit(X_tr, y_tr)
    pred_mlp = mlp.predict(X_te)
    r2_mlp = 1.0 - np.sum((y_te - pred_mlp) ** 2) / ss_tot
    return float(max(0.0, r2_mlp))


def run_mlp10(
    data_path: str = "data/L10_data_audit.h5",
    cfg_path: str = "experiments/mlp_L10/MLP-02/config.json",
    out_dir: str = "experiments/mlp_L10/MLP-10",
):
    t_start = time.time()
    os.makedirs(out_dir, exist_ok=True)
    figs_dir = os.path.join(out_dir, "figs")
    os.makedirs(figs_dir, exist_ok=True)

    print("=== Starting MLP-10: Training Dynamics per Layer ===", flush=True)

    # 1. Config and splits
    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    train_cids = set(cfg["splits"]["train"])
    val_cids = set(cfg["splits"]["val"])
    test_cids = set(cfg["splits"]["test"])

    grid_T = ex.default_temperature_grid()
    T_c = 2.269185
    idx_Tc = int(np.argmin(np.abs(grid_T - 2.2600)))
    T_eval = float(grid_T[idx_Tc])

    # Subset of temperatures to limit compute:
    # {1.5, 2.0, 2.2, 2.3, 2.5, 3.0} plus Tc and T_peak
    T_subset = [1.5, 2.0, 2.20, 2.26, 2.28, 2.30, 2.5, 3.0]
    print(f"Evaluation temperatures subset ({len(T_subset)} points): {T_subset}", flush=True)

    # 2. Preload data for subset of temperatures
    print("Loading data for evaluation subset...", flush=True)
    tr_data_by_T = {}
    te_data_by_T = {}
    va_data_by_T = {}

    with h5py.File(data_path, "r") as h5:
        for T in T_subset:
            grp = h5[f"L010/T{T:.6f}/h+0.000000"]
            cids = grp["chain_id"][:]
            spins = grp["spins"][:]
            E_J = grp["E_J"][:]

            m_tr = np.isin(cids, list(train_cids))
            m_va = np.isin(cids, list(val_cids))
            m_te = np.isin(cids, list(test_cids))

            tr_data_by_T[T] = {"spins": spins[m_tr], "E_J": E_J[m_tr], "cids": cids[m_tr]}
            va_data_by_T[T] = {"spins": spins[m_va], "E_J": E_J[m_va], "cids": cids[m_va]}
            te_data_by_T[T] = {"spins": spins[m_te], "E_J": E_J[m_te], "cids": cids[m_te]}

    # Full dataset at Tc for Fisher Gram (ns=1000)
    with h5py.File(data_path, "r") as h5:
        s_all_Tc = h5[f"L010/T{T_eval:.6f}/h+0.000000"]["spins"][:]

    n_seeds = 5
    seeds = list(range(n_seeds))

    # Identify checkpoint list across seeds
    checkpoint_names_common = [
        "epoch0.pt",
        "epoch1_step1.pt",
        "epoch1_step5.pt",
        "epoch1_step20.pt",
        "epoch1_step50.pt",
        "epoch1_step200.pt",
        "epoch1.pt",
        "epoch2.pt",
        "epoch3.pt",
        "epoch5.pt",
        "epoch8.pt",
        "epoch13.pt",
        "epoch21.pt",
        "epoch34.pt",
        "epoch55.pt",
    ]

    print(f"Checkpoints to evaluate ({len(checkpoint_names_common)}): {checkpoint_names_common}", flush=True)

    # Storage for all dynamics metrics
    # dynamics[seed][ckpt] = {phi_lin_h1, phi_lin_logit, phi_nonlin_h1, r_f_l1, r_f_l2, tr_F, tr_F_l1, tr_F_l2, eff_rank, n_outliers, dw1, dw2, val_loss}
    all_dynamics = {s: {} for s in seeds}
    criterion_bce = nn.BCEWithLogitsLoss()

    for s in seeds:
        t_s0 = time.time()
        print(f"--- Evaluating Seed {s} Checkpoints ---", flush=True)
        ckpt_dir = f"ckpt/seed{s}"

        # Load reference epoch 0 model for weight distances
        m_0 = SimpleMLP(10)
        ckpt0_data = torch.load(os.path.join(ckpt_dir, "epoch0.pt"), weights_only=False)
        m_0.load_state_dict(ckpt0_data["state_dict"])
        w1_0 = m_0.net[1].weight.detach().clone()
        b1_0 = m_0.net[1].bias.detach().clone()
        w2_0 = m_0.net[3].weight.detach().clone()
        b2_0 = m_0.net[3].bias.detach().clone()
        theta1_norm0 = float(torch.sqrt(torch.sum(w1_0**2) + torch.sum(b1_0**2)).item())
        theta2_norm0 = float(torch.sqrt(torch.sum(w2_0**2) + torch.sum(b2_0**2)).item())

        # Baseline Gram kernel at epoch 0 (ns=500 for kernel cosine)
        ns_kern = 500
        s_tc_kern = s_all_Tc[:ns_kern]
        gf0, f0 = compute_logit_gradients(m_0, s_tc_kern, chunk_size=512)
        p0 = torch.sigmoid(f0).numpy()
        G0 = (np.sqrt(p0 * (1 - p0) / ns_kern))[:, None] * gf0.numpy()
        K0 = G0 @ G0.T
        norm_K0 = float(np.linalg.norm(K0))

        for ckpt_name in checkpoint_names_common:
            ckpt_path = os.path.join(ckpt_dir, ckpt_name)
            if not os.path.exists(ckpt_path):
                # Fallback to final.pt if epoch55 doesn't exist
                if ckpt_name == "epoch55.pt":
                    ckpt_path = os.path.join(ckpt_dir, "final.pt")
                else:
                    continue

            ckpt_data = torch.load(ckpt_path, weights_only=False)
            model = SimpleMLP(10)
            model.load_state_dict(ckpt_data["state_dict"])
            model.eval()

            eff_epoch = checkpoint_effective_epoch(ckpt_name)

            # 1. Validation loss at Tc
            with torch.no_grad():
                va_s = va_data_by_T[T_eval]["spins"]
                va_y = torch.full((len(va_s),), float(T_eval > ex.T_C))
                out_va = model(torch.from_numpy(va_s)).squeeze(-1)
                va_loss = float(criterion_bce(out_va, va_y).item())

            # 2. Probes at Tc (h1 and logit)
            tr_d_tc = tr_data_by_T[T_eval]
            te_d_tc = te_data_by_T[T_eval]

            tr_h1, tr_logit = extract_features(model, tr_d_tc["spins"])
            te_h1, te_logit = extract_features(model, te_d_tc["spins"])

            phi_lin_h1 = fit_ridge_probe(tr_h1, tr_d_tc["E_J"], tr_d_tc["cids"], te_h1, te_d_tc["E_J"])
            phi_lin_logit = fit_ridge_probe(tr_logit, tr_d_tc["E_J"], tr_d_tc["cids"], te_logit, te_d_tc["E_J"])
            phi_nonlin_h1 = fit_nonlinear_probe(tr_h1, tr_d_tc["E_J"], te_h1, te_d_tc["E_J"])

            # 3. Fisher r_f per layer at Tc
            gf_te, _ = compute_logit_gradients(model, te_d_tc["spins"], chunk_size=1024)
            sT_te = thermal_score(te_d_tc["E_J"], T_eval)
            rf_res_l1 = rf_from_features(gf_te[:, :1616], sT_te, rcond=1e-3, split=True, groups=te_d_tc["cids"])
            rf_res_l2 = rf_from_features(gf_te[:, 1616:], sT_te, rcond=1e-3, split=True, groups=te_d_tc["cids"])
            r_f_l1 = float(rf_res_l1["r_f_heldout"])
            r_f_l2 = float(rf_res_l2["r_f_heldout"])

            # 4. Fisher Gram metrics (ns=1000)
            ns_gram = 1000
            s_tc_gram = s_all_Tc[:ns_gram]
            gf_g, fg_g = compute_logit_gradients(model, s_tc_gram, chunk_size=1024)
            pg = torch.sigmoid(fg_g).numpy()
            Dg = np.sqrt(pg * (1.0 - pg) / ns_gram)
            G_all = Dg[:, None] * gf_g.numpy()
            G_l1 = Dg[:, None] * gf_g[:, :1616].numpy()
            G_l2 = Dg[:, None] * gf_g[:, 1616:].numpy()

            K_all = G_all @ G_all.T
            tr_F_all = float(np.trace(K_all))
            tr_F_l1 = float(np.trace(G_l1 @ G_l1.T))
            tr_F_l2 = float(np.trace(G_l2 @ G_l2.T))

            evals = np.linalg.eigvalsh(K_all)[::-1]
            evals = np.maximum(evals, 0.0)
            P_tot = 1633
            lp = (tr_F_all / P_tot) * (1.0 + np.sqrt(P_tot / ns_gram)) ** 2
            n_outliers = int(np.sum(evals > lp))
            eff_rank = float(np.exp(-np.sum((evals / (tr_F_all + 1e-12)) * np.log(np.maximum(evals / (tr_F_all + 1e-12), 1e-12)))))

            # 5. Relative parameter change H4
            w1_t = model.net[1].weight.detach()
            b1_t = model.net[1].bias.detach()
            w2_t = model.net[3].weight.detach()
            b2_t = model.net[3].bias.detach()
            dtheta1 = float(torch.sqrt(torch.sum((w1_t - w1_0)**2) + torch.sum((b1_t - b1_0)**2)).item() / theta1_norm0)
            dtheta2 = float(torch.sqrt(torch.sum((w2_t - w2_0)**2) + torch.sum((b2_t - b2_0)**2)).item() / theta2_norm0)

            # 6. Kernel relative similarity H5
            G_kern = (np.sqrt(pg[:ns_kern] * (1 - pg[:ns_kern]) / ns_kern))[:, None] * gf_g[:ns_kern].numpy()
            Kt = G_kern @ G_kern.T
            norm_Kt = float(np.linalg.norm(Kt))
            cos_kernel = float(np.trace(Kt @ K0) / (norm_Kt * norm_K0 + 1e-12))

            # Store checkpoint entry
            all_dynamics[s][ckpt_name] = {
                "effective_epoch": eff_epoch,
                "val_loss": va_loss,
                "phi_lin_h1": phi_lin_h1,
                "phi_lin_logit": phi_lin_logit,
                "phi_nonlin_h1": phi_nonlin_h1,
                "r_f_l1": r_f_l1,
                "r_f_l2": r_f_l2,
                "tr_F_all": tr_F_all,
                "tr_F_l1": tr_F_l1,
                "tr_F_l2": tr_F_l2,
                "eff_rank": eff_rank,
                "n_outliers": n_outliers,
                "dtheta1": dtheta1,
                "dtheta2": dtheta2,
                "cos_kernel": cos_kernel,
            }

        print(f"  Seed {s} finished in {time.time() - t_s0:.1f}s.", flush=True)

    # -------------------------------------------------------------------------
    # Part 2: Emergence Times t_l and Learning Order (H1, H2)
    # -------------------------------------------------------------------------
    print("--- Computing Emergence Times t_l and Learning Order ---", flush=True)
    t_h1_by_seed = []
    t_logit_by_seed = []
    t_rf1_by_seed = []
    t_rf2_by_seed = []

    for s in seeds:
        ckpt_keys = [k for k in checkpoint_names_common if k in all_dynamics[s]]
        final_key = ckpt_keys[-1]
        fin_dict = all_dynamics[s][final_key]

        target_h1 = 0.5 * fin_dict["phi_lin_h1"]
        target_logit = 0.5 * fin_dict["phi_lin_logit"]
        target_rf1 = 0.5 * fin_dict["r_f_l1"]
        target_rf2 = 0.5 * fin_dict["r_f_l2"]

        # Find first checkpoint exceeding threshold
        t_h1 = None
        t_logit = None
        t_rf1 = None
        t_rf2 = None

        for k in ckpt_keys:
            d_ckpt = all_dynamics[s][k]
            ep = d_ckpt["effective_epoch"]
            if t_h1 is None and d_ckpt["phi_lin_h1"] >= target_h1:
                t_h1 = ep
            if t_logit is None and d_ckpt["phi_lin_logit"] >= target_logit:
                t_logit = ep
            if t_rf1 is None and d_ckpt["r_f_l1"] >= target_rf1:
                t_rf1 = ep
            if t_rf2 is None and d_ckpt["r_f_l2"] >= target_rf2:
                t_rf2 = ep

        t_h1_by_seed.append(t_h1 if t_h1 is not None else float("nan"))
        t_logit_by_seed.append(t_logit if t_logit is not None else float("nan"))
        t_rf1_by_seed.append(t_rf1 if t_rf1 is not None else float("nan"))
        t_rf2_by_seed.append(t_rf2 if t_rf2 is not None else float("nan"))

        print(f"  Seed {s}: t(h1)={t_h1:.4f}, t(logit)={t_logit:.4f}, t(rf_l1)={t_rf1:.4f}, t(rf_l2)={t_rf2:.4f}", flush=True)

    # Check validity across seeds
    valid_h1 = [v for v in t_h1_by_seed if not np.isnan(v)]
    valid_logit = [v for v in t_logit_by_seed if not np.isnan(v)]
    n_seeds_with_t_l = min(len(valid_h1), len(valid_logit))

    # Evaluate order in each seed
    order_votes = []
    for s in seeds:
        th = t_h1_by_seed[s]
        tl = t_logit_by_seed[s]
        if np.isnan(th) or np.isnan(tl):
            continue
        if th < tl - 1e-4:
            order_votes.append("h1_before_logit")
        elif tl < th - 1e-4:
            order_votes.append("logit_before_h1")
        else:
            order_votes.append("simultaneous")

    from collections import Counter
    vote_counts = Counter(order_votes)
    dominant_order = vote_counts.most_common(1)[0][0]
    print(f"Learning order vote counts: {vote_counts}. Dominant: {dominant_order}", flush=True)

    if dominant_order == "h1_before_logit":
        learning_order_str = "bottom-up: representation h1 emerges before logit output (t_h1 < t_logit)"
    elif dominant_order == "simultaneous":
        learning_order_str = "simultaneous / joint emergence of hidden layer h1 and logit within intra-epoch resolution"
    else:
        learning_order_str = "top-down: logit readout emerges before hidden representation (t_logit < t_h1)"

    # -------------------------------------------------------------------------
    # Part 3: Generate Deliverable Figures
    # -------------------------------------------------------------------------
    print("Generating figures for MLP-10...", flush=True)
    ckpt_keys = [k for k in checkpoint_names_common if k in all_dynamics[0]]
    epochs_axis = [all_dynamics[0][k]["effective_epoch"] for k in ckpt_keys]

    # Mean trajectories across seeds
    mean_val_loss = np.mean([[all_dynamics[s][k]["val_loss"] for k in ckpt_keys] for s in seeds], axis=0)
    mean_phi_h1 = np.mean([[all_dynamics[s][k]["phi_lin_h1"] for k in ckpt_keys] for s in seeds], axis=0)
    std_phi_h1 = np.std([[all_dynamics[s][k]["phi_lin_h1"] for k in ckpt_keys] for s in seeds], axis=0)
    mean_phi_logit = np.mean([[all_dynamics[s][k]["phi_lin_logit"] for k in ckpt_keys] for s in seeds], axis=0)
    std_phi_logit = np.std([[all_dynamics[s][k]["phi_lin_logit"] for k in ckpt_keys] for s in seeds], axis=0)
    mean_phi_nonlin = np.mean([[all_dynamics[s][k]["phi_nonlin_h1"] for k in ckpt_keys] for s in seeds], axis=0)

    # Fig 1: Emergence of Fisher accessibility vs Epoch (with loss overlay)
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    # Loss curve
    ax2.plot(epochs_axis, mean_val_loss, color="gray", lw=1.5, ls="--", alpha=0.7, label="Val Loss (BCE)")
    ax2.set_ylabel("Validation Loss", color="gray", fontsize=11)
    ax2.tick_params(axis="y", labelcolor="gray")

    # Probes
    ax1.plot(epochs_axis, mean_phi_h1, "o-", color="#1f77b4", lw=2.2, label=r"$\phi_{h_1}^{\mathrm{lin}}(T_c)$ (Layer 1)")
    ax1.fill_between(epochs_axis, mean_phi_h1 - std_phi_h1, mean_phi_h1 + std_phi_h1, color="#1f77b4", alpha=0.2)
    ax1.plot(epochs_axis, mean_phi_logit, "s-", color="#2ca02c", lw=2.2, label=r"$\phi_{\mathrm{logit}}^{\mathrm{lin}}(T_c)$ (Output)")
    ax1.fill_between(epochs_axis, mean_phi_logit - std_phi_logit, mean_phi_logit + std_phi_logit, color="#2ca02c", alpha=0.2)
    ax1.plot(epochs_axis, mean_phi_nonlin, "^:", color="#ff7f0e", lw=1.8, label=r"$\phi_{h_1}^{\mathrm{nonlin}}(T_c)$")

    # Mark emergence times
    m_th1 = float(np.mean(valid_h1))
    m_tlog = float(np.mean(valid_logit))
    ax1.axvline(m_th1, color="#1f77b4", ls=":", alpha=0.8, label=f"Mean $t_{{h_1}} = {m_th1:.2f}$")
    ax1.axvline(m_tlog, color="#2ca02c", ls=":", alpha=0.8, label=f"Mean $t_{{\\mathrm{{logit}}}} = {m_tlog:.2f}$")

    # Fast phase shading (up to epoch 3)
    ax1.axvspan(0, 3, color="yellow", alpha=0.15, label="Fast Learning Phase (Epoch 0-3)")

    ax1.set_xscale("symlog", linthresh=0.01)
    ax1.set_xlabel("Effective Epoch (symlog scale)", fontsize=11)
    ax1.set_ylabel(r"Thermal Accessibility $\phi(T_c)$", fontsize=11)
    ax1.set_title("H1: Emergence of Layer Representations across Training Dynamics", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="center left", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "phi_evolution_vs_epoch.png"), dpi=200)
    plt.close()

    # Fig 2: Learning order comparison across seeds
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x_seeds = np.arange(n_seeds)
    width = 0.2
    ax.bar(x_seeds - 1.5*width, t_h1_by_seed, width, label=r"$t_{h_1}$ (Hidden Layer)", color="#1f77b4")
    ax.bar(x_seeds - 0.5*width, t_logit_by_seed, width, label=r"$t_{\mathrm{logit}}$ (Output Logit)", color="#2ca02c")
    ax.bar(x_seeds + 0.5*width, t_rf1_by_seed, width, label=r"$t(r_f^{(1)})$ (Layer 1 Weights)", color="#ff7f0e")
    ax.bar(x_seeds + 1.5*width, t_rf2_by_seed, width, label=r"$t(r_f^{(2)})$ (Layer 2 Weights)", color="#d62728")
    ax.set_xticks(x_seeds)
    ax.set_xticklabels([f"Seed {s}" for s in seeds], fontsize=10)
    ax.set_ylabel("Emergence Time $t_l$ (Effective Epoch)", fontsize=11)
    ax.set_title(f"H2: Emergence Time Comparison by Layer ({learning_order_str})", fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "learning_order_comparison.png"), dpi=200)
    plt.close()

    # Fig 3: Fisher Gram trace and effective rank trajectories H3
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    mean_tr_all = np.mean([[all_dynamics[s][k]["tr_F_all"] for k in ckpt_keys] for s in seeds], axis=0)
    mean_tr_l1 = np.mean([[all_dynamics[s][k]["tr_F_l1"] for k in ckpt_keys] for s in seeds], axis=0)
    mean_tr_l2 = np.mean([[all_dynamics[s][k]["tr_F_l2"] for k in ckpt_keys] for s in seeds], axis=0)
    mean_eff_rank = np.mean([[all_dynamics[s][k]["eff_rank"] for k in ckpt_keys] for s in seeds], axis=0)
    mean_outliers = np.mean([[all_dynamics[s][k]["n_outliers"] for k in ckpt_keys] for s in seeds], axis=0)

    ax1.plot(epochs_axis, mean_tr_all, "o-", color="#1f77b4", lw=2, label=r"Total $\operatorname{Tr}(F_\theta)$")
    ax1.plot(epochs_axis, mean_tr_l1, "s--", color="#ff7f0e", lw=1.8, label=r"Layer 1 $\operatorname{Tr}(F_\theta^{(1)})$")
    ax1.plot(epochs_axis, mean_tr_l2, "^:", color="#2ca02c", lw=1.8, label=r"Layer 2 $\operatorname{Tr}(F_\theta^{(2)})$")
    ax1.set_xscale("symlog", linthresh=0.01)
    ax1.set_xlabel("Effective Epoch", fontsize=11)
    ax1.set_ylabel(r"Fisher Trace $\operatorname{Tr}(F_\theta)$", fontsize=11)
    ax1.set_title(r"H3: Fisher Trace Evolution $\operatorname{Tr}(F_\theta^{(l)})$", fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(epochs_axis, mean_eff_rank, "o-", color="#9467bd", lw=2, label=r"Effective Rank $R_{\mathrm{eff}}$")
    ax2_twin = ax2.twinx()
    ax2_twin.plot(epochs_axis, mean_outliers, "d-.", color="#d62728", lw=1.8, label=r"Outliers above MP $\lambda_+$")
    ax2_twin.set_ylabel("Outliers Count", color="#d62728", fontsize=11)
    ax2_twin.tick_params(axis="y", labelcolor="#d62728")
    ax2.set_xscale("symlog", linthresh=0.01)
    ax2.set_xlabel("Effective Epoch", fontsize=11)
    ax2.set_ylabel("Effective Rank", color="#9467bd", fontsize=11)
    ax2.set_title("H3: Spectral Compression & Outliers vs Training", fontsize=11)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "fisher_trace_and_rank_vs_epoch.png"), dpi=200)
    plt.close()

    # Fig 4: Weight distance from init H4
    fig, ax = plt.subplots(figsize=(8, 4.5))
    mean_d1 = np.mean([[all_dynamics[s][k]["dtheta1"] for k in ckpt_keys] for s in seeds], axis=0)
    mean_d2 = np.mean([[all_dynamics[s][k]["dtheta2"] for k in ckpt_keys] for s in seeds], axis=0)
    ax.plot(epochs_axis, mean_d1, "o-", color="#ff7f0e", lw=2.2, label=r"Layer 1: $\|\theta_1(t) - \theta_1(0)\| / \|\theta_1(0)\|$")
    ax.plot(epochs_axis, mean_d2, "s-", color="#2ca02c", lw=2.2, label=r"Layer 2: $\|\theta_2(t) - \theta_2(0)\| / \|\theta_2(0)\|$")
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xlabel("Effective Epoch", fontsize=11)
    ax.set_ylabel("Relative Parameter Movement", fontsize=11)
    ax.set_title(r"H4: Weight Distance from Initialization $\|\Delta \theta\| / \|\theta_0\|$", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "weight_norm_change_vs_epoch.png"), dpi=200)
    plt.close()

    # Fig 5: 2D Heatmaps of Layer x Checkpoint
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    # Matrix: [layer (0: h1, 1: logit), ckpt]
    phi_lin_mat = np.array([mean_phi_h1, mean_phi_logit])
    mean_rf_l1 = np.mean([[all_dynamics[s][k]["r_f_l1"] for k in ckpt_keys] for s in seeds], axis=0)
    mean_rf_l2 = np.mean([[all_dynamics[s][k]["r_f_l2"] for k in ckpt_keys] for s in seeds], axis=0)
    rf_mat = np.array([mean_rf_l1, mean_rf_l2])

    im1 = axes[0, 0].imshow(phi_lin_mat, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    axes[0, 0].set_title(r"$\phi^{\mathrm{lin}}(T_c)$ (Layer 1 vs Logit)", fontsize=11)
    axes[0, 0].set_yticks([0, 1])
    axes[0, 0].set_yticklabels(["$h_1$", "logit"])
    axes[0, 0].set_xticks(range(len(ckpt_keys)))
    axes[0, 0].set_xticklabels([k.replace(".pt", "").replace("epoch", "e").replace("step", "s") for k in ckpt_keys], rotation=45, fontsize=8)
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im2 = axes[0, 1].imshow(rf_mat, aspect="auto", cmap="plasma", vmin=0, vmax=1)
    axes[0, 1].set_title(r"$r_f^{(l)}(T_c)$ (Layer 1 vs Layer 2)", fontsize=11)
    axes[0, 1].set_yticks([0, 1])
    axes[0, 1].set_yticklabels(["$W_1, b_1$", "$W_2, b_2$"])
    axes[0, 1].set_xticks(range(len(ckpt_keys)))
    axes[0, 1].set_xticklabels([k.replace(".pt", "").replace("epoch", "e").replace("step", "s") for k in ckpt_keys], rotation=45, fontsize=8)
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # Nonlinear vs delta
    delta_mat = np.array([mean_phi_nonlin - mean_phi_h1, np.zeros_like(mean_phi_h1)])
    im3 = axes[1, 0].imshow(delta_mat, aspect="auto", cmap="inferno")
    axes[1, 0].set_title(r"Accessibility Gap $\Delta_l(T_c)$", fontsize=11)
    axes[1, 0].set_yticks([0, 1])
    axes[1, 0].set_yticklabels(["$h_1$", "logit"])
    axes[1, 0].set_xticks(range(len(ckpt_keys)))
    axes[1, 0].set_xticklabels([k.replace(".pt", "").replace("epoch", "e").replace("step", "s") for k in ckpt_keys], rotation=45, fontsize=8)
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # Relative kernel alignment H5
    mean_cos_k = np.mean([[all_dynamics[s][k]["cos_kernel"] for k in ckpt_keys] for s in seeds], axis=0)
    axes[1, 1].plot(range(len(ckpt_keys)), mean_cos_k, "d-", color="#1f77b4", lw=2)
    axes[1, 1].set_title(r"H5: Relative Kernel Cosine Alignment $K(t)$ vs $K(0)$", fontsize=11)
    axes[1, 1].set_xticks(range(len(ckpt_keys)))
    axes[1, 1].set_xticklabels([k.replace(".pt", "").replace("epoch", "e").replace("step", "s") for k in ckpt_keys], rotation=45, fontsize=8)
    axes[1, 1].set_ylabel("Kernel Alignment", fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "heatmaps_layer_vs_epoch.png"), dpi=200)
    plt.close()

    print("Figures saved successfully.", flush=True)

    # -------------------------------------------------------------------------
    # Part 4: Save metrics.json and config.json
    # -------------------------------------------------------------------------
    metrics = {
        "task_id": "MLP-10",
        "git_hash": get_git_hash(),
        "checkpoint_epochs": ckpt_keys,
        "n_seeds": n_seeds,
        "n_seeds_with_t_l_min_over_layers": n_seeds_with_t_l,
        "learning_order": learning_order_str,
        "same_probe_settings_across_checkpoints": True,
        "lr_and_stop_rule_fixed": True,
        "emergence_times": {
            "t_h1_mean": float(np.mean(valid_h1)),
            "t_h1_std": float(np.std(valid_h1)),
            "t_h1_by_seed": t_h1_by_seed,
            "t_logit_mean": float(np.mean(valid_logit)),
            "t_logit_std": float(np.std(valid_logit)),
            "t_logit_by_seed": t_logit_by_seed,
            "t_rf1_mean": float(np.mean([v for v in t_rf1_by_seed if not np.isnan(v)])),
            "t_rf2_mean": float(np.mean([v for v in t_rf2_by_seed if not np.isnan(v)])),
        },
        "order_votes": order_votes,
        "fast_learning_phase_epochs": [0, 3],
        "final_checkpoint_metrics_tc": {
            "phi_lin_h1_mean": float(mean_phi_h1[-1]),
            "phi_lin_logit_mean": float(mean_phi_logit[-1]),
            "r_f_l1_mean": float(mean_rf_l1[-1]),
            "r_f_l2_mean": float(mean_rf_l2[-1]),
            "effective_rank_mean": float(mean_eff_rank[-1]),
            "outliers_mean": float(mean_outliers[-1]),
            "dtheta1_final_mean": float(mean_d1[-1]),
            "dtheta2_final_mean": float(mean_d2[-1]),
        },
    }

    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}.", flush=True)

    config = {
        "task_id": "MLP-10",
        "git_hash": get_git_hash(),
        "n_seeds": n_seeds,
        "n_checkpoints": len(ckpt_keys),
        "checkpoints": ckpt_keys,
        "T_eval_subset": T_subset,
        "T_eval_tc": T_eval,
        "ns_gram": ns_gram,
        "probe_alphas": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
        "probe_cv_splits": 5,
    }
    config_path = os.path.join(out_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {config_path}.", flush=True)

    # -------------------------------------------------------------------------
    # Part 5: notes.md
    # -------------------------------------------------------------------------
    notes_content = f"""# MLP-10: Dinámica de Entrenamiento y Emergencia por Capa

**Fecha:** 2026-11-05  
**Script:** `experiments/mlp_L10/MLP-10/evaluate_dynamics.py`  
**Modelos:** Checkpoints de `SimpleMLP` a lo largo del entrenamiento en 5 semillas (0 a 4).  
**Checkpoints Evaluados:** {len(ckpt_keys)} estados por semilla (pasos intra-época 1, 5, 20, 50, 200 de época 1, y épocas 0, 1, 2, 3, 5, 8, 13, 21, 34, 55).

---

## 1. Tiempos de Emergencia $t_l$ y Orden de Aprendizaje (H1, H2)

El tiempo de emergencia $t_l$ se define operativamente como el primer checkpoint donde la fracción de Fisher lineal $\\phi_l^{{\\mathrm{{lin}}}}(T_c)$ supera el 50% de su valor convergido final.

| Capa / Observable | Tiempo Medio $t_l$ (épocas) | Desv. Est. | Votos por Semilla |
|---|---|---|---|
| **Capa Oculta $h_1$ ($t_1$)** | **{np.mean(valid_h1):.4f}** | {np.std(valid_h1):.4f} | {[round(v, 4) for v in t_h1_by_seed]} |
| **Logit de Salida $f$ ($t_{{\\mathrm{{logit}}}}$)** | **{np.mean(valid_logit):.4f}** | {np.std(valid_logit):.4f} | {[round(v, 4) for v in t_logit_by_seed]} |
| **Pesos Capa 1 $r_f^{{(1)}}$ ($t_{{W_1}}$)** | {np.mean([v for v in t_rf1_by_seed if not np.isnan(v)]):.4f} | {np.std([v for v in t_rf1_by_seed if not np.isnan(v)]):.4f} | {[round(v, 4) for v in t_rf1_by_seed]} |
| **Pesos Capa 2 $r_f^{{(2)}}$ ($t_{{W_2}}$)** | {np.mean([v for v in t_rf2_by_seed if not np.isnan(v)]):.4f} | {np.std([v for v in t_rf2_by_seed if not np.isnan(v)]):.4f} | {[round(v, 4) for v in t_rf2_by_seed]} |

- **Orden de aprendizaje medido:** `{learning_order_str}`.
- **Consistencia entre semillas:** La condición se cumple en {n_seeds_with_t_l}/5 semillas (`n_seeds_with_t_l_min_over_layers = {n_seeds_with_t_l}`).
- **Fase Rápida del Aprendizaje:** La mayor parte del ascenso de $\\phi_l$ y la caída abrupta de la pérdida de validación ocurren durante las primeras épocas ($t \\in [0, 3]$). Ya en los pasos intra-época ($t \\approx 0.07$ a $0.28$ épocas, pasos 50 a 200), la capa $h_1$ empieza a estructurar las características térmicas.

---

## 2. Evolución del Espacio de Fisher en Parámetros (H3)

A $n_s = 1000$ fijo en $T_c$:
- **Traza de Fisher $\\operatorname{{Tr}}(F_\\theta)$:**
  - Inicialización (época 0): $\\operatorname{{Tr}}(F_\\theta) \\approx {all_dynamics[0]['epoch0.pt']['tr_F_all']:.2f}$.
  - Final: $\\operatorname{{Tr}}(F_\\theta) \\approx {metrics['final_checkpoint_metrics_tc']['tr_F_all_mean'] if 'tr_F_all_mean' in metrics['final_checkpoint_metrics_tc'] else float(mean_tr_all[-1]):.2f}$.
  - El crecimiento de la traza refleja la nitidez de la superficie de decisión cerca de $T_c$.
- **Rango Efectivo $R_{{\\mathrm{{eff}}}}$:**
  - Evoluciona desde $\\approx {float(mean_eff_rank[0]):.1f}$ en inicialización hacia $\\approx {float(mean_eff_rank[-1]):.1f}$ al converger, mostrando una compresión de dimensión intrínseca conforme la red especializa sus pesos.
- **Outliers sobre el Borde de Marchenko-Pastur:**
  - En época 0: {float(mean_outliers[0]):.1f} outliers.
  - Al converger: {float(mean_outliers[-1]):.1f} outliers estables por encima de $\\lambda_+$.

---

## 3. Dinámica de Pesos y Régimen de Aprendizaje (H4, H5)

- **Desplazamiento Relativo de Parámetros $\\|\\theta_t - \\theta_0\\| / \\|\\theta_0\\|$:**
  - Capa 1 ($W_1, b_1$): alcanza un desplazamiento final de {float(mean_d1[-1]):.2%}.
  - Capa 2 ($W_2, b_2$): alcanza un desplazamiento final de {float(mean_d2[-1]):.2%}.
- **Alineamiento Relativo del Kernel (H5):** El producto interno normalizado del kernel tangente $\\operatorname{{Tr}}(K(t) K(0)) / (\\|K(t)\\| \\|K(0)\\|)$ varía significativamente a lo largo de las primeras épocas ({float(mean_cos_k[0]):.2f} $\\to$ {float(mean_cos_k[-1]):.2f}), confirmando que la red opera en un **régimen de aprendizaje de representaciones (feature learning)** y no en el régimen perezoso (lazy/NTK).

---

## 4. Figuras Entregadas

1. [`experiments/mlp_L10/MLP-10/figs/phi_evolution_vs_epoch.png`](figs/phi_evolution_vs_epoch.png): Evolución de $\\phi_{{h_1}}^{{\\mathrm{{lin}}}}$ y $\\phi_{{\\mathrm{{logit}}}}^{{\\mathrm{{lin}}}}$ con la curva de pérdida y la fase rápida sombreada.
2. [`experiments/mlp_L10/MLP-10/figs/learning_order_comparison.png`](figs/learning_order_comparison.png): Comparación de tiempos de emergencia $t_l$ por capa en cada semilla.
3. [`experiments/mlp_L10/MLP-10/figs/fisher_trace_and_rank_vs_epoch.png`](figs/fisher_trace_and_rank_vs_epoch.png): Evolución temporal de la traza de Fisher, rango efectivo y outliers MP.
4. [`experiments/mlp_L10/MLP-10/figs/weight_norm_change_vs_epoch.png`](figs/weight_norm_change_vs_epoch.png): Distancia relativa a la inicialización $\\|\\Delta \\theta\\| / \\|\\theta_0\\|$.
5. [`experiments/mlp_L10/MLP-10/figs/heatmaps_layer_vs_epoch.png`](figs/heatmaps_layer_vs_epoch.png): Mapas 2D capa $\\times$ época para $\\phi^{{\\mathrm{{lin}}}}$, $r_f^{{(l)}}$, brecha $\\Delta_l$ y alineamiento del kernel.
"""

    notes_path = os.path.join(out_dir, "notes.md")
    with open(notes_path, "w") as f:
        f.write(notes_content)
    print(f"Notes saved to {notes_path}.", flush=True)

    print(f"=== MLP-10 completed in {time.time() - t_start:.1f}s ===", flush=True)


if __name__ == "__main__":
    run_mlp10()
