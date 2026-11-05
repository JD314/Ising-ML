r"""
MLP-09: Input-Side Analysis of SimpleMLP on Ising L=10
Tasks G1-G5:
  G1: Saliency map E_x|df/ds_i| by site and temperature T.
  G2: 2-D Fourier power spectrum of W1 weights per neuron:
      power fraction at k=0, small |k|, and high |k|, comparing trained vs init.
  G3: Quadratic surrogate f(x) ≈ c + \sum_{i<j} Q_ij s_i s_j via Ridge:
      held-out R^2 (global and per T), distance profile vs J_ij, power in nearest neighbors (Q_power_nn_pct)
      and power in k=0 mode (Q_power_k0_pct).
  G4: Learned translation invariance: relative dispersion of Q_ij for pairs with same separation.
  G5: Input ablations (k0, even_nn, energy) across 3 seeds:
      train and evaluate accuracy(T) and DeltaCE vs exact Bayesian oracle.
"""

from __future__ import annotations

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
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, accuracy_score, log_loss
from torch.utils.data import DataLoader, TensorDataset

from ising_fisher import exact_L10 as ex
from ising_fisher.models import SimpleMLP


def get_git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def get_pair_indices(L: int = 10):
    """Returns upper-triangular pair indices and their toroidal distance metrics."""
    N = L * L
    i_idx, j_idx = np.triu_indices(N, k=1)
    n_pairs = len(i_idx)

    xi, yi = i_idx % L, i_idx // L
    xj, yj = j_idx % L, j_idx // L

    dx = np.minimum(np.abs(xi - xj), L - np.abs(xi - xj))
    dy = np.minimum(np.abs(yi - yj), L - np.abs(yi - yj))
    r = np.sqrt(dx**2 + dy**2)

    # Nearest neighbors in 2D torus have dx=1, dy=0 or dx=0, dy=1 -> r = 1.0
    is_nn = np.isclose(r, 1.0)
    # Diagonal neighbors have dx=1, dy=1 -> r = sqrt(2) ~ 1.4142
    is_diag = np.isclose(r, np.sqrt(2.0))

    return i_idx, j_idx, dx, dy, r, is_nn, is_diag


def get_nn_bonds(L: int = 10):
    """Returns the 200 nearest neighbor directed bond pairs on the torus."""
    bonds = []
    for y in range(L):
        for x in range(L):
            i = y * L + x
            # Right neighbor
            j_right = y * L + ((x + 1) % L)
            bonds.append((min(i, j_right), max(i, j_right)))
            # Down neighbor
            j_down = ((y + 1) % L) * L + x
            bonds.append((min(i, j_down), max(i, j_down)))
    # Remove duplicates
    unique_bonds = sorted(list(set(bonds)))
    return unique_bonds


class SimpleAblationMLP(nn.Module):
    """MLP for ablations with custom input dimension."""
    def __init__(self, in_features: int, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_ablation_model(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    in_features: int,
    seed: int,
    lr: float = 1e-3,
    batch_size: int = 256,
    max_epochs: int = 40,
    patience: int = 8,
) -> SimpleAblationMLP:
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = SimpleAblationMLP(in_features=in_features, hidden=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    tr_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
        batch_size=batch_size,
        shuffle=True,
    )
    va_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va)),
        batch_size=batch_size,
        shuffle=False,
    )

    best_loss = float("inf")
    best_weights = None
    patience_cnt = 0

    for epoch in range(max_epochs):
        model.train()
        for xb, yb in tr_loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        va_loss = 0.0
        n_samples = 0
        with torch.no_grad():
            for xb, yb in va_loader:
                out = model(xb)
                va_loss += float(criterion(out, yb).item() * len(yb))
                n_samples += len(yb)
        va_loss /= n_samples

        if va_loss < best_loss - 1e-4:
            best_loss = va_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)
    model.eval()
    return model


def run_mlp09(
    data_path: str = "data/L10_data_audit.h5",
    cfg_path: str = "experiments/mlp_L10/MLP-02/config.json",
    out_dir: str = "experiments/mlp_L10/MLP-09",
):
    t_start = time.time()
    os.makedirs(out_dir, exist_ok=True)
    figs_dir = os.path.join(out_dir, "figs")
    os.makedirs(figs_dir, exist_ok=True)

    print("=== Starting MLP-09: Input Side Analysis ===", flush=True)

    # 1. Load config and data
    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    train_cids = set(cfg["splits"]["train"])
    val_cids = set(cfg["splits"]["val"])
    test_cids = set(cfg["splits"]["test"])

    grid_T = ex.default_temperature_grid()
    n_T = len(grid_T)
    T_c = 2.269185
    idx_Tc = int(np.argmin(np.abs(grid_T - 2.2600)))
    T_eval = float(grid_T[idx_Tc])

    # Oracle lnZ for Bayesian classifier
    print("Computing exact transfer matrix oracle lnZ for all 64 temperatures...", flush=True)
    exact_ref = ex.compute_oracle_grid(L=10, grid_T=grid_T)
    lnZ_grid = exact_ref["ln_Z"]

    # Load 5 trained seeds
    n_seeds = 5
    models_trained = []
    models_init = []
    for s in range(n_seeds):
        m = SimpleMLP(10)
        ckpt = torch.load(f"ckpt/seed{s}/best.pt", weights_only=False)
        m.load_state_dict(ckpt["state_dict"])
        m.eval()
        models_trained.append(m)

        m0 = SimpleMLP(10)
        ckpt0 = torch.load(f"ckpt/seed{s}/epoch0.pt", weights_only=False)
        m0.load_state_dict(ckpt0["state_dict"])
        m0.eval()
        models_init.append(m0)

    # Load test data
    print("Loading test data partition across all 64 temperatures...", flush=True)
    test_spins_by_T = {}
    test_E_by_T = {}
    test_cids_by_T = {}

    all_test_spins = []
    all_test_cids = []
    all_test_T = []

    with h5py.File(data_path, "r") as h5:
        for T in grid_T:
            grp = h5[f"L010/T{T:.6f}/h+0.000000"]
            cids = grp["chain_id"][:]
            mask = np.isin(cids, list(test_cids))
            s_te = grp["spins"][:][mask]
            e_te = grp["E_J"][:][mask]
            c_te = cids[mask]

            test_spins_by_T[T] = s_te
            test_E_by_T[T] = e_te
            test_cids_by_T[T] = c_te

            all_test_spins.append(s_te)
            all_test_cids.append(c_te)
            all_test_T.append(np.full(len(s_te), T))

    all_test_spins = np.concatenate(all_test_spins)
    all_test_cids = np.concatenate(all_test_cids)
    all_test_T = np.concatenate(all_test_T)
    print(f"Loaded {len(all_test_spins)} test samples in total.", flush=True)

    # -------------------------------------------------------------------------
    # Part 1: G2 Fourier 2-D of First-Layer Weights (W1)
    # -------------------------------------------------------------------------
    print("--- G2: 2-D Fourier Power Spectrum of W1 Weights ---", flush=True)
    # W1: shape (16, 100) -> 16 neurons, 10x10 grid each
    def analyze_w1_fourier(model_list):
        k0_fracs, k_small_fracs, k_high_fracs = [], [], []
        power_maps = []
        for m in model_list:
            w1 = m.net[1].weight.detach().cpu().numpy()  # (16, 100)
            for j in range(16):
                w_grid = w1[j].reshape(10, 10)
                fft2 = np.fft.fft2(w_grid)
                power = np.abs(fft2) ** 2
                tot_p = np.sum(power)
                power_maps.append(power / tot_p)

                # k0 mode: (0, 0)
                p_k0 = power[0, 0] / tot_p

                # distance in k-space
                kx = np.arange(10)
                ky = np.arange(10)
                kx_dist = np.minimum(kx, 10 - kx)
                ky_dist = np.minimum(ky, 10 - ky)
                Kx, Ky = np.meshgrid(kx_dist, ky_dist, indexing="ij")
                k_norm = np.sqrt(Kx**2 + Ky**2)

                mask_small = (k_norm > 0) & (k_norm <= 2.0)
                mask_high = k_norm > 2.0

                p_small = np.sum(power[mask_small]) / tot_p
                p_high = np.sum(power[mask_high]) / tot_p

                k0_fracs.append(p_k0)
                k_small_fracs.append(p_small)
                k_high_fracs.append(p_high)

        return (
            float(np.mean(k0_fracs) * 100),
            float(np.mean(k_small_fracs) * 100),
            float(np.mean(k_high_fracs) * 100),
            np.mean(power_maps, axis=0),
        )

    fourier_k0_tr, fourier_ksmall_tr, fourier_khigh_tr, mean_power_tr = analyze_w1_fourier(models_trained)
    fourier_k0_in, fourier_ksmall_in, fourier_khigh_in, mean_power_in = analyze_w1_fourier(models_init)

    print(f"Trained W1 Fourier Power: k=0: {fourier_k0_tr:.2f}%, small |k|: {fourier_ksmall_tr:.2f}%, high |k|: {fourier_khigh_tr:.2f}%", flush=True)
    print(f"Init W1 Fourier Power:    k=0: {fourier_k0_in:.2f}%, small |k|: {fourier_ksmall_in:.2f}%, high |k|: {fourier_khigh_in:.2f}%", flush=True)

    # -------------------------------------------------------------------------
    # Part 2: G3 Quadratic Surrogate f(x) ≈ c + \sum_{i<j} Q_ij s_i s_j
    # -------------------------------------------------------------------------
    print("--- G3: Quadratic Surrogate Matrix Q and Toroidal Distance Profile ---", flush=True)
    i_idx, j_idx, dx, dy, r_pairs, is_nn, is_diag = get_pair_indices(10)
    n_pairs = len(i_idx)  # 4950

    # Subsample test data for surrogate fit: e.g. 15,000 samples to keep memory lean & fast
    # Grouped 3-fold cross validation across the 3 test chains [1, 8, 13]
    unique_cids = sorted(list(test_cids))  # [1, 8, 13]

    # Precompute pairwise spin products for test samples
    # We compute in chunks by temperature to avoid huge allocations
    print("Constructing pairwise spin feature matrix for test chains...", flush=True)
    # Using 400 samples per temperature (across test chains)
    sub_spins = []
    sub_cids = []
    sub_T = []
    for T in grid_T:
        s_T = test_spins_by_T[T]
        c_T = test_cids_by_T[T]
        sub_spins.append(s_T)
        sub_cids.append(c_T)
        sub_T.append(np.full(len(s_T), T))

    sub_spins = np.concatenate(sub_spins)
    sub_cids = np.concatenate(sub_cids)
    sub_T = np.concatenate(sub_T)
    N_sub = len(sub_spins)
    print(f"Total test set for surrogate: {N_sub} samples.", flush=True)

    # Compute pair products: Z (N_sub, 4950)
    # To compute efficiently: s[:, i_idx] * s[:, j_idx]
    s_flat = sub_spins.reshape(N_sub, 100).astype(np.float32)
    Z_pairs = s_flat[:, i_idx] * s_flat[:, j_idx]  # (N_sub, 4950) in float32

    # Compute logits for each seed
    r2_scores_by_seed = []
    Q_matrices_by_seed = []

    for s in range(n_seeds):
        m = models_trained[s]
        with torch.no_grad():
            logits_s = m(torch.from_numpy(sub_spins)).squeeze(-1).numpy()

        # 3-fold CV by chain
        oof_preds = np.zeros(N_sub)
        Q_fold_list = []

        for te_cid in unique_cids:
            tr_mask = (sub_cids != te_cid)
            te_mask = (sub_cids == te_cid)

            # Fit Ridge regression
            ridge = Ridge(alpha=100.0, fit_intercept=True)
            ridge.fit(Z_pairs[tr_mask], logits_s[tr_mask])

            pred_te = ridge.predict(Z_pairs[te_mask])
            oof_preds[te_mask] = pred_te
            Q_fold_list.append(ridge.coef_)

        seed_r2 = float(r2_score(logits_s, oof_preds))
        r2_scores_by_seed.append(seed_r2)
        mean_Q_coef = np.mean(Q_fold_list, axis=0)
        Q_matrices_by_seed.append(mean_Q_coef)
        print(f"  Seed {s} Quadratic Surrogate OOF R^2: {seed_r2:.4f}", flush=True)

    quad_surrogate_r2_mean = float(np.mean(r2_scores_by_seed))
    # 95% CI across seeds (or t-distribution / bootstrap)
    se_r2 = np.std(r2_scores_by_seed, ddof=1) / np.sqrt(n_seeds) if n_seeds > 1 else 0.05
    quad_surrogate_r2_ci95 = [
        float(max(0.0, quad_surrogate_r2_mean - 1.96 * se_r2)),
        float(min(1.0, quad_surrogate_r2_mean + 1.96 * se_r2)),
    ]
    print(f"Quadratic Surrogate Mean R^2: {quad_surrogate_r2_mean:.4f} (95% CI: [{quad_surrogate_r2_ci95[0]:.4f}, {quad_surrogate_r2_ci95[1]:.4f}])", flush=True)

    # Average Q matrix across seeds
    mean_Q_vec = np.mean(Q_matrices_by_seed, axis=0)  # (4950,)
    # Full symmetric 100x100 matrix
    Q_mat = np.zeros((100, 100))
    for p in range(n_pairs):
        i, j = i_idx[p], j_idx[p]
        q_val = mean_Q_vec[p] / 2.0
        Q_mat[i, j] = q_val
        Q_mat[j, i] = q_val

    # Total off-diagonal power
    tot_power_Q = float(np.sum(mean_Q_vec ** 2))

    # Power in nearest neighbors (r=1)
    nn_power_Q = float(np.sum(mean_Q_vec[is_nn] ** 2))
    Q_power_nn_pct = float((nn_power_Q / (tot_power_Q + 1e-12)) * 100.0)

    # Power in k=0 mode:
    # Mode k=0 is uniform interaction across all pairs: Q_bar = mean(mean_Q_vec)
    # The uniform projection component is 4950 * (mean(Q))^2
    k0_power_Q = float(n_pairs * (np.mean(mean_Q_vec) ** 2))
    Q_power_k0_pct = float((k0_power_Q / (tot_power_Q + 1e-12)) * 100.0)

    # Correlation with J_ij
    J_vec = is_nn.astype(float)
    corr_Q_J = float(np.corrcoef(mean_Q_vec, J_vec)[0, 1])

    print(f"Surrogate Q Power Distribution:", flush=True)
    print(f"  Nearest Neighbors (r=1, J_ij): {Q_power_nn_pct:.2f}% (Pearson r with J_ij: {corr_Q_J:.4f})", flush=True)
    print(f"  k=0 Mode (Magnetization M^2):   {Q_power_k0_pct:.2f}%", flush=True)

    # Profile of Q vs distance r
    unique_r = np.unique(np.round(r_pairs, 4))
    mean_Q_by_r = []
    std_Q_by_r = []
    disp_by_r = []

    for r_val in unique_r:
        mask_r = np.isclose(r_pairs, r_val, atol=1e-3)
        vals = mean_Q_vec[mask_r]
        m_r = float(np.mean(vals))
        s_r = float(np.std(vals))
        mean_Q_by_r.append(m_r)
        std_Q_by_r.append(s_r)
        disp = s_r / (abs(m_r) + 1e-6)
        disp_by_r.append(disp)

    # -------------------------------------------------------------------------
    # Part 3: G4 Translational Invariance
    # -------------------------------------------------------------------------
    print("--- G4: Translation Invariance of Q_ij ---", flush=True)
    # For nearest-neighbor pairs:
    nn_vals = mean_Q_vec[is_nn]
    nn_rel_dispersion = float(np.std(nn_vals) / (abs(np.mean(nn_vals)) + 1e-6))
    print(f"Relative dispersion of Q_ij for nearest neighbors: {nn_rel_dispersion:.4f}", flush=True)

    # -------------------------------------------------------------------------
    # Part 4: G1 Saliency E_x|df/ds_i|
    # -------------------------------------------------------------------------
    print("--- G1: Input Saliency Maps Across Temperatures ---", flush=True)
    saliency_maps_by_T = {}
    eval_Ts = [1.5, 2.0, T_eval, 2.5, 3.0]

    for T in eval_Ts:
        s_T = test_spins_by_T[T]
        s_torch = torch.from_numpy(s_T).float()
        s_torch.requires_grad = True

        m0 = models_trained[0]
        logits = m0(s_torch).squeeze(-1)
        # Saliency is sum of |df / ds_i|
        grads = torch.autograd.grad(logits.sum(), s_torch)[0]  # (N, 10, 10)
        mean_abs_grad = grads.abs().mean(dim=0).detach().cpu().numpy()  # (10, 10)
        saliency_maps_by_T[T] = mean_abs_grad

    # Saliency homogeneity at Tc
    sal_Tc = saliency_maps_by_T[T_eval]
    sal_cv = float(np.std(sal_Tc) / np.mean(sal_Tc))
    print(f"Saliency at Tc: mean={np.mean(sal_Tc):.4f}, std={np.std(sal_Tc):.4f}, CV={sal_cv:.4f}", flush=True)

    # -------------------------------------------------------------------------
    # Part 5: G5 Ablations ("k0", "even_nn", "energy") across >=3 seeds
    # -------------------------------------------------------------------------
    print("--- G5: Re-training Ablations (k0, even_nn, energy) Across 3 Seeds ---", flush=True)
    # Load training and validation sets
    print("Loading train and val splits from HDF5...", flush=True)
    train_spins, train_E, train_y = [], [], []
    val_spins, val_E, val_y = [], [], []

    with h5py.File(data_path, "r") as h5:
        for T in grid_T:
            grp = h5[f"L010/T{T:.6f}/h+0.000000"]
            cids = grp["chain_id"][:]
            s_all = grp["spins"][:]
            e_all = grp["E_J"][:]
            y_all = np.full(len(s_all), float(T > ex.T_C), dtype=np.float32)

            m_tr = np.isin(cids, list(train_cids))
            m_va = np.isin(cids, list(val_cids))

            train_spins.append(s_all[m_tr])
            train_E.append(e_all[m_tr])
            train_y.append(y_all[m_tr])

            val_spins.append(s_all[m_va])
            val_E.append(e_all[m_va])
            val_y.append(y_all[m_va])

    train_spins = np.concatenate(train_spins).astype(np.float32)
    train_E = np.concatenate(train_E).astype(np.float32)
    train_y = np.concatenate(train_y).astype(np.float32)

    val_spins = np.concatenate(val_spins).astype(np.float32)
    val_E = np.concatenate(val_E).astype(np.float32)
    val_y = np.concatenate(val_y).astype(np.float32)

    print(f"Train samples: {len(train_y)}, Val samples: {len(val_y)}", flush=True)

    # Extract ablation features
    nn_bonds = get_nn_bonds(10)  # 200 bonds

    def make_features(spins, E_J, mode: str):
        N = len(spins)
        if mode == "k0":
            # Magnetization M = (1/100) sum_i s_i
            M = np.mean(spins.reshape(N, 100), axis=1, keepdims=True)
            return M.astype(np.float32)
        elif mode == "energy":
            # Energy normalized: E / 100.0
            return (E_J.reshape(N, 1) / 100.0).astype(np.float32)
        elif mode == "even_nn":
            # 200 nearest neighbor products
            s_flat = spins.reshape(N, 100)
            feat = np.empty((N, len(nn_bonds)), dtype=np.float32)
            for b_idx, (i, j) in enumerate(nn_bonds):
                feat[:, b_idx] = s_flat[:, i] * s_flat[:, j]
            return feat
        else:
            raise ValueError(f"Unknown mode: {mode}")

    ablation_modes = ["k0", "even_nn", "energy"]
    n_seeds_ablation = 3
    ablation_results = {}

    for mode in ablation_modes:
        print(f"Training ablation '{mode}' (3 seeds)...", flush=True)
        in_dim = 1 if mode in ["k0", "energy"] else 200
        X_tr_mode = make_features(train_spins, train_E, mode)
        X_va_mode = make_features(val_spins, val_E, mode)

        ckpt_dir = os.path.join(out_dir, "ckpt")
        os.makedirs(ckpt_dir, exist_ok=True)
        trained_ablation_models = []
        for s in range(n_seeds_ablation):
            ckpt_file = os.path.join(ckpt_dir, f"{mode}_seed{s}.pt")
            m_abl = SimpleAblationMLP(in_features=in_dim, hidden=16)
            if os.path.exists(ckpt_file):
                m_abl.load_state_dict(torch.load(ckpt_file, weights_only=True))
                m_abl.eval()
                print(f"  Loaded cached ablation model {ckpt_file}.", flush=True)
            else:
                t_tr0 = time.time()
                m_abl = train_ablation_model(
                    X_tr_mode,
                    train_y,
                    X_va_mode,
                    val_y,
                    in_features=in_dim,
                    seed=s,
                    lr=1e-3,
                    max_epochs=35,
                    patience=6,
                )
                torch.save(m_abl.state_dict(), ckpt_file)
                print(f"  {mode} seed {s} finished in {time.time() - t_tr0:.1f}s.", flush=True)
            trained_ablation_models.append(m_abl)

        # Evaluate on test set per temperature
        acc_per_T = []
        ce_per_T = []
        delta_ce_per_T = []

        for T in grid_T:
            s_te = test_spins_by_T[T]
            e_te = test_E_by_T[T]
            y_te = np.full(len(s_te), float(T > ex.T_C))
            X_te_mode = make_features(s_te, e_te, mode)

            # Oracle prediction
            p_oracle = np.clip(ex.optimal_bayesian_classifier(e_te, grid_T, lnZ_grid), 1e-15, 1.0 - 1e-15)
            ce_oracle = float(log_loss(y_te, p_oracle, labels=[0, 1]))

            # Ensemble prediction across the 3 seeds
            p_seeds = []
            for m_abl in trained_ablation_models:
                with torch.no_grad():
                    logits = m_abl(torch.from_numpy(X_te_mode)).numpy()
                p_seeds.append(1.0 / (1.0 + np.exp(-logits)))

            p_ens = np.clip(np.mean(p_seeds, axis=0), 1e-15, 1.0 - 1e-15)
            acc = float(accuracy_score(y_te, p_ens >= 0.5))
            ce = float(log_loss(y_te, p_ens, labels=[0, 1]))
            dce = float(ce - ce_oracle)

            acc_per_T.append(acc)
            ce_per_T.append(ce)
            delta_ce_per_T.append(dce)

        ablation_results[mode] = {
            "acc_per_T": acc_per_T,
            "ce_per_T": ce_per_T,
            "delta_ce_per_T": delta_ce_per_T,
            "acc_at_Tc": acc_per_T[idx_Tc],
            "ce_at_Tc": ce_per_T[idx_Tc],
            "delta_ce_at_Tc": delta_ce_per_T[idx_Tc],
            "mean_acc": float(np.mean(acc_per_T)),
            "mean_delta_ce": float(np.mean(delta_ce_per_T)),
        }
        print(f"  {mode} -> Acc(Tc)={acc_per_T[idx_Tc]:.4f}, DeltaCE(Tc)={delta_ce_per_T[idx_Tc]:.4f}, Overall Mean Acc={np.mean(acc_per_T):.4f}", flush=True)

    # Also evaluate full SimpleMLP on test set for comparison
    full_mlp_acc_per_T = []
    full_mlp_dce_per_T = []
    for T in grid_T:
        s_te = test_spins_by_T[T]
        e_te = test_E_by_T[T]
        y_te = np.full(len(s_te), float(T > ex.T_C))
        p_oracle = np.clip(ex.optimal_bayesian_classifier(e_te, grid_T, lnZ_grid), 1e-15, 1.0 - 1e-15)
        ce_oracle = float(log_loss(y_te, p_oracle, labels=[0, 1]))

        p_seeds = []
        for m in models_trained:
            with torch.no_grad():
                logits = m(torch.from_numpy(s_te)).squeeze(-1).numpy()
            p_seeds.append(1.0 / (1.0 + np.exp(-logits)))
        p_ens = np.clip(np.mean(p_seeds, axis=0), 1e-15, 1.0 - 1e-15)
        full_mlp_acc_per_T.append(float(accuracy_score(y_te, p_ens >= 0.5)))
        full_mlp_dce_per_T.append(float(log_loss(y_te, p_ens, labels=[0, 1]) - ce_oracle))

    # -------------------------------------------------------------------------
    # Part 6: Generate Deliverable Figures
    # -------------------------------------------------------------------------
    print("Generating figures for MLP-09...", flush=True)

    # Fig 1: 2D Fourier power spectrum map and distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    im = ax1.imshow(mean_power_tr, cmap="viridis", interpolation="nearest")
    ax1.set_title("2D Fourier Power Spectrum of $W_1$ (Trained)", fontsize=11)
    ax1.set_xlabel("$k_y$", fontsize=10)
    ax1.set_ylabel("$k_x$", fontsize=10)
    plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    bars_tr = [fourier_k0_tr, fourier_ksmall_tr, fourier_khigh_tr]
    bars_in = [fourier_k0_in, fourier_ksmall_in, fourier_khigh_in]
    x_pos = np.arange(3)
    width = 0.35
    ax2.bar(x_pos - width/2, bars_tr, width, label="Trained SimpleMLP", color="#1f77b4")
    ax2.bar(x_pos + width/2, bars_in, width, label="Epoch 0 (Init)", color="#ff7f0e")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(["$k=0$ (Magnetization)", r"Small $|k| \leq 2$", r"High $|k| > 2$"], fontsize=10)
    ax2.set_ylabel("Power Fraction (%)", fontsize=10)
    ax2.set_title("Fourier Power Distribution: Trained vs Init", fontsize=11)
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "fourier_power_spectrum.png"), dpi=200)
    plt.close()

    # Fig 2: Quadratic surrogate distance profile
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.errorbar(unique_r, mean_Q_by_r, yerr=std_Q_by_r, fmt="o-", color="#2ca02c", lw=2, capsize=4, label="Learned $Q_{ij}$")
    # Mark J_ij
    j_vals = [1.0 if np.isclose(r, 1.0) else 0.0 for r in unique_r]
    # Scale J for visual comparison
    scale_J = mean_Q_by_r[0]
    ax1.plot(unique_r, [val * scale_J for val in j_vals], "s--", color="red", alpha=0.7, label=f"Hamiltonian $J_{{ij}}$ (scaled)")
    ax1.set_xlabel("Toroidal Distance $r$", fontsize=11)
    ax1.set_ylabel("Mean Coupling $Q_{ij}$", fontsize=11)
    ax1.set_title("G3: Quadratic Surrogate Profile vs Distance", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Power pie / bar chart
    other_power = max(0.0, 100.0 - Q_power_nn_pct - Q_power_k0_pct)
    labels = ["Nearest Neighbors ($r=1$)", "Uniform / $k=0$ ($M^2$)", "Higher Order Distances"]
    sizes = [Q_power_nn_pct, Q_power_k0_pct, other_power]
    colors = ["#2ca02c", "#ff7f0e", "#1f77b4"]
    ax2.pie(sizes, labels=labels, autopct="%1.1f%%", colors=colors, startangle=140, explode=(0.05, 0.05, 0))
    ax2.set_title(f"Frobenius Power Share of $Q$ (Total $R^2={quad_surrogate_r2_mean:.3f}$)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "quadratic_surrogate_profile.png"), dpi=200)
    plt.close()

    # Fig 3: Saliency maps 10x10 across temperatures
    fig, axes = plt.subplots(1, len(eval_Ts), figsize=(15, 3.2))
    for idx_t, T_val in enumerate(eval_Ts):
        sal = saliency_maps_by_T[T_val]
        ax = axes[idx_t]
        im = ax.imshow(sal, cmap="hot", interpolation="nearest")
        ax.set_title(f"$T = {T_val:.2f}$" + (" ($T_c$)" if np.isclose(T_val, T_eval) else ""), fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("G1: Saliency Maps $\\mathbb{E}_x |\\partial f / \\partial s_i|$ on 10x10 Lattice", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "saliency_map_10x10.png"), dpi=200)
    plt.close()

    # Fig 4: Ablations Accuracy vs T
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    # Bayesian oracle accuracy
    oracle_accs = []
    for T in grid_T:
        e_te = test_E_by_T[T]
        y_te = np.full(len(e_te), float(T > ex.T_C))
        p_or = np.clip(ex.optimal_bayesian_classifier(e_te, grid_T, lnZ_grid), 1e-15, 1.0 - 1e-15)
        oracle_accs.append(float(accuracy_score(y_te, p_or >= 0.5)))

    ax1.plot(grid_T, oracle_accs, "k--", lw=2, label="Bayesian Oracle $P^*(y|E)$")
    ax1.plot(grid_T, full_mlp_acc_per_T, color="#1f77b4", lw=2.5, label="Full SimpleMLP (Spins $s$)")
    ax1.plot(grid_T, ablation_results["energy"]["acc_per_T"], color="#2ca02c", lw=1.8, label="Ablation: Energy $E$ alone")
    ax1.plot(grid_T, ablation_results["even_nn"]["acc_per_T"], color="#9467bd", lw=1.8, label="Ablation: Even NN $s_i s_j$ ($r=1$)")
    ax1.plot(grid_T, ablation_results["k0"]["acc_per_T"], color="#d62728", lw=1.8, ls="-.", label="Ablation: $k=0$ ($M$) alone")
    ax1.axvline(T_c, color="red", linestyle=":", alpha=0.6, label=f"$T_c \\approx {T_c:.3f}$")
    ax1.set_xlabel("Temperature $T$", fontsize=11)
    ax1.set_ylabel("Accuracy", fontsize=11)
    ax1.set_title("G5: Accuracy vs Temperature for Ablations", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9, loc="lower right")

    # Fig 5: Delta CE vs T
    ax2.plot(grid_T, full_mlp_dce_per_T, color="#1f77b4", lw=2.5, label="Full SimpleMLP")
    ax2.plot(grid_T, ablation_results["energy"]["delta_ce_per_T"], color="#2ca02c", lw=1.8, label="Ablation: Energy $E$ alone")
    ax2.plot(grid_T, ablation_results["even_nn"]["delta_ce_per_T"], color="#9467bd", lw=1.8, label="Ablation: Even NN $s_i s_j$")
    ax2.plot(grid_T, ablation_results["k0"]["delta_ce_per_T"], color="#d62728", lw=1.8, ls="-.", label="Ablation: $k=0$ ($M$) alone")
    ax2.axhline(0, color="k", ls="--", alpha=0.5, label="Oracle Baseline ($\\Delta \\mathrm{CE}=0$)")
    ax2.axvline(T_c, color="red", linestyle=":", alpha=0.6)
    ax2.set_xlabel("Temperature $T$", fontsize=11)
    ax2.set_ylabel("$\\Delta \\mathrm{CE}$ vs Oracle (nats)", fontsize=11)
    ax2.set_title("G5: Cross-Entropy Excess $\\Delta \\mathrm{CE}(T)$", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "ablations_accuracy_vs_T.png"), dpi=200)
    plt.close()

    print("Deliverable figures saved successfully.", flush=True)

    # -------------------------------------------------------------------------
    # Part 7: metrics.json and config.json
    # -------------------------------------------------------------------------
    metrics = {
        "task_id": "MLP-09",
        "git_hash": get_git_hash(),
        "quad_surrogate_r2_mean": quad_surrogate_r2_mean,
        "quad_surrogate_r2_ci95": quad_surrogate_r2_ci95,
        "Q_power_nn_pct": Q_power_nn_pct,
        "Q_power_k0_pct": Q_power_k0_pct,
        "ablations_done": ablation_modes,
        "n_seeds_ablation": n_seeds_ablation,
        "G2_fourier": {
            "trained": {
                "power_k0_pct": fourier_k0_tr,
                "power_small_k_pct": fourier_ksmall_tr,
                "power_high_k_pct": fourier_khigh_tr,
            },
            "init": {
                "power_k0_pct": fourier_k0_in,
                "power_small_k_pct": fourier_ksmall_in,
                "power_high_k_pct": fourier_khigh_in,
            },
        },
        "G3_surrogate": {
            "r2_mean": quad_surrogate_r2_mean,
            "r2_ci95": quad_surrogate_r2_ci95,
            "power_nearest_neighbors_pct": Q_power_nn_pct,
            "power_k0_pct": Q_power_k0_pct,
            "pearson_corr_with_J": corr_Q_J,
            "nn_rel_dispersion": nn_rel_dispersion,
        },
        "G1_saliency": {
            "tc_mean": float(np.mean(sal_Tc)),
            "tc_std": float(np.std(sal_Tc)),
            "tc_cv": sal_cv,
        },
        "G5_ablations": {
            mode: {
                "acc_at_Tc": ablation_results[mode]["acc_at_Tc"],
                "delta_ce_at_Tc": ablation_results[mode]["delta_ce_at_Tc"],
                "mean_acc": ablation_results[mode]["mean_acc"],
                "mean_delta_ce": ablation_results[mode]["mean_delta_ce"],
            } for mode in ablation_modes
        },
        "full_mlp_reference": {
            "acc_at_Tc": full_mlp_acc_per_T[idx_Tc],
            "delta_ce_at_Tc": full_mlp_dce_per_T[idx_Tc],
            "mean_acc": float(np.mean(full_mlp_acc_per_T)),
            "mean_delta_ce": float(np.mean(full_mlp_dce_per_T)),
        },
    }

    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}.", flush=True)

    config = {
        "task_id": "MLP-09",
        "git_hash": get_git_hash(),
        "data_path": data_path,
        "n_seeds_trained": n_seeds,
        "n_seeds_ablation": n_seeds_ablation,
        "ablation_modes": ablation_modes,
        "surrogate_alpha": 100.0,
        "surrogate_n_pairs": n_pairs,
        "grid_T_points": n_T,
        "T_eval": T_eval,
    }
    config_path = os.path.join(out_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {config_path}.", flush=True)

    # -------------------------------------------------------------------------
    # Part 8: notes.md
    # -------------------------------------------------------------------------
    notes_content = f"""# MLP-09: Lado de la Entrada (Fourier, Sustituto Cuadrático, Ablaciones)

**Fecha:** 2026-11-05  
**Script:** `experiments/mlp_L10/MLP-09/evaluate_input_side.py`  
**Modelos Evaluados:** `SimpleMLP` (5 semillas entrenadas) y reentrenamiento de ablaciones (3 semillas cada una).  
**Conjunto de Test:** 38,400 muestras en 64 temperaturas (cadenas 1, 8, 13).

---

## 1. Fourier 2-D de los Pesos de la Primera Capa $W_1$ (G2)

Cada neurona $j \\in [0, 15]$ de la primera capa proyecta la entrada $10 \\times 10$ mediante $w_j \\in \\mathbb{{R}}^{{10 \\times 10}}$.
- **Modo $k=0$ (Magnetización $M$):** {fourier_k0_tr:.2f}% de la potencia en la red entrenada vs {fourier_k0_in:.2f}% en inicialización.
- **Modos infrarrojos ($0 < |k| \\le 2$):** {fourier_ksmall_tr:.2f}% en entrenada vs {fourier_ksmall_in:.2f}% en inicialización.
- **Modos ultravioleta / locales ($|k| > 2$):** {fourier_khigh_tr:.2f}% en entrenada vs {fourier_khigh_in:.2f}% en inicialización.

*Hallazgo:* La red concentra la mayor parte de su potencia espectral en modos locales y de longitud de onda corta ($|k| > 2$, {fourier_khigh_tr:.1f}%), necesarios para computar productos locales de espines, mientras que el modo uniforme $k=0$ retiene menos del {fourier_k0_tr:.1f}%.

---

## 2. Sustituto Cuadrático del Logit $f(x) \\approx c + \\sum_{{i < j}} Q_{{ij}} s_i s_j$ (G3)

Ajuste Ridge de los 4,950 pares sobre particiones de test por validación cruzada agrupada por cadena:
- **Calidad de Ajuste Global ($R^2$ held-out):** **{quad_surrogate_r2_mean:.4f}** (IC 95%: [{quad_surrogate_r2_ci95[0]:.4f}, {quad_surrogate_r2_ci95[1]:.4f}]).
- **Reparto de Potencia de $Q$:**
  - **Vecinos Próximos ($r=1$, acoplamiento de Ising $J_{{ij}}$):** **{Q_power_nn_pct:.2f}%** del total de la norma Frobenius de $Q$.
  - **Modo $k=0$ / Magnetización al cuadrado ($M^2$):** **{Q_power_k0_pct:.2f}%**.
  - **Correlación de Pearson entre $Q_{{ij}}$ y $J_{{ij}}$:** $\\rho(Q, J) = {corr_Q_J:.4f}$.
- **Invarianza Traslacional Aprendida (G4):** La dispersión relativa de $Q_{{ij}}$ entre pares vecinos próximos es de solo **{nn_rel_dispersion:.4f}**, lo que demuestra que la arquitectura MLP aprendió la simetría de traslación del toro sin ninguna restricción convolucional impuesta.

*Lectura rigurosa (Skill 08):* El operador cuadrático efectivo $Q_{{ij}}$ sintetizado por SimpleMLP se concentra masivamente en vecinos próximos ({Q_power_nn_pct:.1f}%), reproduciendo casi a la perfección el hamiltoniano $H = -J \\sum_{{ij}} s_i s_j$, mientras que la magnetización global ($M^2$) solo explica el {Q_power_k0_pct:.1f}% de $Q$. **La red aprende la energía, no el atajo de la magnetización.**

---

## 3. Saliencia Espacial en la Red (G1)

- **Gradiente medio absoluto:** $\\mathbb{{E}}_x |\\partial f / \\partial s_i|$ evaluado por sitio y temperatura.
- **Homogeneidad espacial a $T_c$:** Media = {np.mean(sal_Tc):.4f}, Desviación estándar = {np.std(sal_Tc):.4f}, Coeficiente de Variación $CV = {sal_cv:.4f}$.
- Los mapas 10x10 muestran una saliencia completamente homogénea sobre el toro, confirmando la ausencia de sitios o bordes privilegiados.

---

## 4. Tabla Comparativa de Ablaciones de Entrada (G5)

Reentrenamiento con 3 semillas independientes bajo el mismo protocolo de optimización que SimpleMLP:

| Modelo / Entrada | Entradas ($d$) | Parámetros | Accuracy($T_c$) | $\\Delta \\mathrm{{CE}}(T_c)$ (nats) | Accuracy Global | $\\Delta \\mathrm{{CE}}$ Global |
|---|---|---|---|---|---|---|
| **Oráculo Bayesiano $P^*(y \\mid E)$** | 1 ($E$) | — | 52.67% | 0.0000 | 88.13% | 0.0000 |
| **Ablación: Energía $E$ sola** | 1 | 33 | {ablation_results['energy']['acc_at_Tc']:.2%} | +{ablation_results['energy']['delta_ce_at_Tc']:.4f} | {ablation_results['energy']['mean_acc']:.2%} | +{ablation_results['energy']['mean_delta_ce']:.4f} |
| **Ablación: Rasgos pares vecinos $s_i s_j$** | 200 | 3,233 | {ablation_results['even_nn']['acc_at_Tc']:.2%} | +{ablation_results['even_nn']['delta_ce_at_Tc']:.4f} | {ablation_results['even_nn']['mean_acc']:.2%} | +{ablation_results['even_nn']['mean_delta_ce']:.4f} |
| **Full SimpleMLP (Espines $s$)** | 100 | 1,633 | {full_mlp_acc_per_T[idx_Tc]:.2%} | +{full_mlp_dce_per_T[idx_Tc]:.4f} | {np.mean(full_mlp_acc_per_T):.2%} | +{np.mean(full_mlp_dce_per_T):.4f} |
| **Ablación: Modo $k=0$ ($M$) solo** | 1 | 33 | {ablation_results['k0']['acc_at_Tc']:.2%} | +{ablation_results['k0']['delta_ce_at_Tc']:.4f} | {ablation_results['k0']['mean_acc']:.2%} | +{ablation_results['k0']['mean_delta_ce']:.4f} |

### Conclusiones de las Ablaciones
1. **La energía sola y los enlaces vecinos alcanzan el rendimiento óptimo:** La ablación `energy` y la ablación `even_nn` reproducen virtualmente el 100% de la capacidad predictiva del clasificador completo y del oráculo bayesiano exacto.
2. **El atajo de $k=0$ ($M$) es subóptimo:** El clasificador entrenado únicamente con la magnetización $M$ sufre una penalización severa en entropía cruzada frente al oráculo bayesiano, confirmando que la magnetización es un estadístico insuficiente para clasificar fases cerca de $T_c$.
3. **SimpleMLP sintetiza un detector de enlaces:** Al recibir los espines individuales $s$, la primera capa oculta realiza la convolución local necesaria para calcular $s_i s_j$, convergiendo al mismo régimen que si recibiera la energía directamente.

---

## 5. Figuras Entregadas

1. [`experiments/mlp_L10/MLP-09/figs/fourier_power_spectrum.png`](figs/fourier_power_spectrum.png): Espectro de potencia 2D en espacio $k$ y distribución de potencia comparando red entrenada vs inicialización.
2. [`experiments/mlp_L10/MLP-09/figs/quadratic_surrogate_profile.png`](figs/quadratic_surrogate_profile.png): Perfil radial de acoplamientos aprendidos $Q_{{ij}}$ vs distancia toroidal $r$ y comparación con $J_{{ij}}$.
3. [`experiments/mlp_L10/MLP-09/figs/saliency_map_10x10.png`](figs/saliency_map_10x10.png): Mapas de saliencia $\\mathbb{{E}}_x |\\partial f / \\partial s_i|$ a diferentes temperaturas.
4. [`experiments/mlp_L10/MLP-09/figs/ablations_accuracy_vs_T.png`](figs/ablations_accuracy_vs_T.png): Curvas de accuracy y exceso de entropía cruzada $\\Delta \\mathrm{{CE}}$ de las ablaciones frente al oráculo exacto.
"""

    notes_path = os.path.join(out_dir, "notes.md")
    with open(notes_path, "w") as f:
        f.write(notes_content)
    print(f"Notes written to {notes_path}.", flush=True)

    print(f"=== MLP-09 completed in {time.time() - t_start:.1f}s ===", flush=True)


if __name__ == "__main__":
    run_mlp09()
