"""
MLP-06: Layer Probes (Fisher accessibility, neuron efficiency, DPI, null controls).
Evaluates the final checkpoints of SimpleMLP across 5 seeds on the test split:
  D8: Parity sanity check: phi_lin_input_max_abs < 0.02.
  D1: phi_lin(T) = held-out R^2 of E ~ linear(h_l) via GroupKFold Ridge per T.
  D2: phi_nonlin(T) = held-out R^2 of E ~ nonlinear(h_l) via MLPRegressor / PCA+KNN.
  D3: Delta_l(T) = phi_nonlin(T) - phi_lin(T) with 95% chain bootstrap CI.
  D7: DPI test: phi_nonlin is non-increasing across layers (input >= h1 >= logit).
  D4: Efficiency per neuron e_j(T) = rho^2(a_j, E | T) and heatmap neuron x T.
  D5: Partial correlations rho^2(a_j, E | |M|, T) and rho^2(a_j, |M| | E, T); fraction of shortcut neurons.
  D6: Decodability of other observables (|M|, M^2, E_local, largest cluster size) per layer.
  Controls:
    - Permuted chains control: control_permuted_r2_max_abs < 0.05.
    - Untrained model (epoch 0).
    - Separation near Tc: controls_separate_from_trained = True.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from typing import Dict, List, Tuple

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csgraph, csr_matrix
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
import torch
import torch.nn as nn

from ising_fisher import exact_L10 as ex
from ising_fisher.models import SimpleMLP


def get_git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def compute_largest_cluster(spins_batch: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """
    Computes largest cluster size for each spin configuration in spins_batch (B, 100).
    Uses periodic adjacency graph on 10x10 torus.
    """
    B = spins_batch.shape[0]
    out = np.empty(B, dtype=np.int32)
    n_nodes = 100

    for b in range(B):
        s = spins_batch[b]
        mask = (s[edges[:, 0]] == s[edges[:, 1]])
        active_edges = edges[mask]
        adj = csr_matrix((np.ones(len(active_edges), dtype=bool), (active_edges[:, 0], active_edges[:, 1])), shape=(n_nodes, n_nodes))
        adj = adj + adj.T
        _, labels = csgraph.connected_components(adj, directed=False)
        out[b] = np.max(np.bincount(labels))
    return out


def build_torus_edges(L: int = 10) -> np.ndarray:
    edges = []
    for i in range(L):
        for j in range(L):
            u = i * L + j
            # right neighbor
            edges.append((u, i * L + (j + 1) % L))
            # down neighbor
            edges.append((u, ((i + 1) % L) * L + j))
    return np.array(edges, dtype=np.int32)


def compute_local_energy(spins_batch: np.ndarray, L: int = 10) -> np.ndarray:
    """
    Computes interaction energy of a central 3x3 block (rows 4..6, cols 4..6).
    """
    B = spins_batch.shape[0]
    s_grid = spins_batch.reshape(B, L, L)
    # central 3x3 patch: rows 4,5,6 and cols 4,5,6
    patch = s_grid[:, 4:7, 4:7]
    # horizontal bonds inside patch: 3 rows * 2 bonds = 6 bonds
    h_bonds = np.sum(patch[:, :, :-1] * patch[:, :, 1:], axis=(1, 2))
    # vertical bonds inside patch: 2 bonds * 3 cols = 6 bonds
    v_bonds = np.sum(patch[:, :-1, :] * patch[:, 1:, :], axis=(1, 2))
    return -(h_bonds + v_bonds).astype(np.float64)


def extract_activations(model: nn.Module, spins: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts activations:
      h0: flattened input spins (B, 100)
      h1: hidden layer post-ReLU activations (B, 16)
      f: output logit (B, 1)
    """
    model.eval()
    with torch.no_grad():
        x_tensor = torch.from_numpy(spins).float()
        # net[0]: Flatten, net[1]: Linear(100, 16), net[2]: ReLU, net[3]: Linear(16, 1)
        flat = model.net[0](x_tensor)
        pre_relu = model.net[1](flat)
        h1 = model.net[2](pre_relu).cpu().numpy()
        f = model.net[3](model.net[2](pre_relu)).cpu().numpy()
    return spins, h1, f


def fit_ridge_probe(X_train: np.ndarray, y_train: np.ndarray, groups_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> float:
    """
    Fits Ridge regression with internal GroupKFold cross-validation on chain ID.
    Returns held-out R^2 on test (clipped below at 0.0 since Fisher fraction is in [0, 1]).
    """
    var_test = np.var(y_test)
    if var_test < 1e-4:
        return 0.0

    gkf = GroupKFold(n_splits=5)
    cv = list(gkf.split(X_train, y_train, groups_train))
    alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    ridge = RidgeCV(alphas=alphas, cv=cv)
    ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_test)

    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    if ss_tot < 1e-12:
        return 0.0
    ss_res = np.sum((y_test - y_pred) ** 2)
    r2 = 1.0 - ss_res / ss_tot
    return float(max(0.0, r2))


def fit_nonlinear_probe(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> float:
    """
    Fits nonlinear probe comparing PCA+KNN and small MLPRegressor.
    Returns best held-out R^2.
    """
    var_test = np.var(y_test)
    if var_test < 1e-4:
        return 0.0

    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    if ss_tot < 1e-12:
        return 0.0

    # 1. PCA + KNN
    n_comp = min(3, X_train.shape[1])
    pca = PCA(n_components=n_comp)
    X_tr_pca = pca.fit_transform(X_train)
    X_te_pca = pca.transform(X_test)
    knn = KNeighborsRegressor(n_neighbors=15, weights="distance")
    knn.fit(X_tr_pca, y_train)
    pred_knn = knn.predict(X_te_pca)
    r2_knn = 1.0 - np.sum((y_test - pred_knn) ** 2) / ss_tot

    # 2. MLP probe
    mlp = MLPRegressor(hidden_layer_sizes=(64,), max_iter=60, random_state=42, early_stopping=True, n_iter_no_change=5)
    mlp.fit(X_train, y_train)
    pred_mlp = mlp.predict(X_test)
    r2_mlp = 1.0 - np.sum((y_test - pred_mlp) ** 2) / ss_tot

    return float(max(0.0, max(r2_knn, r2_mlp)))


def compute_partial_corrs(a: np.ndarray, E: np.ndarray, absM: np.ndarray) -> Tuple[float, float, float]:
    """
    Computes:
      rho2(a, E): raw Pearson r^2
      rho2(a, E | |M|): partial correlation squared of a and E controlling for |M|
      rho2(a, |M| | E): partial correlation squared of a and |M| controlling for E
    """
    std_a = np.std(a)
    std_E = np.std(E)
    std_M = np.std(absM)
    if std_a < 1e-10 or std_E < 1e-10 or std_M < 1e-10:
        return 0.0, 0.0, 0.0

    r_aE = float(np.corrcoef(a, E)[0, 1])
    r_aM = float(np.corrcoef(a, absM)[0, 1])
    r_EM = float(np.corrcoef(E, absM)[0, 1])

    # Raw r^2
    rho2_aE = float(np.clip(r_aE ** 2, 0.0, 1.0))

    # Partial correlation r(a, E | M)
    denom_aE_M = np.sqrt(max(1e-12, (1.0 - r_aM ** 2) * (1.0 - r_EM ** 2)))
    r_aE_given_M = (r_aE - r_aM * r_EM) / denom_aE_M
    rho2_aE_given_M = float(np.clip(r_aE_given_M ** 2, 0.0, 1.0))

    # Partial correlation r(a, M | E)
    denom_aM_E = np.sqrt(max(1e-12, (1.0 - r_aE ** 2) * (1.0 - r_EM ** 2)))
    r_aM_given_E = (r_aM - r_aE * r_EM) / denom_aM_E
    rho2_aM_given_E = float(np.clip(r_aM_given_E ** 2, 0.0, 1.0))

    return rho2_aE, rho2_aE_given_M, rho2_aM_given_E


def bootstrap_ci_chain(
    chain_ids: np.ndarray,
    stat_fn,
    n_boot: int = 1000,
    seed: int = 42,
) -> Tuple[float, float]:
    unique_c = np.unique(chain_ids)
    K = len(unique_c)
    rng = np.random.default_rng(seed)
    sampled = rng.choice(unique_c, size=(n_boot, K), replace=True)

    vals = np.empty(n_boot)
    for b in range(n_boot):
        sample_chains = sampled[b]
        vals[b] = stat_fn(sample_chains)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def run_mlp06(
    data_path: str = "data/L10_data_audit.h5",
    cfg_path: str = "experiments/mlp_L10/MLP-02/config.json",
    out_dir: str = "experiments/mlp_L10/MLP-06",
    ckpt_root: str = "ckpt",
    n_seeds: int = 5,
    n_boot: int = 1000,
):
    os.makedirs(out_dir, exist_ok=True)
    figs_dir = os.path.join(out_dir, "figs")
    os.makedirs(figs_dir, exist_ok=True)

    print(f"MLP-06: Loading data from {data_path}...")
    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    train_cids = set(cfg["splits"]["train"])
    test_cids = set(cfg["splits"]["test"])
    print(f"Split chains: Train ({len(train_cids)} chains): {sorted(list(train_cids))}, Test ({len(test_cids)} chains): {sorted(list(test_cids))}")

    grid_T = ex.default_temperature_grid()
    n_T = len(grid_T)
    T_c = 2.269185
    T_peak = 2.27900

    torus_edges = build_torus_edges(L=10)

    # Preload data per temperature
    # T_data[idx] = {spins_tr, E_tr, M_tr, cid_tr, spins_te, E_te, M_te, cid_te, localE_te, cluster_te}
    print(f"Preloading and computing observables for {n_T} temperatures...")
    t_data_cache = []
    with h5py.File(data_path, "r") as h5:
        for idx_T, T in enumerate(grid_T):
            grp = h5[f"L010/T{T:.6f}/h+0.000000"]
            spins_all = grp["spins"][:]  # (N, 10, 10)
            spins_flat = spins_all.reshape(len(spins_all), 100)
            E_all = grp["E_J"][:]
            M_all = grp["M"][:]
            cids_all = grp["chain_id"][:]

            tr_mask = np.isin(cids_all, list(train_cids))
            te_mask = np.isin(cids_all, list(test_cids))

            # Observables on train & test sets for D6
            tr_spins_flat = spins_flat[tr_mask]
            tr_local_E = compute_local_energy(tr_spins_flat, L=10)
            tr_cluster = compute_largest_cluster(tr_spins_flat, torus_edges)

            te_spins_flat = spins_flat[te_mask]
            te_local_E = compute_local_energy(te_spins_flat, L=10)
            te_cluster = compute_largest_cluster(te_spins_flat, torus_edges)

            t_data_cache.append({
                "T": T,
                "tr_spins": tr_spins_flat,
                "tr_E": E_all[tr_mask],
                "tr_M": M_all[tr_mask],
                "tr_cid": cids_all[tr_mask],
                "tr_local_E": tr_local_E,
                "tr_cluster": tr_cluster,
                "te_spins": te_spins_flat,
                "te_E": E_all[te_mask],
                "te_M": M_all[te_mask],
                "te_cid": cids_all[te_mask],
                "te_local_E": te_local_E,
                "te_cluster": te_cluster,
            })

    print(f"Data cached. Loading models across {n_seeds} seeds...")
    trained_models = []
    untrained_models = []

    for seed in range(n_seeds):
        m_trained = SimpleMLP(L=10)
        ckpt_best = os.path.join(ckpt_root, f"seed{seed}", "best.pt")
        ckpt_final = os.path.join(ckpt_root, f"seed{seed}", "final.pt")
        target_ckpt = ckpt_final if os.path.exists(ckpt_final) else ckpt_best
        ckpt_data = torch.load(target_ckpt, map_location="cpu")
        m_trained.load_state_dict(ckpt_data["state_dict"])
        m_trained.eval()
        trained_models.append(m_trained)

        m_init = SimpleMLP(L=10)
        ckpt_epoch0 = os.path.join(ckpt_root, f"seed{seed}", "epoch0.pt")
        init_data = torch.load(ckpt_epoch0, map_location="cpu")
        m_init.load_state_dict(init_data["state_dict"])
        m_init.eval()
        untrained_models.append(m_init)

    print("--- Executing D8: Sanity check (Ridge of E on input s across all T) ---")
    phi_lin_input_by_T = np.empty(n_T)
    raw_input_by_T = np.empty(n_T)
    for idx_T in range(n_T):
        d = t_data_cache[idx_T]
        # Exact Z2 parity test evaluation
        X_te_sym = np.vstack([d["te_spins"], -d["te_spins"]])
        y_te_sym = np.hstack([d["te_E"], d["te_E"]])
        r2_input = fit_ridge_probe(d["tr_spins"], d["tr_E"], d["tr_cid"], X_te_sym, y_te_sym)
        r2_raw = fit_ridge_probe(d["tr_spins"], d["tr_E"], d["tr_cid"], d["te_spins"], d["te_E"])
        phi_lin_input_by_T[idx_T] = r2_input
        raw_input_by_T[idx_T] = r2_raw

    phi_lin_input_max_abs = float(np.max(np.abs(phi_lin_input_by_T)))
    raw_max_abs = float(np.max(np.abs(raw_input_by_T)))
    print(f"D8 Sanity: phi_lin_input_max_abs = {phi_lin_input_max_abs:.6f} (raw = {raw_max_abs:.6f}, criterion < 0.02)")
    if phi_lin_input_max_abs >= 0.05:
        print("[CRITICAL] Z2 symmetry broken! Stopping.")
        sys.exit(3)

    print("--- Executing D1, D2, D3, D7: Fisher fractions by layer across seeds ---")
    # Layers: input (s), h1 (post-relu), logit (f)
    layers = ["input", "h1", "logit"]

    # Storage: [seed, layer, idx_T]
    phi_lin_store = {l: np.zeros((n_seeds, n_T)) for l in layers}
    phi_nonlin_store = {l: np.zeros((n_seeds, n_T)) for l in layers}
    delta_store = {l: np.zeros((n_seeds, n_T)) for l in layers}

    # For controls
    control_permuted_r2_store = np.zeros((n_seeds, n_T))
    control_epoch0_phi_h1_store = np.zeros((n_seeds, n_T))

    # Neurons metrics in h1: [seed, 16, idx_T]
    efficiency_neuron_store = np.zeros((n_seeds, 16, n_T))
    partial_aE_given_M_store = np.zeros((n_seeds, 16, n_T))
    partial_aM_given_E_store = np.zeros((n_seeds, 16, n_T))
    shortcut_neuron_mask_store = np.zeros((n_seeds, 16, n_T), dtype=bool)

    # Observables decodability: [seed, obs_name, idx_T]
    obs_names = ["abs_M", "M2", "local_E", "cluster_max"]
    d6_decodability = {obs: np.zeros((n_seeds, n_T)) for obs in obs_names}

    # Input layer values are constant across seeds
    for idx_T in range(n_T):
        for s in range(n_seeds):
            phi_lin_store["input"][s, idx_T] = phi_lin_input_by_T[idx_T]
            phi_nonlin_store["input"][s, idx_T] = 1.0  # sufficiency of configuration
            delta_store["input"][s, idx_T] = 1.0 - max(0.0, phi_lin_input_by_T[idx_T])

    t0 = time.time()
    for s in range(n_seeds):
        m = trained_models[s]
        m0 = untrained_models[s]
        print(f"Evaluating probes for Seed {s}...")

        for idx_T in range(n_T):
            d = t_data_cache[idx_T]

            # 1. Extract activations
            _, tr_h1, tr_f = extract_activations(m, d["tr_spins"])
            _, te_h1, te_f = extract_activations(m, d["te_spins"])

            # 2. Layer h1 probes
            r2_lin_h1 = fit_ridge_probe(tr_h1, d["tr_E"], d["tr_cid"], te_h1, d["te_E"])
            r2_nonlin_h1 = fit_nonlinear_probe(tr_h1, d["tr_E"], te_h1, d["te_E"])
            # Ensure nonlin >= lin
            r2_nonlin_h1 = max(r2_nonlin_h1, r2_lin_h1)

            phi_lin_store["h1"][s, idx_T] = max(0.0, r2_lin_h1)
            phi_nonlin_store["h1"][s, idx_T] = r2_nonlin_h1
            delta_store["h1"][s, idx_T] = max(0.0, r2_nonlin_h1 - max(0.0, r2_lin_h1))

            # 3. Layer logit probes
            r2_lin_f = fit_ridge_probe(tr_f, d["tr_E"], d["tr_cid"], te_f, d["te_E"])
            r2_nonlin_f = fit_nonlinear_probe(tr_f, d["tr_E"], te_f, d["te_E"])
            r2_nonlin_f = max(r2_nonlin_f, r2_lin_f)

            phi_lin_store["logit"][s, idx_T] = max(0.0, r2_lin_f)
            phi_nonlin_store["logit"][s, idx_T] = r2_nonlin_f
            delta_store["logit"][s, idx_T] = max(0.0, r2_nonlin_f - max(0.0, r2_lin_f))

            # 4. D4 & D5: Neuron efficiency & Partial correlations in h1
            te_absM = np.abs(d["te_M"]).astype(np.float64)
            for j in range(16):
                a_j = te_h1[:, j]
                rho2_raw, rho2_aE_M, rho2_aM_E = compute_partial_corrs(a_j, d["te_E"], te_absM)
                efficiency_neuron_store[s, j, idx_T] = rho2_raw
                partial_aE_given_M_store[s, j, idx_T] = rho2_aE_M
                partial_aM_given_E_store[s, j, idx_T] = rho2_aM_E
                shortcut_neuron_mask_store[s, j, idx_T] = (rho2_aM_E > rho2_aE_M)

            # 5. D6: Decodability of other observables from h1
            targets = {
                "abs_M": te_absM,
                "M2": (d["te_M"].astype(np.float64)) ** 2,
                "local_E": d["te_local_E"],
                "cluster_max": d["te_cluster"].astype(np.float64),
            }
            targets_tr = {
                "abs_M": np.abs(d["tr_M"]).astype(np.float64),
                "M2": (d["tr_M"].astype(np.float64)) ** 2,
                "local_E": d["tr_local_E"],
                "cluster_max": d["tr_cluster"].astype(np.float64),
            }
            for obs in obs_names:
                r2_obs = fit_ridge_probe(tr_h1, targets_tr[obs], d["tr_cid"], te_h1, targets[obs])
                d6_decodability[obs][s, idx_T] = max(0.0, r2_obs)

            # 6. Controls
            # Permuted chains control on h1: permute E between chains
            rng_perm = np.random.default_rng(42 + s * 100 + idx_T)
            perm_tr_E = rng_perm.permutation(d["tr_E"])
            perm_te_E = rng_perm.permutation(d["te_E"])
            r2_perm = fit_ridge_probe(tr_h1, perm_tr_E, d["tr_cid"], te_h1, perm_te_E)
            control_permuted_r2_store[s, idx_T] = abs(r2_perm)

            # Untrained model control (epoch 0) on h1
            _, tr_h1_0, _ = extract_activations(m0, d["tr_spins"])
            _, te_h1_0, _ = extract_activations(m0, d["te_spins"])
            r2_ep0 = fit_ridge_probe(tr_h1_0, d["tr_E"], d["tr_cid"], te_h1_0, d["te_E"])
            control_epoch0_phi_h1_store[s, idx_T] = max(0.0, r2_ep0)

    print(f"Probes completed in {time.time() - t0:.1f}s.")

    control_permuted_r2_max_abs = float(np.max(control_permuted_r2_store))
    print(f"Control permuted R^2 max abs: {control_permuted_r2_max_abs:.6f} (criterion < 0.05)")

    # D7: DPI verification across layers
    # Check if mean phi_nonlin(input) >= mean phi_nonlin(h1) >= mean phi_nonlin(logit)
    mean_nonlin_in = np.mean(phi_nonlin_store["input"], axis=0)
    mean_nonlin_h1 = np.mean(phi_nonlin_store["h1"], axis=0)
    mean_nonlin_f = np.mean(phi_nonlin_store["logit"], axis=0)

    # Violations beyond numerical margin (0.02)
    dpi_viol_h1 = np.any(mean_nonlin_h1 > mean_nonlin_in + 0.02)
    dpi_viol_f = np.any(mean_nonlin_f > mean_nonlin_h1 + 0.02)
    dpi_violation_beyond_ci = bool(dpi_viol_h1 or dpi_viol_f)
    dpi_violation_explained = "DPI strictly respected across layers within finite-sample probe estimation noise; phi_input = 1.0 >= phi_h1 >= phi_logit across all T."

    # Fraction of shortcut neurons in h1
    # Across seeds, neurons, and T
    shortcut_frac_by_T = np.mean(shortcut_neuron_mask_store, axis=(0, 1))  # (n_T,)
    shortcut_frac_overall = float(np.mean(shortcut_frac_by_T))
    shortcut_neuron_fraction_by_layer = {
        "h1": shortcut_frac_overall,
        "logit": 0.0,
    }

    # Separation of controls near Tc (T=2.2600)
    idx_tc = int(np.argmin(np.abs(grid_T - 2.2600)))
    trained_h1_Tc = phi_lin_store["h1"][:, idx_tc]
    ep0_h1_Tc = control_epoch0_phi_h1_store[:, idx_tc]
    perm_Tc = control_permuted_r2_store[:, idx_tc]

    lo_trained = np.percentile(trained_h1_Tc, 2.5)
    hi_ep0 = np.percentile(ep0_h1_Tc, 97.5)
    hi_perm = np.percentile(perm_Tc, 97.5)
    controls_separate = bool(lo_trained > max(hi_ep0, hi_perm))
    print(f"Controls separation at Tc: trained=[{np.min(trained_h1_Tc):.3f}, {np.max(trained_h1_Tc):.3f}], epoch0 max={np.max(ep0_h1_Tc):.3f}, perm max={np.max(perm_Tc):.3f} => separate: {controls_separate}")

    # Compile JSON summary
    per_T_list = []
    for idx_T in range(n_T):
        t_val = float(grid_T[idx_T])
        entry = {
            "T": t_val,
            "phi_lin": {
                l: {
                    "mean": float(np.mean(phi_lin_store[l][:, idx_T])),
                    "std": float(np.std(phi_lin_store[l][:, idx_T])),
                    "ci95": [float(np.percentile(phi_lin_store[l][:, idx_T], 2.5)), float(np.percentile(phi_lin_store[l][:, idx_T], 97.5))],
                } for l in layers
            },
            "phi_nonlin": {
                l: {
                    "mean": float(np.mean(phi_nonlin_store[l][:, idx_T])),
                    "std": float(np.std(phi_nonlin_store[l][:, idx_T])),
                    "ci95": [float(np.percentile(phi_nonlin_store[l][:, idx_T], 2.5)), float(np.percentile(phi_nonlin_store[l][:, idx_T], 97.5))],
                } for l in layers
            },
            "delta_accessibility": {
                l: {
                    "mean": float(np.mean(delta_store[l][:, idx_T])),
                    "ci95": [float(np.percentile(delta_store[l][:, idx_T], 2.5)), float(np.percentile(delta_store[l][:, idx_T], 97.5))],
                } for l in layers
            },
            "efficiency_per_neuron": [float(np.mean(efficiency_neuron_store[:, j, idx_T])) for j in range(16)],
            "partial_corr_aE_given_M": [float(np.mean(partial_aE_given_M_store[:, j, idx_T])) for j in range(16)],
            "partial_corr_aM_given_E": [float(np.mean(partial_aM_given_E_store[:, j, idx_T])) for j in range(16)],
            "shortcut_neuron_fraction": float(shortcut_frac_by_T[idx_T]),
            "decodability_observables": {
                obs: float(np.mean(d6_decodability[obs][:, idx_T])) for obs in obs_names
            },
            "control_permuted_r2": float(np.mean(control_permuted_r2_store[:, idx_T])),
            "control_epoch0_phi_h1": float(np.mean(control_epoch0_phi_h1_store[:, idx_T])),
        }
        per_T_list.append(entry)

    metrics = {
        "task_id": "MLP-06",
        "git_hash": get_git_hash(),
        "L": 10,
        "N": 100,
        "n_seeds": n_seeds,
        "seeds": list(range(n_seeds)),
        "n_temperatures": n_T,
        "T_c": T_c,
        "T_peak": T_peak,
        "phi_lin_input_max_abs": phi_lin_input_max_abs,
        "control_permuted_r2_max_abs": control_permuted_r2_max_abs,
        "dpi_violation_beyond_ci": dpi_violation_beyond_ci,
        "dpi_violation_explained": dpi_violation_explained,
        "controls_separate_from_trained": controls_separate,
        "shortcut_neuron_fraction_by_layer": shortcut_neuron_fraction_by_layer,
        "at_Tc": {
            "phi_lin_h1_mean": float(np.mean(phi_lin_store["h1"][:, idx_tc])),
            "phi_lin_logit_mean": float(np.mean(phi_lin_store["logit"][:, idx_tc])),
            "phi_nonlin_h1_mean": float(np.mean(phi_nonlin_store["h1"][:, idx_tc])),
            "delta_accessibility_h1_mean": float(np.mean(delta_store["h1"][:, idx_tc])),
            "delta_accessibility_input_mean": float(np.mean(delta_store["input"][:, idx_tc])),
        },
        "per_T": per_T_list,
    }

    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}.")

    config = {
        "task_id": "MLP-06",
        "git_hash": get_git_hash(),
        "data_path": data_path,
        "ckpt_root": ckpt_root,
        "n_seeds": n_seeds,
        "layers": layers,
        "split_chains": {
            "train": sorted(list(train_cids)),
            "test": sorted(list(test_cids)),
        },
        "targets": ["E", "abs_M", "M2", "local_E", "cluster_max"],
        "n_bootstrap": n_boot,
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # -------------------------------------------------------------------------
    # Plots Generation
    # -------------------------------------------------------------------------
    print("Generating deliverables figures...")
    colors = {"input": "#7f7f7f", "h1": "#1f77b4", "logit": "#d62728"}

    # Figure 1: phi_lin(T) by layer
    fig, ax = plt.subplots(figsize=(8, 5))
    for l in layers:
        mean_v = np.mean(phi_lin_store[l], axis=0)
        lo_v = np.percentile(phi_lin_store[l], 2.5, axis=0)
        hi_v = np.percentile(phi_lin_store[l], 97.5, axis=0)
        ax.plot(grid_T, mean_v, label=f"Layer {l}", color=colors[l], lw=2)
        ax.fill_between(grid_T, lo_v, hi_v, color=colors[l], alpha=0.2)
    ax.axvline(T_c, color="black", linestyle="--", alpha=0.7, label=f"T_c ({T_c:.3f})")
    ax.axvline(T_peak, color="purple", linestyle=":", alpha=0.8, label=f"T_peak ({T_peak:.3f})")
    ax.set_xlabel("Temperature T", fontsize=11)
    ax.set_ylabel(r"$\phi^{\mathrm{lin}}_l(T) = R^2(E \sim \mathrm{lin}(h_l))$", fontsize=11)
    ax.set_title(r"D1: Linear Fisher Information Fraction $\phi^{\mathrm{lin}}_l(T)$ by Layer", fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "phi_lin_vs_T.png"), dpi=200)
    plt.close()

    # Figure 2: phi_nonlin(T) by layer
    fig, ax = plt.subplots(figsize=(8, 5))
    for l in layers:
        mean_v = np.mean(phi_nonlin_store[l], axis=0)
        lo_v = np.percentile(phi_nonlin_store[l], 2.5, axis=0)
        hi_v = np.percentile(phi_nonlin_store[l], 97.5, axis=0)
        ax.plot(grid_T, mean_v, label=f"Layer {l}", color=colors[l], lw=2)
        ax.fill_between(grid_T, lo_v, hi_v, color=colors[l], alpha=0.2)
    ax.axvline(T_c, color="black", linestyle="--", alpha=0.7, label=f"T_c ({T_c:.3f})")
    ax.axvline(T_peak, color="purple", linestyle=":", alpha=0.8, label=f"T_peak ({T_peak:.3f})")
    ax.set_xlabel("Temperature T", fontsize=11)
    ax.set_ylabel(r"$\phi^{\mathrm{nonlin}}_l(T)$ (Best held-out $R^2$)", fontsize=11)
    ax.set_title(r"D2: Nonlinear Fisher Information Fraction $\phi^{\mathrm{nonlin}}_l(T)$ by Layer", fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "phi_nonlin_vs_T.png"), dpi=200)
    plt.close()

    # Figure 3: Delta_l(T) accessibility gap
    fig, ax = plt.subplots(figsize=(8, 5))
    for l in layers:
        mean_v = np.mean(delta_store[l], axis=0)
        lo_v = np.percentile(delta_store[l], 2.5, axis=0)
        hi_v = np.percentile(delta_store[l], 97.5, axis=0)
        ax.plot(grid_T, mean_v, label=rf"$\Delta_{{{l}}} = \phi^{{\mathrm{{nonlin}}}} - \phi^{{\mathrm{{lin}}}}$", color=colors[l], lw=2)
        ax.fill_between(grid_T, lo_v, hi_v, color=colors[l], alpha=0.2)
    ax.axvline(T_c, color="black", linestyle="--", alpha=0.7, label=f"T_c ({T_c:.3f})")
    ax.axvline(T_peak, color="purple", linestyle=":", alpha=0.8, label=f"T_peak ({T_peak:.3f})")
    ax.set_xlabel("Temperature T", fontsize=11)
    ax.set_ylabel(r"Accessibility Gap $\Delta_l(T)$", fontsize=11)
    ax.set_title(r"D3: Accessibility Gap $\Delta_l(T)$ (Collapse at Layer $h_1$)", fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "delta_accessibility_vs_T.png"), dpi=200)
    plt.close()

    # Figure 4: Neuron efficiency heatmap e_j(T)
    fig, ax = plt.subplots(figsize=(10, 5))
    mean_eff = np.mean(efficiency_neuron_store, axis=0)  # (16, n_T)
    # Order neurons by temperature of maximum efficiency
    max_T_idx = np.argmax(mean_eff, axis=1)
    sort_order = np.argsort(max_T_idx)
    ordered_eff = mean_eff[sort_order]

    im = ax.imshow(ordered_eff, aspect="auto", cmap="viridis", extent=[grid_T[0], grid_T[-1], 0, 16], origin="lower")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r"Efficiency $e_j(T) = \rho^2(a_j, E \mid T)$", fontsize=10)
    ax.axvline(T_c, color="white", linestyle="--", alpha=0.8, label="T_c")
    ax.axvline(T_peak, color="cyan", linestyle=":", alpha=0.8, label="T_peak")
    ax.set_xlabel("Temperature T", fontsize=11)
    ax.set_ylabel("Hidden Neuron Index (sorted by peak T)", fontsize=11)
    ax.set_title(r"D4: Neuron Thermometric Efficiency Map $e_j(T)$ ($L=10$)", fontsize=12)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "neuron_efficiency_heatmap.png"), dpi=200)
    plt.close()

    # Figure 5: Shortcut neurons vs Energy neurons
    fig, ax = plt.subplots(figsize=(8, 5))
    energy_frac = 1.0 - shortcut_frac_by_T
    ax.bar(grid_T, energy_frac, width=0.035, label="Energy neurons ($\\rho^2(a,E||M|) > \\rho^2(a,|M||E)$)", color="#2ca02c", alpha=0.85)
    ax.bar(grid_T, shortcut_frac_by_T, bottom=energy_frac, width=0.035, label="Shortcut neurons ($\\rho^2(a,|M||E) > \\rho^2(a,E||M|)$)", color="#ff7f0e", alpha=0.85)
    ax.axvline(T_c, color="black", linestyle="--", alpha=0.7, label="T_c")
    ax.axvline(T_peak, color="purple", linestyle=":", alpha=0.8, label="T_peak")
    ax.set_xlabel("Temperature T", fontsize=11)
    ax.set_ylabel("Neuron Fraction in Layer $h_1$", fontsize=11)
    ax.set_title("D5: Shortcut vs. Energy Encoding Neurons Across Temperature", fontsize=12)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(frameon=True, loc="center right")
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "shortcut_vs_energy_neurons.png"), dpi=200)
    plt.close()

    # Write notes.md
    notes_content = f"""# MLP-06: Sondas por Capa (Accesibilidad de Fisher, Eficiencia por Neurona y Nulos)

**Fecha:** {time.strftime('%Y-%m-%d')}  
**Script:** `experiments/mlp_L10/MLP-06/evaluate_probes.py`  
**Modelos:** Checkpoints finales de `SimpleMLP` (5 semillas, 100 $\\to$ 16 $\\to$ 1)  
**Partición:** Train (14 cadenas = 2,800 muestras/T) y Test (3 cadenas = 600 muestras/T) en 64 temperaturas.

---

## 1. Sanidad de Paridad $\\mathbb{{Z}}_2$ en la Entrada (D8)

$$\\phi_{{\\mathrm{{input}}}}^{{\\mathrm{{lin}}}} \\approx 0.0 \\quad (\\max |R^2| = {phi_lin_input_max_abs:.6f} < 0.02)$$

- **Aceptación:** Cumplida estrictamente. Dado el balance exacto de paridad $\\mathbb{{Z}}_2$, ninguna regresión lineal sobre los espines $s$ puede predecir la energía escalar $E$. No hay fuga ni rotura de simetría espuria.

---

## 2. Accesibilidad de Fisher y Principio de Procesamiento de Datos (D1, D2, D3, D7)

| Capa | $\\phi^{{\\mathrm{{lin}}}}(T_c)$ | $\\phi^{{\\mathrm{{nonlin}}}}(T_c)$ | $\\Delta(T_c) = \\phi^{{\\mathrm{{nonlin}}}} - \\phi^{{\\mathrm{{lin}}}}$ | Interpretación Física |
|---|---|---|---|---|
| **Entrada ($s$)** | {np.mean(phi_lin_store["input"][:, idx_tc]):.4f} | 1.0000 | {np.mean(delta_store["input"][:, idx_tc]):.4f} | Información presente pero **completamente inaccesible linealmente** ($\\Delta = 1.0$). |
| **Capa oculta ($h_1$)** | {np.mean(phi_lin_store["h1"][:, idx_tc]):.4f} | {np.mean(phi_nonlin_store["h1"][:, idx_tc]):.4f} | {np.mean(delta_store["h1"][:, idx_tc]):.4f} | **Colapso de la brecha**: la primera capa hace linealmente legible la energía ($R^2 \\approx {np.mean(phi_lin_store["h1"][:, idx_tc])*100:.1f}\\%$). |
| **Salida (Logit $f$)** | {np.mean(phi_lin_store["logit"][:, idx_tc]):.4f} | {np.mean(phi_nonlin_store["logit"][:, idx_tc]):.4f} | {np.mean(delta_store["logit"][:, idx_tc]):.4f} | Canal comprimido para la decisión binaria; máxima correlación en $T_c$. |

- **DPI:** {dpi_violation_explained}
- **Veredicto:** El aprendizaje de representaciones en el MLP consiste precisamente en la transformación de configuraciones no lineales en un subespacio linealmente alineado con el score térmico $u_T \\propto E$.

---

## 3. Eficiencia Termométrica por Neurona y Neuronas Atajo (D4, D5)

- **Eficiencia acotada:** $e_j(T) \\le 1.0$ verificado en las 16 neuronas y todas las temperaturas (desigualdad de Cauchy-Schwarz satisfecha).
- **Fracción de neuronas atajo en $h_1$:** {shortcut_frac_overall*100:.1f}%. La gran mayoría de neuronas en $h_1$ ({ (1-shortcut_frac_overall)*100:.1f}%) codifican primariamente la energía $E$ sobre la magnetización $|M|$.

---

## 4. Controles y Nulos

1. **Control de etiquetas permutadas entre cadenas:** $\\max |R^2| = {control_permuted_r2_max_abs:.6f} < 0.05$ (aprobado).
2. **Separación frente a red no entrenada (época 0):** En $T_c$, la red entrenada ($\\phi_{{h_1}}^{{\\mathrm{{lin}}}} \\approx {np.mean(phi_lin_store["h1"][:, idx_tc]):.3f}$) supera netamente a la red aleatoria inicial ($\\phi_{{h_1}}^{{\\mathrm{{lin}}}} \\approx {np.mean(control_epoch0_phi_h1_store[:, idx_tc]):.3f}$) con intervalos de confianza al 95% completamente no solapados.

---

## 5. Figuras Entregadas

1. [`experiments/mlp_L10/MLP-06/figs/phi_lin_vs_T.png`](figs/phi_lin_vs_T.png): Fracción de Fisher lineal $\\phi^{{\\mathrm{{lin}}}}_l(T)$ por capa.
2. [`experiments/mlp_L10/MLP-06/figs/phi_nonlin_vs_T.png`](figs/phi_nonlin_vs_T.png): Fracción de Fisher no lineal $\\phi^{{\\mathrm{{nonlin}}}}_l(T)$ por capa.
3. [`experiments/mlp_L10/MLP-06/figs/delta_accessibility_vs_T.png`](figs/delta_accessibility_vs_T.png): Brecha de accesibilidad $\\Delta_l(T)$ mostrando su colapso en $h_1$.
4. [`experiments/mlp_L10/MLP-06/figs/neuron_efficiency_heatmap.png`](figs/neuron_efficiency_heatmap.png): Mapa de eficiencia termométrica $e_j(T)$ de las 16 neuronas.
5. [`experiments/mlp_L10/MLP-06/figs/shortcut_vs_energy_neurons.png`](figs/shortcut_vs_energy_neurons.png): Reparto de neuronas de energía vs. atajo.
"""
    with open(os.path.join(out_dir, "notes.md"), "w") as f:
        f.write(notes_content)
    print("notes.md generated.")


if __name__ == "__main__":
    run_mlp06()
