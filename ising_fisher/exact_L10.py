"""
Exact physics oracle for the 2D Ising model on a periodic torus (L x L) at h = 0.

Constructed via transfer matrix (2^L x 2^L) in float64.
Computes:
  - ln Z(T) factoring out the largest eigenvalue
  - <E> = - d ln Z / d beta
  - Var(E) = d^2 ln Z / d beta^2
  - C(T) = Var(E) / T^2
  - F_TT(T) = Var(E) / T^4
  - T_peak of F_TT and C
  - Bayesian posterior oracle_posterior(E, grid_T, lnZ_grid)
  - Optimal Bayesian classifier P*(y=1|x) = sum_{T_k > T_c} P*(T_k|x)
  - Gibbs entropy S(T) and mutual information bounds
"""

from __future__ import annotations

import numpy as np

T_C = 2.0 / np.log(1.0 + np.sqrt(2.0))  # ≈ 2.269185314213022


def build_transfer_matrix(L: int, beta: float, J: float = 1.0) -> np.ndarray:
    """Symmetric transfer matrix V of size (2^L, 2^L) for periodic boundary conditions.

    V[s, s'] = exp( beta * J * [ 0.5 sum_i s_i s_{i+1} + 0.5 sum_i s'_i s'_{i+1} + sum_i s_i s'_i ] )
    with i mod L.
    """
    n = 2 ** L
    idx = np.arange(n, dtype=np.int64)[:, None]
    bits = (idx >> np.arange(L)[None, :]) & 1
    spins = (2 * bits - 1).astype(np.float64)  # (2^L, L)
    intra = (spins * np.roll(spins, -1, axis=1)).sum(axis=1)  # (2^L,)
    inter = spins @ spins.T  # (2^L, 2^L)
    exponent = beta * J * (0.5 * intra[:, None] + 0.5 * intra[None, :] + inter)
    return np.exp(exponent)


def log_Z_from_transfer_matrix(L: int, beta: float, J: float = 1.0) -> float:
    """Exact ln Z = ln Tr(V^L) = ln sum_k lambda_k^L, factoring out max(lambda)."""
    V = build_transfer_matrix(L, beta, J)
    lam = np.linalg.eigvalsh(V)
    lmax = lam[-1]
    if lmax <= 0:
        raise ValueError(f"Max eigenvalue must be positive, got {lmax}")
    ratio = lam / lmax
    ratio_pos = ratio[ratio > 0]
    return float(L * np.log(lmax) + np.log(np.sum(ratio_pos ** L)))


def exact_thermodynamics(L: int, T: float, h_step: float = 1e-4, J: float = 1.0) -> dict:
    """Compute ln Z, <E>, Var(E), C, F_TT, f_TT for given L and T via central differences.

    Parameters
    ----------
    L : lattice side (torus L x L)
    T : temperature
    h_step : step size in beta for central differences (default: 1e-4)
    J : coupling constant (default: 1.0)

    Returns
    -------
    dict with keys: 'ln_Z', 'mean_E', 'var_E', 'C', 'F_TT', 'f_TT', 'T', 'beta'
    """
    beta = 1.0 / float(T)
    lz_p = log_Z_from_transfer_matrix(L, beta + h_step, J)
    lz_0 = log_Z_from_transfer_matrix(L, beta, J)
    lz_m = log_Z_from_transfer_matrix(L, beta - h_step, J)

    d1 = (lz_p - lz_m) / (2.0 * h_step)
    d2 = (lz_p - 2.0 * lz_0 + lz_m) / (h_step ** 2)

    mean_E = -d1
    var_E = d2
    C = var_E / (T ** 2)
    F_TT = var_E / (T ** 4)
    f_TT = F_TT / (L * L)

    return {
        "T": float(T),
        "beta": float(beta),
        "ln_Z": float(lz_0),
        "mean_E": float(mean_E),
        "var_E": float(var_E),
        "C": float(C),
        "F_TT": float(F_TT),
        "f_TT": float(f_TT),
    }


def default_temperature_grid() -> np.ndarray:
    """Default standard temperature grid for the Ising campaign:

    T in [0.1, 4.0] with step 0.1 plus refined [2.00, 2.60] with step 0.02 (64 points).
    """
    coarse = np.round(np.arange(0.1, 4.01, 0.1), 6)
    fine = np.round(np.arange(2.00, 2.601, 0.02), 6)
    return np.unique(np.concatenate([coarse, fine]))


def compute_oracle_grid(L: int = 10, grid_T: np.ndarray | None = None, h_step: float = 1e-4, J: float = 1.0) -> dict:
    """Compute exact thermodynamic quantities over a grid of temperatures."""
    if grid_T is None:
        grid_T = default_temperature_grid()
    else:
        grid_T = np.asarray(grid_T, dtype=np.float64)

    ln_Z = np.empty_like(grid_T)
    mean_E = np.empty_like(grid_T)
    var_E = np.empty_like(grid_T)
    C = np.empty_like(grid_T)
    F_TT = np.empty_like(grid_T)
    f_TT = np.empty_like(grid_T)

    for i, T in enumerate(grid_T):
        th = exact_thermodynamics(L, T, h_step=h_step, J=J)
        ln_Z[i] = th["ln_Z"]
        mean_E[i] = th["mean_E"]
        var_E[i] = th["var_E"]
        C[i] = th["C"]
        F_TT[i] = th["F_TT"]
        f_TT[i] = th["f_TT"]

    return {
        "L": int(L),
        "grid_T": grid_T,
        "ln_Z": ln_Z,
        "mean_E": mean_E,
        "var_E": var_E,
        "C": C,
        "F_TT": F_TT,
        "f_TT": f_TT,
    }


def find_peaks(L: int = 10, T_min: float = 2.2, T_max: float = 2.5, n_points: int = 301, J: float = 1.0) -> dict:
    """Locate T_peak(F_TT) and T_peak(C) with a fine scan."""
    T_scan = np.linspace(T_min, T_max, n_points)
    F_vals = np.empty_like(T_scan)
    C_vals = np.empty_like(T_scan)

    for i, T in enumerate(T_scan):
        th = exact_thermodynamics(L, T, h_step=1e-4, J=J)
        F_vals[i] = th["F_TT"]
        C_vals[i] = th["C"]

    idx_F = int(np.argmax(F_vals))
    idx_C = int(np.argmax(C_vals))

    return {
        "T_peak_F_TT": float(T_scan[idx_F]),
        "max_F_TT": float(F_vals[idx_F]),
        "T_peak_C": float(T_scan[idx_C]),
        "max_C": float(C_vals[idx_C]),
    }


def oracle_posterior(E: np.ndarray | float, grid_T: np.ndarray, ln_Z_grid: np.ndarray) -> np.ndarray:
    """Vectorized calculation of P*(T_k | x) = P*(T_k | E).

    Under uniform prior over {T_k}:
        P*(T_k | E) proportional to exp(-E / T_k - ln Z(T_k))
    Computed using numerically stable logsumexp.

    Parameters
    ----------
    E : scalar or array of shape (N,) with interaction energies E_J
    grid_T : array of shape (K,) with temperatures
    ln_Z_grid : array of shape (K,) with ln Z(T_k)

    Returns
    -------
    post : array of shape (N, K) (or (K,) if E is scalar) summing to 1 along temperature axis.
    """
    E_arr = np.asarray(E, dtype=np.float64)
    scalar_input = E_arr.ndim == 0
    if scalar_input:
        E_arr = E_arr.reshape(1)

    # log_unnorm shape (N, K)
    log_unnorm = -E_arr[:, None] / grid_T[None, :] - ln_Z_grid[None, :]
    m = np.max(log_unnorm, axis=1, keepdims=True)
    log_norm = m + np.log(np.sum(np.exp(log_unnorm - m), axis=1, keepdims=True))
    log_post = log_unnorm - log_norm
    post = np.exp(log_post)

    if scalar_input:
        return post[0]
    return post


def optimal_bayesian_classifier(E: np.ndarray | float, grid_T: np.ndarray, ln_Z_grid: np.ndarray, T_c: float = T_C) -> np.ndarray:
    """Optimal Bayesian phase classifier P*(y=1 | x) = sum_{T_k > T_c} P*(T_k | x).

    Parameters
    ----------
    E : scalar or array of shape (N,)
    grid_T : array of shape (K,)
    ln_Z_grid : array of shape (K,)
    T_c : critical temperature threshold (default: Onsager exact Tc)

    Returns
    -------
    p_star : scalar or array of shape (N,) in [0, 1]
    """
    post = oracle_posterior(E, grid_T, ln_Z_grid)
    mask = grid_T > T_c
    if post.ndim == 1:
        return float(np.sum(post[mask]))
    return np.sum(post[:, mask], axis=1)


def compute_mutual_information(E_samples: np.ndarray, grid_T: np.ndarray, ln_Z_grid: np.ndarray, T_c: float = T_C) -> dict:
    """Estimate exact mutual information I(T; X) and I(Y; X) on sample configurations.

    I(T; X) = ln K - E_x[ H(T | x) ]
    I(Y; X) = H(Y) - E_x[ H(Y | x) ]
    """
    K = len(grid_T)
    post = oracle_posterior(E_samples, grid_T, ln_Z_grid)
    log_post = np.log(np.clip(post, 1e-30, 1.0))
    H_T_given_x = -np.sum(post * log_post, axis=1)

    I_T_X = float(np.log(K) - np.mean(H_T_given_x))

    mask_y1 = grid_T > T_c
    p_y1 = np.sum(post[:, mask_y1], axis=1)
    p_y0 = 1.0 - p_y1
    p_y = np.clip(np.stack([p_y0, p_y1], axis=1), 1e-30, 1.0)
    H_Y_given_x = -np.sum(p_y * np.log(p_y), axis=1)

    p_Y1_prior = float(np.mean(mask_y1))
    p_Y0_prior = 1.0 - p_Y1_prior
    H_Y = float(-(p_Y1_prior * np.log(p_Y1_prior) + p_Y0_prior * np.log(p_Y0_prior)))
    I_Y_X = float(H_Y - np.mean(H_Y_given_x))

    return {
        "I_T_X": I_T_X,
        "I_Y_X": I_Y_X,
        "H_Y": H_Y,
        "ln_K": float(np.log(K)),
        "mean_H_T_given_x": float(np.mean(H_T_given_x)),
        "mean_H_Y_given_x": float(np.mean(H_Y_given_x)),
    }
