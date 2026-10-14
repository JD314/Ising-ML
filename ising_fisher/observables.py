"""
Exact enumeration (small L) and Fisher-information estimators.

With E_J = -J sum s_i s_j, M = sum s_i and p(s|T,h) ∝ exp[-(E_J - h M)/T]:

    F_TT = Var(E_J - h M) / T^4
    F_hh = Var(M) / T^2
    F_Th = Cov(E_J - h M, M) / T^3
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

MAX_EXACT_L = 4  # 2^16 = 65536 states


@lru_cache(maxsize=None)
def _enumerate_states(L: int, J: float):
    """E_J and M for all 2^(L^2) configurations (periodic bonds, each counted once)."""
    if L > MAX_EXACT_L:
        raise ValueError(f"exact enumeration limited to L <= {MAX_EXACT_L}, got L={L}")
    n = L * L
    idx = np.arange(2 ** n, dtype=np.int64)[:, None]
    bits = (idx >> np.arange(n)[None, :]) & 1
    s = (2 * bits - 1).astype(np.int8).reshape(-1, L, L)
    s64 = s.astype(np.int64)
    bonds = (s64 * np.roll(s64, -1, axis=1)).sum(axis=(1, 2)) \
        + (s64 * np.roll(s64, -1, axis=2)).sum(axis=(1, 2))
    E_J = -J * bonds.astype(np.float64)
    M = s64.sum(axis=(1, 2))
    E_J.setflags(write=False)
    M.setflags(write=False)
    return E_J, M


def exact_distribution(L: int, T: float, h: float, J: float = 1.0):
    """Return (E_J, M, p) over all 2^(L^2) states; p is the Boltzmann probability."""
    E_J, M = _enumerate_states(int(L), float(J))
    logw = -(E_J - h * M) / T
    logw = logw - logw.max()
    p = np.exp(logw)
    p /= p.sum()
    return E_J, M, p


def fisher_from_moments(var_E, var_M, cov_EM, T):
    """Assemble (F_TT, F_hh, F_Th) from the moments of (E_J - hM, M)."""
    return var_E / T ** 4, var_M / T ** 2, cov_EM / T ** 3


def exact_fisher(L: int, T: float, h: float, J: float = 1.0) -> dict:
    """Exact Fisher information in (T, h) for the L x L periodic Ising model."""
    E_J, M, p = exact_distribution(L, T, h, J)
    E = E_J - h * M
    mE, mM = p @ E, p @ M
    var_E = p @ (E - mE) ** 2
    var_M = p @ (M - mM) ** 2
    cov = p @ ((E - mE) * (M - mM))
    F_TT, F_hh, F_Th = fisher_from_moments(var_E, var_M, cov, T)
    return {
        "F_TT": float(F_TT), "F_hh": float(F_hh), "F_Th": float(F_Th),
        "mean_E_J": float(p @ E_J), "mean_M": float(mM),
    }


def fisher_from_samples(E_J, M, T: float, h: float, ddof: int = 0) -> dict:
    """Fisher information from MC (or exact-weighted-sampled) E_J and M arrays."""
    E_J = np.asarray(E_J, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)
    E = E_J - h * M
    n = E.size
    var_E = E.var(ddof=ddof)
    var_M = M.var(ddof=ddof)
    cov = ((E - E.mean()) * (M - M.mean())).sum() / (n - ddof)
    F_TT, F_hh, F_Th = fisher_from_moments(var_E, var_M, cov, T)
    return {"F_TT": float(F_TT), "F_hh": float(F_hh), "F_Th": float(F_Th)}
