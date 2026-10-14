"""
Metropolis Monte Carlo for the 2D Ising model (Stage 1 of the Fisher exploration).

Hamiltonian (periodic boundary conditions, spins s_i in {-1, +1}):

    H = -J sum_<ij> s_i s_j  -  h sum_i s_i

The two pieces are tracked separately:

    E_J = -J sum_<ij> s_i s_j        (interaction energy)
    M   =  sum_i s_i                 (total magnetisation)

so that H = E_J - h M and p(s | T, h) ∝ exp[-(E_J - h M) / T]   (k_B = 1).

Conventions
-----------
* Time is measured in **sweeps**; one sweep = L^2 attempted single-spin flips
  at uniformly random sites.
* Each chain is independent (own RNG stream, own initial state) and runs at a
  single fixed (T, h): there is **no** cross-temperature annealing.
* Local flip cost (the legacy notebook had a sign bug in the field term):

      Delta E = 2 * s_ij * (J * sum_nn s_nn + h)
"""

from __future__ import annotations

import numpy as np
from numba import njit


# --------------------------------------------------------------------------- #
# Low-level kernels
# --------------------------------------------------------------------------- #
@njit(cache=True)
def local_delta_E(spins, i, j, J, h):
    """Energy change H(S with s_ij flipped) - H(S) for H = -J sum s s - h sum s."""
    L = spins.shape[0]
    nn = (
        np.int64(spins[(i + 1) % L, j])
        + np.int64(spins[(i - 1) % L, j])
        + np.int64(spins[i, (j + 1) % L])
        + np.int64(spins[i, (j - 1) % L])
    )
    return 2.0 * spins[i, j] * (J * nn + h)


@njit(cache=True)
def interaction_energy(spins, J):
    """E_J = -J sum_<ij> s_i s_j (each bond counted once, periodic)."""
    L = spins.shape[0]
    acc = 0
    for i in range(L):
        for j in range(L):
            s = np.int64(spins[i, j])
            acc += s * np.int64(spins[(i + 1) % L, j])
            acc += s * np.int64(spins[i, (j + 1) % L])
    return -J * acc


@njit(cache=True)
def magnetization(spins):
    """M = sum_i s_i."""
    L = spins.shape[0]
    acc = 0
    for i in range(L):
        for j in range(L):
            acc += np.int64(spins[i, j])
    return acc


@njit(cache=True)
def sweep(spins, T, h, J):
    """One sweep = L^2 attempted flips at uniformly random sites (in place)."""
    L = spins.shape[0]
    for _ in range(L * L):
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)
        dE = local_delta_E(spins, i, j, J, h)
        if dE <= 0.0 or np.random.random() < np.exp(-dE / T):
            spins[i, j] = -spins[i, j]


@njit(cache=True)
def _run_chain(L, T, h, J, n_burn, n_samples, thin, seed, init):
    np.random.seed(seed)

    spins = np.empty((L, L), dtype=np.int8)
    for i in range(L):
        for j in range(L):
            if init == 0:  # hot (random) start
                spins[i, j] = 1 if np.random.random() < 0.5 else -1
            elif init > 0:  # all up
                spins[i, j] = 1
            else:  # all down
                spins[i, j] = -1

    for _ in range(n_burn):
        sweep(spins, T, h, J)

    out_spins = np.empty((n_samples, L, L), dtype=np.int8)
    out_E = np.empty(n_samples, dtype=np.float64)
    out_M = np.empty(n_samples, dtype=np.int32)
    for k in range(n_samples):
        for _ in range(thin):
            sweep(spins, T, h, J)
        out_spins[k] = spins
        out_E[k] = interaction_energy(spins, J)
        out_M[k] = magnetization(spins)
    return out_spins, out_E, out_M


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def run_chain(L, T, h=0.0, J=1.0, n_burn=1000, n_samples=1000, thin=10, seed=0, init="random"):
    """Run ONE independent Metropolis chain at fixed (T, h).

    Parameters
    ----------
    n_burn : burn-in length in sweeps.
    n_samples : number of recorded configurations.
    thin : sweeps between consecutive recorded configurations (the first sample
        is recorded after `thin` sweeps following burn-in).
    init : 'random' (hot start), 'up' or 'down'.

    Returns
    -------
    spins : (n_samples, L, L) int8 in {-1, +1}
    E_J   : (n_samples,) float64, E_J = -J sum s_i s_j
    M     : (n_samples,) int32, M = sum s_i
    """
    init_code = {"random": 0, "up": 1, "down": -1}[init]
    return _run_chain(
        int(L), float(T), float(h), float(J),
        int(n_burn), int(n_samples), int(thin), int(seed), init_code,
    )


def run_chains(L, T, h=0.0, J=1.0, n_chains=4, n_burn=1000, n_samples=1000, thin=10,
               seed=0, init="random"):
    """Run `n_chains` independent chains at the same (T, h) and concatenate them.

    Seeds are spawned from `np.random.SeedSequence(seed)` so chains are
    statistically independent and results are reproducible.

    Returns
    -------
    spins (n_chains*n_samples, L, L) int8, E_J float64, M int32, chain_id int32
    """
    child_seeds = np.random.SeedSequence(seed).generate_state(n_chains, dtype=np.uint32)
    S, E, Mg, cid = [], [], [], []
    for c, s in enumerate(child_seeds):
        sp, e, m = run_chain(L, T, h, J, n_burn, n_samples, thin, int(s), init)
        S.append(sp)
        E.append(e)
        Mg.append(m)
        cid.append(np.full(n_samples, c, dtype=np.int32))
    return (np.concatenate(S), np.concatenate(E), np.concatenate(Mg), np.concatenate(cid))
