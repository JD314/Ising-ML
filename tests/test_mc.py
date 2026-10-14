import numpy as np
import pytest

from ising_fisher import mc
from ising_fisher.observables import exact_fisher, fisher_from_samples


def total_hamiltonian(s, J, h):
    """Independent (pure numpy) H = -J sum_<ij> s s - h sum s with PBC."""
    s = s.astype(np.int64)
    bonds = (s * np.roll(s, -1, axis=0)).sum() + (s * np.roll(s, -1, axis=1)).sum()
    return -J * bonds - h * s.sum()


@pytest.mark.parametrize("J", [1.0, 0.7])
def test_local_delta_E_matches_total_energy_difference(J):
    h = 0.5
    rng = np.random.default_rng(1234)
    for L in (3, 4, 5):
        for _ in range(5):
            s = rng.choice(np.array([-1, 1], dtype=np.int8), size=(L, L))
            for i in range(L):
                for j in range(L):
                    flipped = s.copy()
                    flipped[i, j] *= -1
                    exact = total_hamiltonian(flipped, J, h) - total_hamiltonian(s, J, h)
                    assert mc.local_delta_E(s, i, j, J, h) == pytest.approx(exact, abs=1e-12)


def test_energy_and_magnetization_kernels():
    rng = np.random.default_rng(7)
    s = rng.choice(np.array([-1, 1], dtype=np.int8), size=(6, 6))
    assert mc.magnetization(s) == s.sum()
    assert mc.interaction_energy(s, 1.3) == pytest.approx(total_hamiltonian(s, 1.3, 0.0))


def test_fisher_matches_exact_enumeration_L4_Tc_h0():
    L, T, h = 4, 2.269, 0.0
    spins, E_J, M = mc.run_chain(L, T, h, n_burn=1000, n_samples=10000, thin=10, seed=2024)

    assert spins.dtype == np.int8 and E_J.dtype == np.float64 and M.dtype == np.int32
    assert set(np.unique(spins)) <= {-1, 1}

    est = fisher_from_samples(E_J, M, T, h)
    ref = exact_fisher(L, T, h)
    for k in ("F_TT", "F_hh"):
        assert est[k] == pytest.approx(ref[k], rel=0.05), (k, est[k], ref[k])
