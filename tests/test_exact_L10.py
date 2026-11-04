"""Tests for ising_fisher.exact_L10 (Transfer Matrix Exact Oracle)."""

import numpy as np
import pytest

from ising_fisher import exact_L10 as ex
from ising_fisher.observables import _enumerate_states, exact_distribution


@pytest.mark.parametrize("L", [2, 3, 4])
@pytest.mark.parametrize("T", [1.5, 2.269, 3.0])
def test_log_Z_matches_exact_enumeration(L, T):
    """MANDATORY VALIDATION: relative error in ln Z must be < 1e-10."""
    beta = 1.0 / T
    lz_tm = ex.log_Z_from_transfer_matrix(L, beta)

    E_J, _ = _enumerate_states(L, 1.0)
    w = -E_J * beta
    wmax = np.max(w)
    lz_enum = float(wmax + np.log(np.sum(np.exp(w - wmax))))

    rel_err = abs(lz_tm - lz_enum) / abs(lz_enum)
    assert rel_err < 1e-10, f"L={L} T={T} rel_err={rel_err}"


@pytest.mark.parametrize("L", [2, 3, 4])
@pytest.mark.parametrize("T", [1.5, 2.269, 3.0])
def test_thermodynamics_matches_exact_enumeration(L, T):
    """MANDATORY VALIDATION: relative error in <E> and Var(E) must be < 1e-6."""
    th = ex.exact_thermodynamics(L, T, h_step=1e-4)
    E_J, _, p = exact_distribution(L, T, 0.0)

    mE_ref = float(np.sum(p * E_J))
    varE_ref = float(np.sum(p * (E_J - mE_ref) ** 2))

    rel_err_mE = abs(th["mean_E"] - mE_ref) / abs(mE_ref)
    rel_err_varE = abs(th["var_E"] - varE_ref) / abs(varE_ref)

    assert rel_err_mE < 1e-6, f"L={L} T={T} <E> rel_err={rel_err_mE}"
    assert rel_err_varE < 1e-6, f"L={L} T={T} Var(E) rel_err={rel_err_varE}"
    assert th["F_TT"] > 0
    assert th["C"] > 0


def test_step_invariance_L10():
    """Verify that changing h_step (1e-3, 1e-4, 1e-5) produces stable F_TT."""
    T = 2.269
    th_3 = ex.exact_thermodynamics(10, T, h_step=1e-3)
    th_4 = ex.exact_thermodynamics(10, T, h_step=1e-4)
    th_5 = ex.exact_thermodynamics(10, T, h_step=1e-5)

    rel_diff_34 = abs(th_3["F_TT"] - th_4["F_TT"]) / th_4["F_TT"]
    rel_diff_45 = abs(th_4["F_TT"] - th_5["F_TT"]) / th_4["F_TT"]

    assert rel_diff_34 < 1e-4
    assert rel_diff_45 < 1e-5


def test_oracle_posterior_normalization_and_classifier():
    """Verify posterior sums to 1 and bayesian classifier is in [0, 1]."""
    grid_T = ex.default_temperature_grid()
    assert len(grid_T) == 64

    # Dummy ln_Z values for testing
    ln_Z = np.linspace(100.0, 50.0, len(grid_T))
    E_samples = np.array([-180.0, -140.0, -80.0, -20.0])

    post = ex.oracle_posterior(E_samples, grid_T, ln_Z)
    assert post.shape == (4, 64)
    assert np.allclose(post.sum(axis=1), 1.0)
    assert np.all(post >= 0.0)

    # Scalar input test
    post_scalar = ex.oracle_posterior(-140.0, grid_T, ln_Z)
    assert post_scalar.shape == (64,)
    assert np.isclose(post_scalar.sum(), 1.0)

    # Optimal bayesian classifier
    p_star = ex.optimal_bayesian_classifier(E_samples, grid_T, ln_Z)
    assert p_star.shape == (4,)
    assert np.all((p_star >= 0.0) & (p_star <= 1.0))
    # Higher energy (paramagnetic/disordered) should have higher p_star
    assert p_star[-1] > p_star[0]
