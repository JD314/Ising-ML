"""Tests for ising_fisher.fisher: score computation and the §6.5 structural identities (a)-(d)."""

import math

import numpy as np
import pytest
import torch

from ising_fisher import fisher as fz
from ising_fisher import mc
from ising_fisher.models import SimpleCNN, SimpleGenCNN, SimpleMLP
from ising_fisher.observables import exact_distribution

T_C = 2.0 / math.log(1.0 + math.sqrt(2.0))


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def make_gen(z2: bool, hidden: int = 16, seed: int = 0, scale: float = 2.5) -> SimpleGenCNN:
    """Random, deliberately non-uniform double-precision generative model."""
    torch.manual_seed(seed)
    m = SimpleGenCNN(hidden=hidden, z2_equivariant=z2).double()
    with torch.no_grad():
        for p in m.parameters():
            p.mul_(scale)
    return m


def well_conditioned(P: int, seed: int, log_spread: float = 1.0) -> torch.Tensor:
    """Random invertible A = Q1 diag(exp(U[-s, s])) Q2  (cond(A) <= e^{2s})."""
    g = torch.Generator().manual_seed(seed)
    Q1, _ = torch.linalg.qr(torch.randn(P, P, generator=g, dtype=torch.float64))
    Q2, _ = torch.linalg.qr(torch.randn(P, P, generator=g, dtype=torch.float64))
    d = torch.exp((2 * torch.rand(P, generator=g, dtype=torch.float64) - 1) * log_spread)
    return Q1 @ torch.diag(d) @ Q2


def zscores(prod: torch.Tensor, var_of_mean: torch.Tensor = None) -> torch.Tensor:
    """z = mean / SE per column (columns with zero variance dropped)."""
    N = prod.shape[0]
    mean = prod.mean(0)
    se = (prod.var(0) / N).sqrt() if var_of_mean is None else var_of_mean.sqrt()
    ok = se > 0
    return mean[ok] / se[ok]


def exact_iid(L: int, T: float, N: int, seed: int = 0):
    E_J, M, p = exact_distribution(L, T, 0.0)
    idx = np.random.default_rng(seed).choice(p.size, N, p=p)
    return fz.all_states(L)[idx], E_J[idx], M[idx]


# --------------------------------------------------------------------------- #
# score computation
# --------------------------------------------------------------------------- #
def test_all_states_matches_observables_ordering():
    from ising_fisher.observables import _enumerate_states
    E_J, M = _enumerate_states(3, 1.0)
    s = fz.all_states(3)
    assert torch.equal(s.sum(dim=(1, 2)).long(), torch.as_tensor(M))
    bonds = (s * s.roll(-1, 1)).sum((1, 2)) + (s * s.roll(-1, 2)).sum((1, 2))
    assert torch.allclose(-bonds, torch.as_tensor(E_J))


@pytest.mark.parametrize("z2", [False, True])
def test_generative_scores_match_autograd(z2):
    m = make_gen(z2, hidden=8, seed=3)
    x = m.sample(5, 4, 2.0, 0.3)
    sc = fz.compute_generative_scores(m, x, 2.0, 0.3, active_only=False)
    assert sc.u_theta.dtype == torch.float64
    for i in range(5):
        T = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
        h = torch.tensor([0.3], dtype=torch.float64, requires_grad=True)
        lp = m.log_prob(x[i:i + 1], T, h).sum()
        g = torch.autograd.grad(lp, [T, h] + list(m.parameters()))
        assert torch.allclose(sc.u_T[i], g[0][0], atol=1e-12)
        assert torch.allclose(sc.u_h[i], g[1][0], atol=1e-12)
        assert torch.allclose(sc.u_theta[i], torch.cat([t.reshape(-1) for t in g[2:]]), atol=1e-12)
        assert torch.allclose(sc.log_q[i], lp.detach(), atol=1e-12)


def test_active_parameters_drop_only_dead_columns():
    m = make_gen(True)
    x = m.sample(64, 4, 2.3, 0.0)
    full = fz.compute_generative_scores(m, x, 2.3, 0.0, active_only=False)
    act = fz.compute_generative_scores(m, x, 2.3, 0.0, active_only=True)
    assert act.n_params == m.n_effective_parameters() == len(act.param_labels)
    assert full.n_params == sum(p.numel() for p in m.parameters())
    # the active columns are a subset of the full ones and the dropped columns are exactly 0
    assert torch.allclose(full.u_theta.pow(2).sum(), act.u_theta.pow(2).sum(), rtol=1e-13)
    assert torch.equal(full.u_T, act.u_T)


def test_cond_scores_span_control_score():
    """Structural fact of SimpleGenCNN: T enters only via T*W_T, so T u_T = W_T . u_{W_T}."""
    for z2, h in ((True, 0.0), (False, 0.4)):
        m = make_gen(z2, seed=1)
        x = m.sample(256, 4, 2.1, h)
        sc = fz.compute_generative_scores(m, x, 2.1, h, active_only=False)
        hid = m.hidden
        uW = sc.u_theta[:, : hid * 3 * 9].reshape(-1, hid, 3, 3, 3)
        W = m.conv1.weight.detach()
        lhs_T = (uW[:, :, 1] * W[:, 1]).sum((1, 2, 3))
        assert torch.allclose(2.1 * sc.u_T, lhs_T, atol=1e-10)
        lhs_h = (uW[:, :, 2] * W[:, 2]).sum((1, 2, 3))
        assert torch.allclose(h * sc.u_h, lhs_h, atol=1e-10)
        cols = fz.conditioning_columns(m, active_only=False, channel=0)
        assert int(cols.sum()) == hid * 9


def test_schur_svd_equals_pinv_formula():
    torch.manual_seed(0)
    N, P = 3000, 40
    U = torch.randn(N, P, dtype=torch.float64) @ torch.randn(P, P, dtype=torch.float64)
    U[:, :5] = 0.0                                   # dead parameters -> singular F_theta
    uc = U[:, 5:15].sum(1) + torch.randn(N, dtype=torch.float64)
    w = torch.rand(N, dtype=torch.float64)
    for weights in (None, w):
        res = fz.compute_schur_invariant(U, uc, weights=weights, split=False)
        ww = torch.full((N,), 1.0 / N, dtype=torch.float64) if weights is None else w / w.sum()
        F = (ww[:, None] * U).T @ U
        G = (ww[:, None] * U).T @ uc
        Fcc = float(ww @ uc ** 2)
        r = float(G @ torch.linalg.pinv(F, rtol=1e-6, hermitian=True) @ G) / Fcc
        assert abs(res["r_c"] - r) < 1e-10
        assert abs(res["S_c"] - Fcc * (1 - r)) < 1e-10 * Fcc
        assert torch.allclose(res["F_theta"], F) and torch.allclose(res["G_c"], G)
        mom = fz.schur_from_moments(F, G, Fcc)
        assert abs(mom["r_c"] - r) < 1e-10 and mom["rank"] == res["rank"] == P - 5
        assert 0.0 <= res["r_c"] <= 1.0


def test_schur_heldout_and_multicolumn():
    torch.manual_seed(1)
    N, P = 4000, 30
    U = torch.randn(N, P, dtype=torch.float64)
    beta = torch.randn(P, dtype=torch.float64)
    uc = torch.stack([U @ beta, U @ beta + 3 * torch.randn(N, dtype=torch.float64)], 1)
    res = fz.compute_schur_invariant(U, uc)
    assert res["r_c"].shape == (2,)
    assert abs(res["r_c"][0] - 1) < 1e-12 and abs(res["r_c_heldout"][0] - 1) < 1e-12
    true_r2 = float(beta @ beta / (beta @ beta + 9))
    assert abs(res["r_c_heldout"][1] - true_r2) < 0.03
    assert res["r_c"][1] >= res["r_c_heldout"][1]      # in-sample is optimistically biased
    single = fz.compute_schur_invariant(U, uc[:, 1])
    assert abs(single["r_c"] - res["r_c"][1]) < 1e-12


# --------------------------------------------------------------------------- #
# Identity (a): reparametrisation invariance
# --------------------------------------------------------------------------- #
def test_identity_a_reparametrisation_invariance_generative():
    m = make_gen(False, hidden=4, seed=1, scale=1.0)
    x = m.sample(3000, 4, 2.5, 0.3)
    sc = fz.compute_generative_scores(m, x, 2.5, 0.3)
    # Restrict to a well-conditioned basis of the live score space so that no singular
    # value sits near the pinv cutoff (otherwise a reparametrisation could move it across).
    _, S, Vh = torch.linalg.svd(sc.u_theta, full_matrices=False)
    k = int((S > 1e-2 * S[0]).sum())
    U = sc.u_theta @ Vh[:k].T
    A = well_conditioned(k, seed=7)
    for c in (sc.u_T, sc.u_h):
        r0 = fz.compute_schur_invariant(U, c, rcond=1e-12)
        r1 = fz.compute_schur_invariant(U @ A.T, c, rcond=1e-12)
        assert r0["rank"] == r1["rank"] == k
        assert abs(r0["r_c"] - r1["r_c"]) < 1e-10
        assert abs(r0["S_c"] - r1["S_c"]) < 1e-10 * r0["F_cc_model"]
        assert abs(r0["r_c_heldout"] - r1["r_c_heldout"]) < 1e-8
        # G -> A G, F -> A F A^T  (moment form)
        assert torch.allclose(r1["G_c"], A @ r0["G_c"], atol=1e-10)
        assert torch.allclose(r1["F_theta"], A @ r0["F_theta"] @ A.T, atol=1e-10)


def test_identity_a_reparametrisation_invariance_singular():
    """Exactly-dead parameters (zero columns) are mixed by A but stay in the null space."""
    torch.manual_seed(2)
    N, P = 5000, 60
    U = torch.randn(N, P, dtype=torch.float64)
    U[:, 50:] = 0.0
    uc = U[:, :20] @ torch.randn(20, dtype=torch.float64) + 2 * torch.randn(N, dtype=torch.float64)
    w = torch.rand(N, dtype=torch.float64)
    A = well_conditioned(P, seed=3)
    for weights in (None, w):
        r0 = fz.compute_schur_invariant(U, uc, weights=weights)
        r1 = fz.compute_schur_invariant(U @ A.T, uc, weights=weights)
        assert r0["rank"] == r1["rank"] == 50
        assert abs(r0["r_c"] - r1["r_c"]) < 1e-10
        assert abs(r0["r_c_heldout"] - r1["r_c_heldout"]) < 1e-8


# --------------------------------------------------------------------------- #
# Identity (b): classifier with model-sampled labels has r_T = 0
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def mc_states_L6():
    S, E, M, cid = mc.run_chains(6, 2.4, 0.0, 1.0, n_chains=4, n_burn=300, n_samples=2000,
                                 thin=5, seed=11)
    return S, E, M, cid


def test_identity_b_classifier_sampled_labels(mc_states_L6):
    S, E, _, _ = mc_states_L6
    torch.manual_seed(0)
    model = SimpleCNN(6)
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(3.0)
    gf, f = fz.compute_logit_gradients(model, S)
    with torch.no_grad():                     # centre the logits so labels are not saturated
        model.head.bias -= float(f.mean())
    gf, f = fz.compute_logit_gradients(model, S)
    sig = torch.sigmoid(f)
    assert 0.05 < float(sig.std()) and 0.2 < float((sig > 0.5).double().mean()) < 0.8
    y = torch.bernoulli(sig, generator=torch.Generator().manual_seed(5))
    u_theta = fz.classifier_label_scores(gf, f, y)
    u_T = fz.thermal_score(E, 2.4)
    N = u_theta.shape[0]

    # G_T: each component is mean-zero conditionally on x, with exactly known conditional
    # covariance Cov[G_T | x] = (1/N^2) sum_i sigma_i(1-sigma_i) (u_T,i grad f_i)(u_T,i grad f_i)^T
    prod = u_theta * u_T[:, None]
    G = prod.mean(0)
    a = (sig * (1 - sig)).sqrt()[:, None] * gf * u_T[:, None] / N
    z = zscores(prod, (a ** 2).sum(0))
    assert z.abs().max() < 5.0
    # whitened aggregate: chi2 = G^T Cov^+ G ~ chi^2_rank under the identity
    Ua, Sa, Vha = torch.linalg.svd(a, full_matrices=False)
    keep = Sa > 1e-6 * Sa[0]
    chi2 = float(((Vha[keep] @ G) / Sa[keep]).pow(2).sum())
    k = int(keep.sum())
    assert chi2 / k < 1.0 + 5.0 * math.sqrt(2.0 / k)

    res = fz.compute_schur_invariant(u_theta, u_T)
    # in-sample r_T is pure overfitting noise ~ rank/N; held-out is <= 0 up to noise
    assert res["r_c"] < 1.5 * res["rank"] / N + 0.05
    assert res["r_c_heldout"] < 3.0 / math.sqrt(N)
    # power check: regressing u_T on grad f itself (no label noise) is NOT ~0
    assert fz.rf_from_features(gf, u_T)["r_f_heldout"] > 0.3


def test_identity_b_classifier_analytic_label_expectation(mc_states_L6):
    S, E, _, _ = mc_states_L6
    torch.manual_seed(0)
    model = SimpleCNN(6)
    gf, f = fz.compute_logit_gradients(model, S)
    sig = torch.sigmoid(f)
    N = gf.shape[0]
    u_T = fz.thermal_score(E, 2.4)
    # E_{y|x} taken exactly: each x contributes (y=1, w=sigma) and (y=0, w=1-sigma)
    ones, zeros = torch.ones(N, dtype=torch.float64), torch.zeros(N, dtype=torch.float64)
    U = torch.cat([fz.classifier_label_scores(gf, f, ones), fz.classifier_label_scores(gf, f, zeros)])
    w = torch.cat([sig, 1 - sig]) / N
    res = fz.compute_schur_invariant(U, torch.cat([u_T, u_T]), weights=w, split=False)
    scale = float((gf.abs() * u_T.abs()[:, None]).mean())
    assert res["G_c"].abs().max() < 1e-12 * scale
    assert res["r_c"] < 1e-12
    assert abs(res["S_c"] - res["F_cc_model"]) < 1e-12 * res["F_cc_model"]


# --------------------------------------------------------------------------- #
# Identity (c): Z2-equivariant generator at h = 0 has r_h = 0 and F_Th^model = 0
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def exact_z2_L4():
    m = make_gen(True)
    return m, fz.exact_generative_fisher(m, 4, 2.269, 0.0)


def test_identity_c_exact_enumeration(exact_z2_L4):
    m, ex = exact_z2_L4
    assert ex["n_states"] == 2 ** 16 and ex["n_params"] == m.n_effective_parameters()
    assert abs(ex["Z"] - 1.0) < 1e-12
    # score identity E_q[u] = 0 (exact weighting)
    assert ex["mean_u_theta"].abs().max() < 1e-12
    assert abs(ex["mean_u_T"]) < 1e-12 and abs(ex["mean_u_h"]) < 1e-12
    # the identity itself
    assert ex["F_hh_model"] > 1e-3 and ex["F_TT_model"] > 1e-3      # non-degenerate controls
    assert ex["r_h"] < 1e-12
    assert abs(ex["F_Th_model"]) < 1e-12
    assert ex["G_h"].abs().max() < 1e-12
    assert abs(ex["S_h"] - ex["F_hh_model"]) < 1e-12


def test_identity_c_power_non_equivariant_exact():
    """Same architecture without the Z2 symmetrisation: r_h and F_Th are clearly nonzero."""
    m = make_gen(False, hidden=4, seed=0)
    ex = fz.exact_generative_fisher(m, 4, 2.269, 0.0)
    assert ex["r_h"] > 1e-3
    assert abs(ex["F_Th_model"]) > 1e-4


def test_identity_c_model_samples():
    m = make_gen(True)
    N = 4000
    torch.manual_seed(1)
    x = m.sample(N, 4, 2.269, 0.0)
    sc = fz.compute_generative_scores(m, x, 2.269, 0.0)
    z = zscores(sc.u_theta * sc.u_h[:, None])
    assert z.abs().max() < 5.0
    z_th = zscores((sc.u_T * sc.u_h)[:, None])
    assert z_th.abs().max() < 4.0
    res = fz.compute_schur_invariant(sc.u_theta, sc.u_h)
    assert res["r_c"] < 1.5 * res["rank"] / N + 0.05
    cf = fz.model_control_fisher(sc.u_T, sc.u_h)
    assert abs(cf["F_Th_model"]) < 4.0 * float((sc.u_T * sc.u_h).std()) / math.sqrt(N)


# --------------------------------------------------------------------------- #
# Identity (d): function-space r_f
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cls", [SimpleMLP, SimpleCNN])
def test_identity_d_rf_in_unit_interval(cls):
    torch.manual_seed(0)
    X, E_J, _ = exact_iid(4, 2.5, 12000, seed=1)
    model = cls(4)
    res = fz.compute_classifier_rf(model, X, E_J, 2.5)
    assert res["n_params"] == sum(p.numel() for p in model.parameters())
    for r in (res["r_f"], res["r_f_heldout"], *res["r_f_heldout_folds"]):
        assert 0.0 <= r <= 1.0
    assert res["r_f"] >= res["r_f_heldout"] - 0.02


def test_identity_d_rf_linear_invariance():
    torch.manual_seed(0)
    X, E_J, _ = exact_iid(4, 2.3, 8000, seed=2)
    gf, _ = fz.compute_logit_gradients(SimpleMLP(4), X)
    s_T = fz.thermal_score(E_J, 2.3)
    groups = np.repeat(np.arange(8), 1000)
    # well-conditioned basis of the (centred) feature space, as in identity (a)
    _, S, Vh = torch.linalg.svd(gf - gf.mean(0), full_matrices=False)
    k = int((S > 1e-2 * S[0]).sum())
    feats = gf @ Vh[:k].T
    r0 = fz.rf_from_features(feats, s_T, groups=groups, rcond=1e-12)
    A = well_conditioned(k, seed=4)
    shift = torch.randn(k, dtype=torch.float64)
    r1 = fz.rf_from_features(feats @ A.T + shift, s_T, groups=groups, rcond=1e-12)  # affine
    assert r0["rank"] == r1["rank"] == k
    assert abs(r0["r_f"] - r1["r_f"]) < 1e-10
    assert abs(r0["r_f_heldout"] - r1["r_f_heldout"]) < 1e-8
    assert 0.0 <= r0["r_f_heldout"] <= 1.0
    # full feature set (default rcond): dead-ReLU columns are exactly zero -> same projection
    full = fz.rf_from_features(gf, s_T, groups=groups)
    assert 0.0 <= full["r_f_heldout"] <= full["r_f"] <= 1.0
    # s_T scaling / centring does not matter either
    r2 = fz.rf_from_features(feats, 7.0 * s_T + 3.0, groups=groups, rcond=1e-12)
    assert abs(r0["r_f_heldout"] - r2["r_f_heldout"]) < 1e-10


def test_thermal_score_with_field():
    E = np.array([-8.0, -4.0, 0.0, 4.0])
    M = np.array([4, 2, 0, -2])
    s = fz.thermal_score(E, 2.0, h=0.5, M=M)
    Eh = E - 0.5 * M
    assert torch.allclose(s, torch.as_tensor((Eh - Eh.mean()) / 4.0))
    with pytest.raises(ValueError):
        fz.thermal_score(E, 2.0, h=0.5)
