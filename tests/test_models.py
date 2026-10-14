import math

import numpy as np
import pytest
import torch

from ising_fisher.models import MaskedConv2d, SimpleCNN, SimpleGenCNN, SimpleMLP
from ising_fisher.observables import exact_distribution
from ising_fisher.train import (evaluate_generator, train_classifier,
                                train_generator)


def all_states(L: int, dtype=torch.float64) -> torch.Tensor:
    n = L * L
    idx = np.arange(2 ** n)[:, None]
    bits = (idx >> np.arange(n)[None, :]) & 1
    return torch.tensor(2 * bits - 1, dtype=dtype).reshape(-1, L, L)


def make_model(z2: bool, seed: int = 0, scale: float = 2.5) -> SimpleGenCNN:
    """Random, deliberately non-trivial (non-uniform) double-precision model."""
    torch.manual_seed(seed)
    m = SimpleGenCNN(z2_equivariant=z2).double()
    with torch.no_grad():
        for p in m.parameters():
            p.mul_(scale)
    return m


# --------------------------------------------------------------------------
# Generative model: normalisation and symmetry (the required tests)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("z2", [False, True])
@pytest.mark.parametrize("T,h", [(2.269, 0.0), (1.3, 0.4), (3.0, -0.7)])
def test_gen_cnn_normalised_L4(z2, T, h):
    model = make_model(z2)
    x = all_states(4)
    with torch.no_grad():
        lp = model.log_prob(x, T, h)
    assert lp.shape == (2 ** 16,)
    total = lp.exp().sum().item()
    assert abs(total - 1.0) < 1e-5, total
    # make sure the test is not trivially passed by a uniform distribution
    assert lp.max() - lp.min() > 0.1


@pytest.mark.parametrize("T,h", [(2.0, 0.0), (1.5, 0.6), (3.2, -1.1)])
def test_z2_equivariance_exact(T, h):
    model = make_model(True, seed=3)
    x = all_states(4)
    with torch.no_grad():
        a = model.log_prob(x, T, h)
        b = model.log_prob(-x, T, -h)
    assert torch.allclose(a, b, rtol=0, atol=1e-13)
    # exact: same two numbers fed to logaddexp in swapped order
    assert torch.equal(a, b)


def test_non_equivariant_model_is_not_z2_symmetric():
    model = make_model(False, seed=3)
    x = all_states(4)
    with torch.no_grad():
        a = model.log_prob(x, 1.5, 0.6)
        b = model.log_prob(-x, 1.5, -0.6)
    assert (a - b).abs().max() > 1e-3


def test_z2_override_flag():
    model = make_model(False)
    x = all_states(2)
    with torch.no_grad():
        a = model.log_prob(x, 1.5, 0.5, z2_equivariant=True)
        b = make_model(True).log_prob(x, 1.5, 0.5)
    assert torch.equal(a, b)


# --------------------------------------------------------------------------
# Generative model: autoregressive structure, sampling
# --------------------------------------------------------------------------
def test_autoregressive_masking():
    """z_i depends only on s_{<i} (raster order), never on s_i or later sites."""
    L = 5
    model = make_model(False, seed=1)
    torch.manual_seed(0)
    x = (2 * torch.randint(0, 2, (4, L, L)) - 1).double()
    with torch.no_grad():
        z0 = model.logits(x, 1.7, 0.3).reshape(4, -1)
        for k in range(L * L):
            x2 = x.clone().reshape(4, -1)
            x2[:, k:] = -x2[:, k:]          # flip site k and everything after it
            z1 = model.logits(x2.reshape(4, L, L), 1.7, 0.3).reshape(4, -1)
            assert torch.equal(z0[:, :k + 1], z1[:, :k + 1]), k


def test_conditioning_channels_unmasked():
    m = MaskedConv2d(3, 16, 3, "A", n_cond_channels=2)
    assert m.mask[:, 1:].all()                      # T, h channels fully unmasked
    spin = m.mask[0, 0]
    assert spin[1, 1] == 0                          # type A excludes current pixel
    assert spin[:1].all() and spin[1, 0] == 1       # row above + left neighbour
    assert spin[1, 2] == 0 and spin[2].sum() == 0   # right neighbour, row below
    mb = MaskedConv2d(16, 16, 3, "B")
    assert mb.mask[0, 0, 1, 1] == 1
    assert mb.mask.sum() == 16 * 16 * 5


def test_parameter_counts():
    g = SimpleGenCNN()
    n = sum(p.numel() for p in g.parameters())
    assert 1000 <= n <= 3000 and g.n_effective_parameters() <= n
    for M in (SimpleMLP, SimpleCNN):
        n = sum(p.numel() for p in M(10).parameters())
        assert 1000 <= n <= 3000, (M.__name__, n)


@pytest.mark.parametrize("z2", [False, True])
def test_sample_matches_log_prob(z2):
    L, n = 2, 40000
    model = make_model(z2, seed=5, scale=2.0)
    torch.manual_seed(0)
    T, h = 1.8, 0.5
    s = model.sample(n, L, T, h)
    assert s.shape == (n, L, L) and set(s.unique().tolist()) <= {-1.0, 1.0}
    states = all_states(L)
    with torch.no_grad():
        p = model.log_prob(states, T, h).exp()
    assert abs(p.sum().item() - 1) < 1e-10
    code = ((s.reshape(n, -1) + 1) / 2 * (2 ** torch.arange(L * L).double())).sum(1).long()
    emp = torch.bincount(code, minlength=2 ** (L * L)).double() / n
    assert (emp - p).abs().max() < 0.012


def test_sample_dtype_float32_default():
    m = SimpleGenCNN()
    s = m.sample(3, 4, 2.0, 0.0)
    assert s.dtype == torch.float32 and m.log_prob(s, 2.0, 0.0).shape == (3,)


# --------------------------------------------------------------------------
# Classifiers
# --------------------------------------------------------------------------
@pytest.mark.parametrize("M", [SimpleMLP, SimpleCNN])
def test_classifier_shapes(M):
    model = M(6)
    x = (2 * torch.randint(0, 2, (5, 6, 6)) - 1).float()
    assert model(x).shape == (5, 1)
    assert model(x.unsqueeze(1)).shape == (5, 1)


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def _exact_samples(L, T, h, n, seed=0):
    _, _, p = exact_distribution(L, T, h)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(p), size=n, p=p)
    return all_states(L, torch.float32)[idx]


def test_train_classifier_separates_phases():
    L = 4
    lo = _exact_samples(L, 1.0, 0.0, 600, 0)
    hi = _exact_samples(L, 6.0, 0.0, 600, 1)
    # |m| is even under Z2; use odd-agnostic features by flipping random signs
    X = torch.cat([lo, hi])
    y = torch.cat([torch.zeros(600), torch.ones(600)])
    perm = torch.randperm(1200, generator=torch.Generator().manual_seed(0))
    X, y = X[perm], y[perm]
    for M in (SimpleMLP, SimpleCNN):
        model = M(L)
        hist = train_classifier(model, X[:900], y[:900], X_val=X[900:], y_val=y[900:],
                                epochs=40, lr=1e-2)
        assert hist["train_loss"][-1] < hist["train_loss"][0]
        assert hist["val_acc"][-1] > 0.8, (M.__name__, hist["val_acc"][-1])


@pytest.mark.parametrize("z2", [False, True])
def test_train_generator_reduces_nll_towards_entropy(z2):
    L, T, h = 3, 2.0, 0.0
    X = _exact_samples(L, T, h, 4000, seed=2)
    _, _, p = exact_distribution(L, T, h)
    entropy = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    torch.manual_seed(0)
    model = SimpleGenCNN(z2_equivariant=z2)
    start = evaluate_generator(model, X, T, h)
    hist = train_generator(model, X, T, h, epochs=25, lr=1e-2, batch_size=128)
    end = evaluate_generator(model, X, T, h)
    assert start > math.log(2) * L * L - 0.5       # untrained ~ uniform
    assert end < start - 0.3
    assert end > entropy - 0.1                      # cannot beat the true entropy (up to MC noise)
    assert hist["train_nll"][-1] < hist["train_nll"][0]
