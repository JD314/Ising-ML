"""
Fisher-information / Schur-invariant estimators for Stage 1 (CONTEXT_ISING_ML.md §6).

Generative model q_theta(x | T, h) (``SimpleGenCNN``):

    u_theta(x) = grad_theta log q_theta(x | T, h)          (N, P_active)
    u_c(x)     = d/dc       log q_theta(x | T, h),  c in {T, h}   (N,)

    F_theta = E_q[u_theta u_theta^T],   G_c = E_q[u_theta u_c],   F_cc^model = E_q[u_c^2]

    r_c = G_c^T F_theta^+ G_c / F_cc^model  in [0, 1],     S_c = F_cc^model (1 - r_c)

``r_c`` is the (uncentred) R^2 of regressing u_c on u_theta, so it is computed
here by a weighted least-squares projection (SVD of the weighted score matrix),
which is numerically much better conditioned than forming ``pinv(F_theta)``.
The two are identical in exact arithmetic with the cutoff convention

    keep eigen-directions of F_theta with  lambda_k > rcond * lambda_max
    (equivalently singular values of sqrt(w) U with  sigma_k > sqrt(rcond) * sigma_max).

Classifier f_theta(x) (``SimpleMLP`` / ``SimpleCNN``): ``r_f`` is the centred R^2
of regressing the physical thermal score s_T = (E - <E>)/T^2, E = E_J - h M, onto
the per-sample logit gradients grad_theta f(x) (§6.5 (d)).

All Fisher matrices / moments are accumulated in float64; the model is copied
to float64 for every score computation (the caller's model is not modified).
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from torch import nn
from torch.func import functional_call, grad_and_value, vmap

from .models.gen_cnn import MaskedConv2d
from .observables import MAX_EXACT_L

DTYPE = torch.float64
ArrayLike = Union[np.ndarray, torch.Tensor, Sequence[float], float]


# --------------------------------------------------------------------------- #
# Small utilities
# --------------------------------------------------------------------------- #
def _as64(a) -> torch.Tensor:
    if torch.is_tensor(a):
        return a.detach().to(DTYPE)
    return torch.as_tensor(np.asarray(a), dtype=DTYPE)


def _broadcast64(v, n: int) -> torch.Tensor:
    v = _as64(v).reshape(-1)
    if v.numel() == 1:
        return v.expand(n).clone()
    if v.numel() != n:
        raise ValueError(f"condition has {v.numel()} entries, expected 1 or {n}")
    return v


def _spins64(x) -> torch.Tensor:
    x = _as64(x)
    if x.dim() == 4:
        x = x.squeeze(1)
    if x.dim() != 3 or x.shape[-1] != x.shape[-2]:
        raise ValueError(f"expected spins of shape (N, L, L), got {tuple(x.shape)}")
    return x


def _model64(model: nn.Module) -> nn.Module:
    return copy.deepcopy(model).to(DTYPE).eval()


def all_states(L: int) -> torch.Tensor:
    """All 2^(L^2) configurations, float64 (2^(L^2), L, L), same ordering as
    ``ising_fisher.observables`` (bit k of the state index is site k, row-major)."""
    if L > MAX_EXACT_L:
        raise ValueError(f"exact enumeration limited to L <= {MAX_EXACT_L}, got L={L}")
    n = L * L
    idx = torch.arange(2 ** n, dtype=torch.int64)[:, None]
    bits = (idx >> torch.arange(n, dtype=torch.int64)[None, :]) & 1
    return (2 * bits - 1).to(DTYPE).reshape(-1, L, L)


class _MethodModule(nn.Module):
    """Expose ``model.<method>`` as ``forward`` so ``torch.func.functional_call`` can use it."""

    def __init__(self, model: nn.Module, method: str, **kwargs):
        super().__init__()
        self.m = model
        self.method = method
        self.kwargs = kwargs

    def forward(self, *args):
        return getattr(self.m, self.method)(*args, **self.kwargs)


def _live_masks(wrapper: nn.Module, active_only: bool) -> Dict[str, torch.Tensor]:
    """Boolean live-entry mask per parameter name (masked conv taps are dead)."""
    masks = {name: torch.ones_like(p, dtype=torch.bool) for name, p in wrapper.named_parameters()}
    if active_only:
        for mname, mod in wrapper.named_modules():
            if isinstance(mod, MaskedConv2d):
                masks[f"{mname}.weight"] = mod.mask.to(torch.bool)
    return masks


def _flatten(grads: Dict[str, torch.Tensor], masks: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Per-sample gradient dict -> (B, P_active) using the live masks."""
    cols = []
    for name, mk in masks.items():
        g = grads[name]
        cols.append(g.reshape(g.shape[0], -1)[:, mk.reshape(-1)])
    return torch.cat(cols, dim=1)


def _param_labels(masks: Dict[str, torch.Tensor]) -> List[str]:
    labels = []
    for name, mk in masks.items():
        flat = mk.reshape(-1)
        idx = torch.nonzero(flat).reshape(-1).tolist()
        clean = name[2:] if name.startswith("m.") else name
        labels += [f"{clean}[{i}]" for i in idx]
    return labels


# --------------------------------------------------------------------------- #
# Generative scores
# --------------------------------------------------------------------------- #
@dataclass
class GenerativeScores:
    """Per-sample scores of a conditional generative model (all float64)."""

    u_theta: torch.Tensor   # (N, P_active)
    u_T: torch.Tensor       # (N,)
    u_h: torch.Tensor       # (N,)
    log_q: torch.Tensor     # (N,)
    param_labels: List[str] = field(default_factory=list, repr=False)

    @property
    def n_params(self) -> int:
        return self.u_theta.shape[1]


def _gen_score_fn(model: nn.Module, z2_equivariant: Optional[bool], active_only: bool):
    wrapper = _MethodModule(_model64(model), "log_prob", z2_equivariant=z2_equivariant)
    params = {k: v.detach() for k, v in wrapper.named_parameters()}
    buffers = {k: v.detach() for k, v in wrapper.named_buffers()}
    masks = _live_masks(wrapper, active_only)

    def logq(p, T, h, x):
        return functional_call(wrapper, (p, buffers), (x.unsqueeze(0), T.reshape(1), h.reshape(1)))[0]

    batched = vmap(grad_and_value(logq, argnums=(0, 1, 2)), in_dims=(None, 0, 0, 0))

    def scores(x, T, h):
        (g_p, g_T, g_h), lq = batched(params, T, h, x)
        return _flatten(g_p, masks), g_T, g_h, lq

    return scores, masks


def compute_generative_scores(model: nn.Module, x, T, h, *, active_only: bool = True,
                              z2_equivariant: Optional[bool] = None,
                              chunk_size: int = 1024) -> GenerativeScores:
    """
    Per-sample scores of ``SimpleGenCNN`` (or any module with ``log_prob(x, T, h)``).

    Parameters
    ----------
    x : (N, L, L) spins in {-1, +1}. For the score identity E_q[u] = 0 these should be
        *model* samples, x ~ q_theta(. | T, h) (§6.2).
    T, h : scalars or (N,) arrays.
    active_only : keep only live parameters (masked-out conv taps, which have identically
        zero gradient, are dropped), giving P_active = ``model.n_effective_parameters()``.
        With ``False`` the full P is returned (dead columns are exactly zero).
    z2_equivariant : override ``model.z2_equivariant`` (``None`` = use the model's flag).

    Returns
    -------
    GenerativeScores with u_theta (N, P), u_T (N,), u_h (N,), log_q (N,), all float64.
    """
    xs = _spins64(x)
    N = xs.shape[0]
    Tt, ht = _broadcast64(T, N), _broadcast64(h, N)
    fn, masks = _gen_score_fn(model, z2_equivariant, active_only)
    out = [fn(xs[i:i + chunk_size], Tt[i:i + chunk_size], ht[i:i + chunk_size])
           for i in range(0, N, chunk_size)]
    u, uT, uh, lq = (torch.cat(parts) for parts in zip(*out))
    return GenerativeScores(u, uT, uh, lq, _param_labels(masks))


def conditioning_columns(model: nn.Module, active_only: bool = True, channel: Optional[int] = None) -> torch.Tensor:
    """Boolean (P,) mask over the columns of ``u_theta`` that belong to the weights acting on
    the unmasked conditioning channels (T, h) of every ``MaskedConv2d``.

    ``channel``: ``None`` = all conditioning channels, 0 = T only, 1 = h only.

    Why it matters: in ``SimpleGenCNN`` the condition enters only as ``T * W_T`` (constant
    input channel times its weights), so  T u_T = sum_k W_T,k d/dW_T,k log q  holds exactly;
    u_T lies in the span of these columns and r_T = 1 identically (likewise h u_h for h).
    Dropping them gives the Schur invariant relative to the remaining (spin-path) weights.
    """
    wrapper = _MethodModule(model, "log_prob")
    masks = _live_masks(wrapper, active_only)
    tags = {name: torch.zeros_like(mk) for name, mk in masks.items()}
    for mname, mod in wrapper.named_modules():
        if isinstance(mod, MaskedConv2d) and mod.n_cond_channels > 0:
            first = mod.in_channels - mod.n_cond_channels
            sl = slice(first, None) if channel is None else slice(first + channel, first + channel + 1)
            tags[f"{mname}.weight"][:, sl] = True
    return torch.cat([tags[n].reshape(-1)[masks[n].reshape(-1)] for n in masks])


def exact_generative_fisher(model: nn.Module, L: int, T: float, h: float, *,
                            active_only: bool = True, z2_equivariant: Optional[bool] = None,
                            rcond: float = 1e-6, chunk_size: int = 4096,
                            return_scores: bool = False) -> dict:
    """
    Exact model Fisher matrices by enumerating all 2^(L^2) states (L <= 4), weighting each
    state's scores by its exact probability q_theta(x | T, h) = exp(log_q).

    Moments are streamed chunk-by-chunk in float64, so the (2^(L^2), P) score matrix is
    never held in memory unless ``return_scores=True`` (~0.9 GB at L=4, P_active=1745).

    Returns a dict with
        Z (= sum q, should be 1), mean_u_theta, mean_u_T, mean_u_h (score identity: all 0),
        F_theta (P, P), G_T, G_h (P,), F_TT_model, F_hh_model, F_Th_model,
        r_T, S_T, r_h, S_h, rank, rcond, n_params
    and, if requested, the full 'scores' (GenerativeScores) and 'weights' (q).
    """
    states = all_states(L)
    fn, masks = _gen_score_fn(model, z2_equivariant, active_only)
    P = int(sum(int(m.sum()) for m in masks.values()))
    acc = dict(Z=torch.zeros((), dtype=DTYPE), mean_u_theta=torch.zeros(P, dtype=DTYPE),
               mean_u_T=torch.zeros((), dtype=DTYPE), mean_u_h=torch.zeros((), dtype=DTYPE),
               F_theta=torch.zeros(P, P, dtype=DTYPE), G_T=torch.zeros(P, dtype=DTYPE),
               G_h=torch.zeros(P, dtype=DTYPE), F_TT_model=torch.zeros((), dtype=DTYPE),
               F_hh_model=torch.zeros((), dtype=DTYPE), F_Th_model=torch.zeros((), dtype=DTYPE))
    keep = []
    for i in range(0, states.shape[0], chunk_size):
        xb = states[i:i + chunk_size]
        n = xb.shape[0]
        u, uT, uh, lq = fn(xb, _broadcast64(T, n), _broadcast64(h, n))
        q = lq.exp()
        qu = q[:, None] * u
        acc["Z"] += q.sum()
        acc["mean_u_theta"] += qu.sum(0)
        acc["mean_u_T"] += q @ uT
        acc["mean_u_h"] += q @ uh
        acc["F_theta"] += qu.T @ u
        acc["G_T"] += qu.T @ uT
        acc["G_h"] += qu.T @ uh
        acc["F_TT_model"] += q @ (uT * uT)
        acc["F_hh_model"] += q @ (uh * uh)
        acc["F_Th_model"] += q @ (uT * uh)
        if return_scores:
            keep.append((u, uT, uh, lq))

    out = {k: (float(v) if v.dim() == 0 else v) for k, v in acc.items()}
    for c in ("T", "h"):
        s = schur_from_moments(acc["F_theta"], acc[f"G_{c}"], float(acc[f"F_{c}{c}_model"]), rcond)
        out[f"r_{c}"], out[f"S_{c}"], out["rank"] = s["r_c"], s["S_c"], s["rank"]
    out.update(rcond=rcond, n_params=P, n_states=states.shape[0])
    if return_scores:
        u, uT, uh, lq = (torch.cat(p) for p in zip(*keep))
        out["scores"] = GenerativeScores(u, uT, uh, lq, _param_labels(masks))
        out["weights"] = lq.exp()
    return out


def model_control_fisher(u_T, u_h, weights=None) -> dict:
    """F_TT^model, F_hh^model, F_Th^model = E[u_T^2], E[u_h^2], E[u_T u_h] (float64)."""
    uT, uh = _as64(u_T).reshape(-1), _as64(u_h).reshape(-1)
    w = _weights(weights, uT.shape[0])
    return {"F_TT_model": float(w @ (uT * uT)), "F_hh_model": float(w @ (uh * uh)),
            "F_Th_model": float(w @ (uT * uh))}


# --------------------------------------------------------------------------- #
# Schur invariant
# --------------------------------------------------------------------------- #
def schur_from_moments(F_theta, G_c, F_cc: float, rcond: float = 1e-6) -> dict:
    """r_c = G^T F^+ G / F_cc and S_c = F_cc (1 - r_c) from accumulated moments.

    F^+ keeps eigen-directions with lambda_k > rcond * lambda_max (eigh of symmetric F)."""
    F = _as64(F_theta)
    G = _as64(G_c).reshape(-1)
    F = 0.5 * (F + F.T)
    lam, V = torch.linalg.eigh(F)
    lmax = lam.max().clamp_min(0.0)
    keep = lam > rcond * lmax if lmax > 0 else torch.zeros_like(lam, dtype=torch.bool)
    proj = V[:, keep].T @ G
    quad = float((proj ** 2 / lam[keep]).sum()) if keep.any() else 0.0
    r = quad / F_cc if F_cc > 0 else float("nan")
    return {"r_c": r, "S_c": F_cc * (1.0 - r), "rank": int(keep.sum()), "quad": quad}


def _weights(weights, n: int) -> torch.Tensor:
    if weights is None:
        return torch.full((n,), 1.0 / n, dtype=DTYPE)
    w = _as64(weights).reshape(-1)
    if w.shape[0] != n:
        raise ValueError(f"weights has {w.shape[0]} entries, expected {n}")
    if (w < 0).any():
        raise ValueError("weights must be non-negative")
    return w / w.sum()


def _wls_fit(X: torch.Tensor, Y: torch.Tensor, w: torch.Tensor, rcond: float, center: bool) -> dict:
    """Weighted least squares Y ~ X (optionally with intercept) via a truncated SVD.

    X (n, P), Y (n, k), w (n,) summing to 1. Singular values of sqrt(w) X below
    sqrt(rcond) * sigma_max are discarded (== eigenvalues of X^T W X below rcond * lambda_max).
    """
    if center:
        mx, my = w @ X, w @ Y
        X, Y = X - mx, Y - my
    else:
        mx = torch.zeros(X.shape[1], dtype=DTYPE)
        my = torch.zeros(Y.shape[1], dtype=DTYPE)
    sw = w.sqrt()[:, None]
    Xs, Ys = sw * X, sw * Y
    Uu, S, Vh = torch.linalg.svd(Xs, full_matrices=False)
    keep = S > math.sqrt(rcond) * S[0] if S.numel() and S[0] > 0 else torch.zeros_like(S, dtype=torch.bool)
    Uk, Sk, Vk = Uu[:, keep], S[keep], Vh[keep]
    proj = Uk.T @ Ys                                    # (rank, k)
    beta = Vk.T @ (proj / Sk[:, None])                  # (P, k)
    ss = (Ys ** 2).sum(0)
    r2 = (proj ** 2).sum(0) / ss
    return {"beta": beta, "mx": mx, "my": my, "rank": int(keep.sum()), "r2": r2,
            "XtWX": Xs.T @ Xs, "XtWY": Xs.T @ Ys, "YtWY": ss}


def _oos_r2(fit: dict, X: torch.Tensor, Y: torch.Tensor, w: torch.Tensor, center: bool) -> torch.Tensor:
    pred = (X - fit["mx"]) @ fit["beta"] + fit["my"]
    sse = w @ (Y - pred) ** 2
    sst = w @ (Y - w @ Y) ** 2 if center else w @ Y ** 2
    return 1.0 - sse / sst


def _split_halves(n: int, seed: int, groups=None):
    """50/50 split of row indices; with ``groups`` whole groups (e.g. chain ids) go to one side."""
    rng = np.random.default_rng(seed)
    if groups is None:
        perm = rng.permutation(n)
        a = np.sort(perm[: n // 2])
    else:
        groups = np.asarray(groups).reshape(-1)
        if groups.shape[0] != n:
            raise ValueError("groups must have one entry per sample")
        ug = rng.permutation(np.unique(groups))
        if ug.size < 2:
            raise ValueError("need at least 2 groups for a group split")
        a = np.flatnonzero(np.isin(groups, ug[: ug.size // 2]))
    mask = np.zeros(n, dtype=bool)
    mask[a] = True
    return torch.as_tensor(np.flatnonzero(mask)), torch.as_tensor(np.flatnonzero(~mask))


def _heldout(X, Y, w, rcond, center, seed, groups):
    """Two-fold (A->B, B->A) out-of-sample R^2, averaged; also returns the folds."""
    iA, iB = _split_halves(X.shape[0], seed, groups)
    folds = []
    for tr, te in ((iA, iB), (iB, iA)):
        wtr, wte = w[tr] / w[tr].sum(), w[te] / w[te].sum()
        fit = _wls_fit(X[tr], Y[tr], wtr, rcond, center)
        folds.append(_oos_r2(fit, X[te], Y[te], wte, center))
    folds = torch.stack(folds)                          # (2, k)
    return folds.mean(0), folds


def _out(v: torch.Tensor, scalar: bool):
    v = v.detach()
    return float(v.reshape(-1)[0]) if scalar else v.cpu().numpy()


def compute_schur_invariant(u_theta, u_c, weights=None, rcond: float = 1e-6, split: bool = True,
                            seed: int = 0, groups=None) -> dict:
    """
    Schur invariant of a control score u_c against the weight scores u_theta (§6.3-6.4).

    Parameters
    ----------
    u_theta : (N, P) per-sample weight scores.
    u_c : (N,) control score (or (N, k) for k controls sharing one decomposition).
    weights : optional (N,) non-negative weights (e.g. exact q(x) under enumeration);
        normalised internally. ``None`` = uniform 1/N (Monte Carlo average).
    rcond : relative eigenvalue cutoff for F_theta^+ (lambda_k > rcond * lambda_max).
    split : also return the 50/50 held-out estimate: regression coefficients
        beta = F_theta^+ G_c fitted on one half, uncentred out-of-sample R^2 on the other,
        averaged over both directions. Held-out r_c removes the ~rank/N in-sample bias
        but can be negative when P is not << N.
    groups : optional (N,) group labels (chain ids) so the split keeps groups intact.

    Returns
    -------
    dict with F_theta (P, P), G_c (P,) [or (P, k)], F_cc_model, r_c, S_c, rank, n, rcond,
    and (if split) r_c_heldout, S_c_heldout, r_c_heldout_folds.
    """
    U = _as64(u_theta)
    if U.dim() != 2:
        raise ValueError("u_theta must be (N, P)")
    Y = _as64(u_c)
    scalar = Y.dim() == 1
    Y = Y.reshape(U.shape[0], -1)
    w = _weights(weights, U.shape[0])

    fit = _wls_fit(U, Y, w, rcond, center=False)
    r = fit["r2"]
    F_cc = fit["YtWY"]
    res = {
        "F_theta": fit["XtWX"],
        "G_c": fit["XtWY"][:, 0] if scalar else fit["XtWY"],
        "F_cc_model": _out(F_cc, scalar),
        "r_c": _out(r, scalar),
        "S_c": _out(F_cc * (1.0 - r), scalar),
        "rank": fit["rank"], "n": U.shape[0], "n_params": U.shape[1], "rcond": rcond,
    }
    if split:
        r_ho, folds = _heldout(U, Y, w, rcond, False, seed, groups)
        res["r_c_heldout"] = _out(r_ho, scalar)
        res["S_c_heldout"] = _out(F_cc * (1.0 - r_ho), scalar)
        res["r_c_heldout_folds"] = folds[:, 0].cpu().numpy() if scalar else folds.cpu().numpy()
    return res


# --------------------------------------------------------------------------- #
# Classifier: logit gradients and function-space r_f
# --------------------------------------------------------------------------- #
def compute_logit_gradients(model: nn.Module, x, *, chunk_size: int = 1024):
    """Per-sample grad_theta f(x) (N, P) and logits f(x) (N,), float64.

    ``model(x)`` must return one logit per sample (shape (B,) or (B, 1))."""
    m = _model64(model)
    params = {k: v.detach() for k, v in m.named_parameters()}
    buffers = {k: v.detach() for k, v in m.named_buffers()}

    def f(p, xi):
        return functional_call(m, (p, buffers), (xi.unsqueeze(0),)).reshape(())

    batched = vmap(grad_and_value(f), in_dims=(None, 0))
    xs = _spins64(x)
    masks = {k: torch.ones_like(v, dtype=torch.bool) for k, v in params.items()}
    gs, fs = [], []
    for i in range(0, xs.shape[0], chunk_size):
        g, v = batched(params, xs[i:i + chunk_size])
        gs.append(_flatten(g, masks))
        fs.append(v)
    return torch.cat(gs), torch.cat(fs)


def classifier_label_scores(grad_f, f, y) -> torch.Tensor:
    """u_theta = grad_theta log p_theta(y | x) = (y - sigmoid(f)) grad_theta f, y in {0, 1}."""
    g, fv, yv = _as64(grad_f), _as64(f).reshape(-1), _as64(y).reshape(-1)
    return (yv - torch.sigmoid(fv))[:, None] * g


def thermal_score(E_J, T: float, h: float = 0.0, M=None) -> torch.Tensor:
    """Physical T-score s_T = (E - mean E) / T^2 with E = E_J - h M (sample mean)."""
    E = _as64(E_J).reshape(-1)
    if h != 0.0:
        if M is None:
            raise ValueError("M is required when h != 0")
        E = E - h * _as64(M).reshape(-1)
    return (E - E.mean()) / T ** 2


def rf_from_features(grad_f, s_T, *, rcond: float = 1e-6, split: bool = True, seed: int = 0,
                     groups=None) -> dict:
    """Centred R^2 of s_T ~ grad_theta f (with intercept), in-sample and 50/50 held-out.

    Invariant under any invertible affine map of the features (§6.5 (d))."""
    X = _as64(grad_f)
    Y = _as64(s_T).reshape(-1, 1)
    w = _weights(None, X.shape[0])
    fit = _wls_fit(X, Y, w, rcond, center=True)
    res = {"r_f": float(fit["r2"][0]), "rank": fit["rank"], "n": X.shape[0],
           "n_params": X.shape[1], "rcond": rcond}
    if split:
        r_ho, folds = _heldout(X, Y, w, rcond, True, seed, groups)
        res["r_f_heldout"] = float(r_ho[0])
        res["r_f_heldout_folds"] = folds[:, 0].cpu().numpy()
    return res


def compute_classifier_rf(model: nn.Module, x, E_J, T: float, h: float = 0.0, M=None,
                          rcond: float = 1e-6, *, split: bool = True, seed: int = 0,
                          groups=None, chunk_size: int = 1024) -> dict:
    """
    Function-space invariant r_f(T) of a classifier (§6.5 (d)).

    Regresses s_T = (E_J - h M - mean(E_J - h M)) / T^2 on grad_theta f(x) over the given
    (physical, MC) configurations. Returns in-sample ``r_f`` and the 50/50 held-out
    ``r_f_heldout`` (split by ``groups``, e.g. chain ids, when given so that no correlated
    configurations straddle the split).
    """
    gf, _ = compute_logit_gradients(model, x, chunk_size=chunk_size)
    sT = thermal_score(E_J, T, h, M)
    return rf_from_features(gf, sT, rcond=rcond, split=split, seed=seed, groups=groups)
