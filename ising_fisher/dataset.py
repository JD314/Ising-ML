"""
HDF5 storage for Ising MC samples keyed by (L, T, h).

Layout::

    /L{L:03d}/T{T:.6f}/h{h:+.6f}/
        spins     (n, L, L) int8     in {-1, +1}
        E_J       (n,)      float64  -J sum s_i s_j
        M         (n,)      int32    sum s_i
        chain_id  (n,)      int32
        attrs: L, T, h, J

Train/test splits are done by `chain_id` (whole chains go to one side), never
by random sample. No S -> -S augmentation is stored: the odd (magnetisation)
channel must stay informative.
"""

from __future__ import annotations

import h5py
import numpy as np


def make_key(L, T, h) -> str:
    return f"L{int(L):03d}/T{float(T):.6f}/h{float(h):+.6f}"


def save_samples(path, L, T, h, spins, E_J, M, chain_id, J=1.0, overwrite=False):
    """Append (or with overwrite=True replace) the samples for one (L, T, h)."""
    spins = np.ascontiguousarray(spins, dtype=np.int8)
    if spins.ndim != 3 or spins.shape[1:] != (L, L):
        raise ValueError(f"spins must have shape (n, {L}, {L}), got {spins.shape}")
    n = spins.shape[0]
    E_J = np.asarray(E_J, dtype=np.float64)
    M = np.asarray(M, dtype=np.int32)
    chain_id = np.asarray(chain_id, dtype=np.int32)
    for name, a in (("E_J", E_J), ("M", M), ("chain_id", chain_id)):
        if a.shape != (n,):
            raise ValueError(f"{name} must have shape ({n},), got {a.shape}")

    key = make_key(L, T, h)
    with h5py.File(path, "a") as f:
        if key in f:
            if not overwrite:
                raise KeyError(f"{key} already exists in {path} (use overwrite=True)")
            del f[key]
        g = f.create_group(key)
        g.create_dataset("spins", data=spins, compression="gzip", chunks=True)
        g.create_dataset("E_J", data=E_J)
        g.create_dataset("M", data=M)
        g.create_dataset("chain_id", data=chain_id)
        g.attrs.update({"L": int(L), "T": float(T), "h": float(h), "J": float(J)})


def load_samples(path, L, T, h) -> dict:
    """Load everything stored for (L, T, h) as a dict of numpy arrays (+ 'J')."""
    key = make_key(L, T, h)
    with h5py.File(path, "r") as f:
        if key not in f:
            raise KeyError(f"{key} not found in {path}")
        g = f[key]
        out = {name: g[name][()] for name in ("spins", "E_J", "M", "chain_id")}
        out["J"] = float(g.attrs["J"])
    return out


def list_keys(path):
    """List stored (L, T, h) triples, sorted."""
    keys = []
    with h5py.File(path, "r") as f:
        for lk in f:
            for tk in f[lk]:
                for hk in f[lk][tk]:
                    a = f[lk][tk][hk].attrs
                    keys.append((int(a["L"]), float(a["T"]), float(a["h"])))
    return sorted(keys)


def split_by_chain(data: dict, test_fraction=0.25, seed=0, test_chains=None):
    """Split a loaded dict into (train, test) dicts by whole `chain_id`s.

    Chains are shuffled with `seed`; at least one chain lands on each side.
    Pass `test_chains` to choose explicitly.
    """
    chain_id = data["chain_id"]
    chains = np.unique(chain_id)
    if chains.size < 2:
        raise ValueError("need at least 2 chains to split by chain_id")
    if test_chains is None:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(chains)
        n_test = int(np.clip(round(test_fraction * chains.size), 1, chains.size - 1))
        test_chains = perm[:n_test]
    test_mask = np.isin(chain_id, np.asarray(test_chains))
    if test_mask.all() or not test_mask.any():
        raise ValueError("split would leave train or test empty")

    def take(mask):
        return {k: (v[mask] if isinstance(v, np.ndarray) else v) for k, v in data.items()}

    return take(~test_mask), take(test_mask)
