"""
Autoregressive (PixelCNN-style) generative model for Ising spins, conditioned on (T, h).

    q_theta(x | T, h) = prod_i  Bernoulli(s_i = +1 ; sigmoid(z_i(s_{<i}, T, h)))

with sites ordered in raster order (row-major). The log-likelihood is exact and
sampling is exact (ancestral). The conditioning enters as two constant spatial
channels (T, h) that are *not* masked; only the spin channel (and, in deeper
layers, the spin-derived hidden features) is masked.

Optional exact Z2 symmetrisation (h -> -h, x -> -x):

    log q(x | T, h) = logaddexp(log p(x | T, h), log p(-x | T, -h)) - log 2

which satisfies q(-x | T, -h) = q(x | T, h) for every theta and is exactly
normalised (it is the equal mixture of p(.|T,h) and the push-forward of
p(.|T,-h) under x -> -x).
"""

from __future__ import annotations

import math
from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

Cond = Union[float, torch.Tensor]


class MaskedConv2d(nn.Conv2d):
    """
    Conv2d with a binary raster-order autoregressive mask on the *spin* input channels.

    Kernel taps strictly above the centre row, and taps left of the centre in the
    centre row, are kept. The centre tap is kept for mask_type 'B' and removed for
    'A' (first layer: the output at site i must not see s_i).

    The last `n_cond_channels` input channels are conditioning channels (T, h);
    they are left unmasked. Zero padding is used (circular padding would wrap
    "future" pixels into the receptive field and break autoregressivity).
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 mask_type: str = "B", n_cond_channels: int = 0):
        if mask_type not in ("A", "B"):
            raise ValueError("mask_type must be 'A' or 'B'")
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd")
        if not 0 <= n_cond_channels < in_channels:
            raise ValueError("need at least one non-conditioning (spin) input channel")
        super().__init__(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.mask_type = mask_type
        self.n_cond_channels = n_cond_channels

        k = kernel_size
        c = k // 2
        spatial = torch.zeros(k, k)
        spatial[:c, :] = 1.0
        spatial[c, :c] = 1.0
        if mask_type == "B":
            spatial[c, c] = 1.0
        mask = torch.ones(out_channels, in_channels, k, k)
        n_spin = in_channels - n_cond_channels
        mask[:, :n_spin] = spatial
        self.register_buffer("mask", mask)
        with torch.no_grad():
            self.weight.mul_(self.mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight * self.mask, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class SimpleGenCNN(nn.Module):
    """
    Three-layer masked CNN: (spin, T, h) -> hidden -> hidden -> 1 logit per site.

    Layer 1 is Type A (excludes the current pixel), layers 2-3 are Type B.
    ~3 k parameters for hidden=16 (a fraction of the weights are masked to zero
    and receive zero gradient; `n_effective_parameters` counts the live ones).

    Spins are {-1, +1} at the interface. The logit z_i is for P(s_i = +1 | s_<i, T, h).
    """

    def __init__(self, hidden: int = 16, z2_equivariant: bool = False):
        super().__init__()
        self.hidden = hidden
        self.z2_equivariant = z2_equivariant
        self.conv1 = MaskedConv2d(3, hidden, 3, "A", n_cond_channels=2)
        self.conv2 = MaskedConv2d(hidden, hidden, 3, "B")
        self.conv3 = MaskedConv2d(hidden, 1, 3, "B")

    # ---- helpers -------------------------------------------------------
    @staticmethod
    def _spins(x: torch.Tensor) -> torch.Tensor:
        """(B, L, L) or (B, 1, L, L) -> (B, L, L) float with values in {-1, +1}."""
        if x.dim() == 4:
            x = x.squeeze(1)
        if x.dim() != 3 or x.shape[-1] != x.shape[-2]:
            raise ValueError(f"expected spins of shape (B, L, L), got {tuple(x.shape)}")
        return x if x.is_floating_point() else x.to(torch.get_default_dtype())

    @staticmethod
    def _cond(v: Cond, B: int, ref: torch.Tensor) -> torch.Tensor:
        """Scalar or (B,)/(1,) tensor -> (B,) tensor on ref's device/dtype."""
        v = torch.as_tensor(v, dtype=ref.dtype, device=ref.device)
        return v.reshape(-1).expand(B) if v.numel() in (1, B) else v.reshape(B)

    def logits(self, x: torch.Tensor, T: Cond, h: Cond) -> torch.Tensor:
        """Per-site logits z_i for P(s_i=+1 | s_<i, T, h); shape (B, L, L)."""
        x = self._spins(x).to(self.conv1.weight.dtype)
        B, L, _ = x.shape
        Tc = self._cond(T, B, x).view(B, 1, 1, 1).expand(B, 1, L, L)
        hc = self._cond(h, B, x).view(B, 1, 1, 1).expand(B, 1, L, L)
        a = torch.cat([x.unsqueeze(1), Tc, hc], dim=1)
        a = F.relu(self.conv1(a))
        a = F.relu(self.conv2(a))
        return self.conv3(a).squeeze(1)

    # ---- likelihood ----------------------------------------------------
    def _log_prob_base(self, x: torch.Tensor, T: Cond, h: Cond) -> torch.Tensor:
        z = self.logits(x, T, h)
        s = self._spins(x).to(z.dtype)
        # log sigmoid(s z) = log P(s_i | .) for s in {-1, +1}
        return F.logsigmoid(s * z).sum(dim=(1, 2))

    def log_prob(self, x: torch.Tensor, T: Cond, h: Cond,
                 z2_equivariant: Optional[bool] = None) -> torch.Tensor:
        """Exact log q(x | T, h) per sample, shape (B,)."""
        z2 = self.z2_equivariant if z2_equivariant is None else z2_equivariant
        lp = self._log_prob_base(x, T, h)
        if not z2:
            return lp
        x = self._spins(x)
        h_t = self._cond(h, x.shape[0], x)
        lp_flip = self._log_prob_base(-x, T, -h_t)
        return torch.logaddexp(lp, lp_flip) - math.log(2.0)

    # ---- sampling ------------------------------------------------------
    @torch.no_grad()
    def _sample_base(self, n: int, L: int, T: Cond, h: Cond) -> torch.Tensor:
        p = next(self.parameters())
        x = torch.zeros(n, L, L, dtype=p.dtype, device=p.device)
        Tt = self._cond(T, n, x)
        ht = self._cond(h, n, x)
        for i in range(L):
            for j in range(L):
                z = self.logits(x, Tt, ht)[:, i, j]
                b = torch.bernoulli(torch.sigmoid(z))
                x[:, i, j] = 2.0 * b - 1.0
        return x

    @torch.no_grad()
    def sample(self, n: int, L: int, T: Cond, h: Cond,
               z2_equivariant: Optional[bool] = None) -> torch.Tensor:
        """Exact ancestral samples in {-1, +1}, shape (n, L, L) (float)."""
        z2 = self.z2_equivariant if z2_equivariant is None else z2_equivariant
        if not z2:
            return self._sample_base(n, L, T, h)
        # q = 1/2 p(.|T,h) + 1/2 [x -> -x]_# p(.|T,-h)
        p = next(self.parameters())
        ref = torch.zeros(n, dtype=p.dtype, device=p.device)
        h_t = self._cond(h, n, ref)
        flip = torch.rand(n, device=p.device) < 0.5
        h_used = torch.where(flip, -h_t, h_t)
        x = self._sample_base(n, L, T, h_used)
        return torch.where(flip.view(n, 1, 1), -x, x)

    # ---- bookkeeping ---------------------------------------------------
    def n_effective_parameters(self) -> int:
        """Parameters that are not identically masked to zero."""
        total = 0
        for m in (self.conv1, self.conv2, self.conv3):
            total += int(m.mask.sum().item()) + m.bias.numel()
        return total
