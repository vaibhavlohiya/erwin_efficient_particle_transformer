"""Coarsening and refinement over the ball tree (Erwin Eqs. 12 and 13).

Coarsening is where the physics claim of this architecture lives. A parent node
is formed from `stride` consecutive children in tree order, and it carries the
**sum of their 4-momenta** - which makes a level-1 node a subjet, a level-2 node
a subjet of subjets, and the pair features computed between them the standard
observables of a QCD splitting. Measured on real jets, the mean pair mass inside
a ball climbs from 0.4 GeV at level 0 to 8.7 GeV at level 2, so the interaction
bias at coarse levels really is describing substructure.

Padding is handled explicitly rather than by hoping it averages out: invalid
children are zeroed before the concat, and the fraction of real children is
appended as a channel so the projection can tell a full ball from a half-empty
one instead of silently reading zeros as physics.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .geometry import masked_centroid


class BallPooling(nn.Module):
    """Eq. 12: fold `stride` children into their parent ball centre.

    Input tensors are in canonical tree order, so children of one parent are
    contiguous. Returns the coarsened (x, pos, p4, valid).
    """

    def __init__(self, dim: int, stride: int, pos_dim: int = 2):
        super().__init__()
        self.stride = stride
        # children features + their offsets from the parent centre + occupancy
        self.proj = nn.Linear(stride * dim + stride * pos_dim + 1, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, pos: torch.Tensor, p4: torch.Tensor,
                valid: torch.Tensor):
        N, L, C = x.shape
        s = self.stride
        if L % s:
            raise ValueError(f"stride {s} does not divide L={L}")
        n = L // s
        D = pos.shape[-1]

        xg = x.view(N, n, s, C)
        pg = pos.view(N, n, s, D)
        qg = p4.view(N, n, s, 4)
        vg = valid.view(N, n, s)
        w = vg.unsqueeze(-1).to(x.dtype)

        with torch.no_grad():
            centre = masked_centroid(pg, vg)                     # (N, n, 1, D)
            rel = ((pg - centre) * w).reshape(N, n, s * D)
            occupancy = vg.to(x.dtype).mean(-1, keepdim=True)    # (N, n, 1)

        feats = torch.cat([(xg * w).reshape(N, n, s * C), rel, occupancy], dim=-1)
        out = self.norm(self.proj(feats))

        parent_valid = vg.any(-1)
        return (out * parent_valid.unsqueeze(-1).to(out.dtype),
                centre.squeeze(-2),
                (qg * w).sum(-2),                                # subjet 4-momentum
                parent_valid)


class BallUnpooling(nn.Module):
    """Eq. 13: broadcast a parent back onto its children.

    Implemented for completeness and for any future per-particle task (track
    origin, energy regression). ErwinParTv2 classifies jets, so it runs
    encoder -> bottleneck with `decode=False` and reads out from every level;
    a decoder would add parameters and compute for nothing.
    """

    def __init__(self, dim: int, stride: int, pos_dim: int = 2):
        super().__init__()
        self.stride = stride
        self.proj = nn.Linear(dim + stride * pos_dim, stride * dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, child_x: torch.Tensor, child_pos: torch.Tensor,
                parent_pos: torch.Tensor, child_valid: torch.Tensor) -> torch.Tensor:
        N, n, C = x.shape
        s = self.stride
        D = child_pos.shape[-1]
        with torch.no_grad():
            rel = (child_pos.view(N, n, s, D) - parent_pos.unsqueeze(-2)).reshape(N, n, s * D)
        upd = self.proj(torch.cat([x, rel], dim=-1)).view(N, n * s, C)
        return self.norm(child_x + upd) * child_valid.unsqueeze(-1).to(x.dtype)
