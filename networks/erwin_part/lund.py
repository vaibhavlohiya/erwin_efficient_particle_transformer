"""Lund-plane splitting tokens (PLuM, arXiv:2605.26821) for ErwinParTv2.

Every merge in a sequential reclustering of the jet is one splitting, and each
splitting becomes one token carrying

    [ln kT, ln z, ln deltaR]        softer branch b off harder branch a:
                                    deltaR in (rapidity, phi), z = pt_b/(pt_a+pt_b),
                                    kT = pt_b * deltaR

in the same order and with the same clamps as weaver's `pairwise_lv_fts`, so a
splitting token and a within-ball pair feature for the same two pseudojets are
bit-for-bit the same numbers.

The full Lund tree is the *set* of splittings, and every merge is one node of it
whatever order the tree is walked in. So the declustering-from-the-root pass of
the literature is not needed here: recording each merge as the clustering makes
it yields exactly the same tokens.

Why this is computed in the model and not offline: the JetClass volume is
read-only on the cluster and weaver's `new_variables` cannot run a clustering,
so precomputing would mean writing a second copy of the dataset. The clustering
is a short chain of argmins under no_grad - the same shape of cost as the ball
tree - and it runs on the model's own L pt-sorted slots, so it sees exactly the
constituents the rest of the network does.

This module depends only on torch, like `geometry` and `balltree_torch`, so the
clustering can be checked against FastJet in an interpreter without weaver
(tests/s12b_lund_fastjet.py).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

EPS = 1e-8

# Exponent p of the generalised-kT distance d_ij = min(pt_i^2p, pt_j^2p) dR_ij^2.
# There is no beam distance: the input is one jet, and reclustering it with an
# effectively infinite R merges everything, which is what FastJet does with R=1000.
ALGORITHMS = {"kt": 1.0, "ca": 0.0}

# (mean, std) of [ln kT, ln z, ln deltaR] over the tokens a jet actually keeps,
# measured on 4000 JetClass jets with L=64, max_splits=48 and no kT cut.
# Recompute with tests/calibrate_lund_norm.py if any of those change.
LUND_FEATURE_NORM = {
    "kt": ((-1.022, -1.574, -3.209), (1.748, 0.715, 1.196)),   # 38.8 tokens/jet
    "ca": ((-1.058, -2.024, -3.205), (1.677, 1.287, 1.246)),
}


def _pt_rap_phi(p4: torch.Tensor):
    """(..., 4) -> pt, rapidity, phi, matching weaver's `to_ptrapphim`."""
    px, py, pz, e = p4.unbind(-1)
    pt = torch.sqrt(px * px + py * py)
    rap = 0.5 * torch.log(1 + (2 * pz) / (e - pz).clamp(min=1e-20))
    return pt, rap, torch.atan2(py, px)


def _wrap(dphi: torch.Tensor) -> torch.Tensor:
    return (dphi + math.pi) % (2 * math.pi) - math.pi


def lund_splittings(p4: torch.Tensor, valid: torch.Tensor,
                    pos: torch.Tensor | None = None, alg: str = "kt",
                    max_splits: int | None = 48, ktcut: float | None = None):
    """Recluster each jet and return its splittings as tokens.

    Args:
        p4:         (N, L, 4) [px, py, pz, E] constituent slots.
        valid:      (N, L) bool.
        pos:        (N, L, 2) jet-frame (delta_eta, delta_phi), optional. A
                    token's position is the pt-weighted centroid of everything
                    its parent pseudojet contains - the same frame `pos` is in.
        alg:        'kt' (PLuM's choice) or 'ca' (angular ordered, LundNet's).
        max_splits: keep this many tokens per jet, hardest kT first. None keeps
                    all L-1.
        ktcut:      drop splittings with ln(kT/GeV) <= ktcut, the notes'
                    robustness knob against the hadronisation model.

    Returns:
        feats (N, S, 3) [ln kT, ln z, ln deltaR], pos (N, S, 2) or None,
        valid (N, S) bool - sorted by descending kT, invalid tokens last.
    """
    if alg not in ALGORITHMS:
        raise ValueError(f"alg must be one of {sorted(ALGORITHMS)}, got {alg!r}")
    expo = ALGORITHMS[alg]
    N, L, _ = p4.shape
    steps = max(L - 1, 0)

    with torch.no_grad():
        w = valid.unsqueeze(-1).to(p4.dtype)
        p = p4 * w
        q = pos * w if pos is not None else None
        # Scalar pt carried by each pseudojet: the weight of its position, so a
        # token sits at the pt-weighted centroid of its constituents exactly.
        wpt = torch.hypot(p[..., 0], p[..., 1])
        active = valid.clone()
        pt, rap, phi = _pt_rap_phi(p)
        rows = torch.arange(N, device=p4.device)
        self_pair = torch.eye(L, dtype=torch.bool, device=p4.device)
        inf = torch.tensor(float("inf"), dtype=p4.dtype, device=p4.device)

        def distance(pt_a, rap_a, phi_a, pt_b, rap_b, phi_b):
            dr2 = (rap_a - rap_b) ** 2 + _wrap(phi_a - phi_b) ** 2
            if expo == 0.0:
                return dr2
            return torch.minimum(pt_a, pt_b).pow(2 * expo) * dr2

        # The full distance matrix is built once; after each merge only the row
        # and column of the merged pseudojet change, so they are all that is
        # recomputed. That keeps a step at O(L) work plus one argmin.
        d = distance(pt.unsqueeze(2), rap.unsqueeze(2), phi.unsqueeze(2),
                     pt.unsqueeze(1), rap.unsqueeze(1), phi.unsqueeze(1))
        pair_ok = active.unsqueeze(2) & active.unsqueeze(1) & ~self_pair
        d = torch.where(pair_ok, d, inf)

        feats = p4.new_zeros(N, steps, 3)
        tok_pos = p4.new_zeros(N, steps, 2) if q is not None else None
        tok_valid = torch.zeros(N, steps, dtype=torch.bool, device=p4.device)

        for s in range(steps):
            dmin, flat = d.view(N, -1).min(dim=1)
            has = torch.isfinite(dmin)
            if not bool(has.any()):
                break
            i, j = flat // L, flat % L

            pi, pj = p[rows, i], p[rows, j]
            pti, ptj = pt[rows, i], pt[rows, j]
            delta = torch.sqrt((rap[rows, i] - rap[rows, j]) ** 2
                               + _wrap(phi[rows, i] - phi[rows, j]) ** 2)
            ptmin = torch.minimum(pti, ptj)
            feats[:, s, 0] = torch.log((ptmin * delta).clamp(min=EPS))
            feats[:, s, 1] = torch.log((ptmin / (pti + ptj).clamp(min=EPS)).clamp(min=EPS))
            feats[:, s, 2] = torch.log(delta.clamp(min=EPS))
            tok_valid[:, s] = has

            if q is not None:
                qi, qj = q[rows, i], q[rows, j]
                wi, wj = wpt[rows, i], wpt[rows, j]
                frac = (wj / (wi + wj).clamp(min=EPS)).unsqueeze(-1)
                diff = torch.stack([qj[:, 0] - qi[:, 0], _wrap(qj[:, 1] - qi[:, 1])], -1)
                qm = qi + frac * diff
                tok_pos[:, s] = qm
                q[rows, i] = torch.where(has.unsqueeze(-1), qm, qi)
                wpt[rows, i] = torch.where(has, wi + wj, wi)

            # E-scheme recombination into slot i; slot j is retired.
            pm = torch.where(has.unsqueeze(-1), pi + pj, pi)
            p[rows, i] = pm
            active[rows, j] = active[rows, j] & ~has
            ptm, rapm, phim = _pt_rap_phi(pm)
            pt[rows, i], rap[rows, i], phi[rows, i] = ptm, rapm, phim

            row = distance(ptm.unsqueeze(1), rapm.unsqueeze(1), phim.unsqueeze(1),
                           pt, rap, phi)                                    # (N, L)
            keep = active & active[rows, i].unsqueeze(1)
            keep[rows, i] = False
            row = torch.where(keep, row, inf)
            hcol = has.unsqueeze(1)
            d[rows, i, :] = torch.where(hcol, row, d[rows, i, :])
            d[rows, :, i] = torch.where(hcol, row, d[rows, :, i])
            d[rows, j, :] = torch.where(hcol, inf, d[rows, j, :])
            d[rows, :, j] = torch.where(hcol, inf, d[rows, :, j])

        if ktcut is not None:
            tok_valid = tok_valid & (feats[..., 0] > ktcut)

        # Hardest first, so truncation drops the softest splittings - the ones a
        # kT cut would remove anyway, and the least perturbative.
        key = torch.where(tok_valid, feats[..., 0], torch.full_like(feats[..., 0], -inf))
        S = steps if max_splits is None else min(max_splits, steps)
        order = torch.argsort(key, dim=1, descending=True)[:, :S]
        feats = torch.gather(feats, 1, order.unsqueeze(-1).expand(-1, -1, 3))
        tok_valid = torch.gather(tok_valid, 1, order)
        if tok_pos is not None:
            tok_pos = torch.gather(tok_pos, 1, order.unsqueeze(-1).expand(-1, -1, 2))
            tok_pos = tok_pos * tok_valid.unsqueeze(-1).to(tok_pos.dtype)
        feats = feats * tok_valid.unsqueeze(-1).to(feats.dtype)
    return feats, tok_pos, tok_valid


class LundEmbed(nn.Module):
    """Splitting features -> token embedding, PLuM's 3 -> 64 -> 256 -> C MLP.

    Same layer pattern as ParT's `Embed` - LayerNorm, Linear, GELU per layer -
    except that the first layer has no LayerNorm. Over three input features a
    LayerNorm would subtract their mean and divide out their spread, discarding
    two of the three degrees of freedom; ParT can afford it only because its
    particle input is 17-wide. The input is instead standardised with fixed
    constants from data, as the pair features are (pair_features.py).
    """

    def __init__(self, dims: list[int], alg: str = "kt", activation: str = "gelu"):
        super().__init__()
        mean, std = LUND_FEATURE_NORM[alg]
        self.register_buffer("mean", torch.tensor(mean))
        self.register_buffer("std", torch.tensor(std))
        layers: list[nn.Module] = []
        in_dim = len(mean)
        for k, dim in enumerate(dims):
            if k:
                layers.append(nn.LayerNorm(in_dim))
            layers += [nn.Linear(in_dim, dim),
                       nn.GELU() if activation == "gelu" else nn.ReLU()]
            in_dim = dim
        self.embed = nn.Sequential(*layers)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """(N, S, 3) -> (N, S, C)."""
        return self.embed((feats - self.mean) / self.std)
