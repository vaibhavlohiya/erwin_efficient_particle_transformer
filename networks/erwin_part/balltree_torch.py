"""A batched, fixed-length ball tree in pure PyTorch.

Why not use the `balltree` C++/Cython package that Erwin ships with:

  * It pads each point cloud to next_pow2(n) individually, so shapes are ragged
    across a batch. Jet multiplicity varies (median 39, p95 62), so that means a
    different tensor shape almost every step - which forbids torch.compile,
    complicates masking, and buys nothing here.
  * ErwinParTv2 pads every jet to a fixed L (64 by default), so the tree is the
    same size for every jet in the batch and the whole construction collapses to
    a handful of batched sorts.
  * It is a compiled dependency that is absent from the weaver environment.

This module is a drop-in replacement for that one job. `balltree` stays as the
verification oracle in tests/s2_tree_parity.py; nothing in the training path
imports it.

The construction is Erwin's: recursively split each node along its
widest-spread coordinate at the median, until leaves hold `min_leaf` points.
The result is returned as a permutation, which gives the property the whole
design rests on - **points in the same ball are contiguous in memory at every
level** - so ball attention is a `.view`, and pooling is a strided reduction.
"""
from __future__ import annotations

import math

import torch


def rotate_positions(pos: torch.Tensor, degrees: float) -> torch.Tensor:
    """Rotate 2-D positions about the jet axis.

    Erwin's cross-ball mechanism: balls are disjoint, so alternating blocks use a
    tree built on rotated coordinates, and information crosses ball boundaries
    the way it does between shifted windows in a Swin Transformer.

    A rotation in the (delta_eta, delta_phi) plane is not a physical symmetry -
    it does not need to be. It only needs to produce a *different valid
    partition* of the same points.
    """
    if degrees % 360 == 0:
        return pos
    th = math.radians(degrees)
    c, s = math.cos(th), math.sin(th)
    rot = torch.tensor([[c, -s], [s, c]], dtype=pos.dtype, device=pos.device)
    return pos @ rot.transpose(0, 1)


def build_ball_tree(pos: torch.Tensor, valid: torch.Tensor,
                    min_leaf: int = 1) -> torch.Tensor:
    """Batched recursive median split.

    Args:
        pos:      (N, L, D) positions, L a power of two.
        valid:    (N, L) bool. Invalid slots are pushed to the end of every node
                  they belong to, so padding concentrates in the last balls
                  instead of being smeared across all of them.
        min_leaf: recursion stops when nodes hold this many points.

    Returns:
        perm: (N, L) long. `pos.gather(1, perm)` is the tree ordering; slicing it
        into contiguous chunks of any power-of-two size yields that level's balls.

    Ties: sorts are stable, so points with an equal split coordinate keep their
    incoming order. Callers feed slots in descending-pt order, which makes the
    tie-break deterministic and independent of the order particles arrived in.
    Exact float ties between distinct constituents do not occur in practice.
    """
    N, L, D = pos.shape
    if L & (L - 1):
        raise ValueError(f"L must be a power of two, got {L}")
    if min_leaf & (min_leaf - 1) or min_leaf < 1:
        raise ValueError(f"min_leaf must be a power of two, got {min_leaf}")

    perm = torch.arange(L, device=pos.device).expand(N, L).contiguous()
    big = torch.finfo(pos.dtype).max

    for depth in range(int(math.log2(L // min_leaf))):
        groups, size = 1 << depth, L >> depth

        p = torch.gather(pos, 1, perm.unsqueeze(-1).expand(N, L, D)).view(N, groups, size, D)
        v = torch.gather(valid, 1, perm).view(N, groups, size)
        vp = v.unsqueeze(-1)

        # Widest-spread axis, measured over real points only: a padded slot
        # sitting at the origin would otherwise inflate the apparent extent.
        lo = torch.where(vp, p, torch.full_like(p, big)).amin(dim=2)
        hi = torch.where(vp, p, torch.full_like(p, -big)).amax(dim=2)
        split_dim = (hi - lo).argmax(dim=-1)                       # (N, groups)

        key = torch.gather(
            p, 3, split_dim.view(N, groups, 1, 1).expand(N, groups, size, 1)
        ).squeeze(-1)                                              # (N, groups, size)
        key = torch.where(v, key, torch.full_like(key, float("inf")))

        order = torch.argsort(key, dim=2, stable=True)
        perm = torch.gather(perm.view(N, groups, size), 2, order).view(N, L)

    return perm


def ball_view(perm: torch.Tensor, ball_size: int) -> torch.Tensor:
    """(N, L) tree permutation -> (N, n_balls, ball_size) membership."""
    N, L = perm.shape
    if L % ball_size:
        raise ValueError(f"ball_size {ball_size} does not divide L={L}")
    return perm.view(N, L // ball_size, ball_size)


def build_levels(pos: torch.Tensor, valid: torch.Tensor,
                 ball_sizes: list[int], strides: list[int],
                 rotate: float = 45.0):
    """Everything the encoder needs, built once per forward pass.

    Returns a list with one dict per level, coarsest last:
        perm        (N, L_l)            tree order at this level
        perm_rot    (N, L_l) | None     tree order under the rotated partition
        valid       (N, L_l)            per-node validity
        n_balls     int
    Level l+1 has L_l // strides[l] nodes; a parent is valid iff any child is.
    """
    if len(strides) != len(ball_sizes) - 1:
        raise ValueError("len(strides) must be len(ball_sizes) - 1")

    levels, cur_pos, cur_valid = [], pos, valid
    for li, ball_size in enumerate(ball_sizes):
        L = cur_pos.shape[1]
        if L % ball_size:
            raise ValueError(
                f"level {li}: ball_size {ball_size} does not divide L={L}")

        perm = build_ball_tree(cur_pos, cur_valid, min_leaf=1)
        perm_rot = (build_ball_tree(rotate_positions(cur_pos, rotate), cur_valid,
                                    min_leaf=1) if rotate % 360 else None)

        # Blocks hold features in canonical tree order, so what they need is the
        # permutation *from* that order into rotated order (and back), not the
        # two absolute orderings.
        if perm_rot is None:
            rel = rel_inv = None
        else:
            perm_inv = torch.argsort(perm, dim=1)
            rel = torch.gather(perm_inv, 1, perm_rot)
            rel_inv = torch.argsort(rel, dim=1)

        levels.append({
            "perm": perm,
            "perm_rot": perm_rot,
            "rot_rel": rel,
            "rot_rel_inv": rel_inv,
            "valid": torch.gather(cur_valid, 1, perm),
            "ball_size": ball_size,
            "n_balls": L // ball_size,
        })

        if li == len(ball_sizes) - 1:
            break

        # Coarsen: consecutive `stride` nodes in tree order become one parent.
        stride = strides[li]
        N, D = cur_pos.shape[0], cur_pos.shape[2]
        ordered_pos = torch.gather(cur_pos, 1, perm.unsqueeze(-1).expand(N, L, D))
        ordered_valid = torch.gather(cur_valid, 1, perm)

        gp = ordered_pos.view(N, L // stride, stride, D)
        gv = ordered_valid.view(N, L // stride, stride)
        w = gv.unsqueeze(-1).to(gp.dtype)
        # Parent position is the centroid of its *real* children (Erwin coarsens
        # by mean); an all-padding parent gets the origin and is marked invalid.
        cur_pos = (gp * w).sum(dim=2) / w.sum(dim=2).clamp(min=1.0)
        cur_valid = gv.any(dim=2)

    return levels
