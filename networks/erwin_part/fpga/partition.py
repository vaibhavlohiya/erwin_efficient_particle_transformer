"""Fixed-size tile layouts for ErwinParTLite, without a recursive ball tree.

Every layout returns a permutation of the L slots such that consecutive groups
of m slots form one tile. All are torch.fx-traceable (constants only, no data-
dependent Python control flow).

'morton'  Z-order: quantize (delta_eta, delta_phi) to a 2^bits x 2^bits grid over
          the jet cone, interleave the bits of the two cell indices, and sort by
          that key. Consecutive Z-order runs are quadtree cells, so tiles - and
          the pairs merged by pooling - are spatial neighbours. One sort of L keys
          (a bitonic network in hardware) replaces the median tree's recursive
          sorts. `shift=True` offsets the grid by half a cell: alternate blocks use
          it to mix across tile edges, in place of ErwinParTv2's rotated tree.
'pt'      the input's pT order unchanged: tiles of consecutive pT rank, no sorting
          logic at all. Tiles are not spatial neighbourhoods, so the pooled "subjets"
          of this layout are not physical; it is the comparison baseline (a flat
          pT-patch model, PHAT-JeT-like), not a hierarchy layout.
"""
import torch

R_JET = 0.8                     # JetClass anti-kT R = 0.8


def morton_key(pos: torch.Tensor, valid: torch.Tensor, bits: int = 5,
               shift: bool = False, radius: float = R_JET) -> torch.Tensor:
    """(N, L, 2), (N, L) -> (N, L) int64 Z-order key; padding sorts last."""
    n = 2 ** bits
    cell = 2 * radius / n
    off = 0.5 * cell if shift else 0.0
    q = ((pos + radius + off) / cell).floor().clamp(0, n - 1).long()   # (N, L, 2)
    x, y = q[..., 0], q[..., 1]
    key = torch.zeros_like(x)
    for b in range(bits):                                             # constant loop
        key = key | (((x >> b) & 1) << (2 * b)) | (((y >> b) & 1) << (2 * b + 1))
    return torch.where(valid, key, torch.full_like(key, n * n))


def morton_order(pos: torch.Tensor, valid: torch.Tensor, bits: int = 5,
                 shift: bool = False) -> torch.Tensor:
    """Permutation into Z-order; ties keep the input (pT) order."""
    return torch.argsort(morton_key(pos, valid, bits, shift), dim=1, stable=True)


def gather_slots(t: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """Select/reorder (N, P, C) along dim 1 by (N, L) indices, L <= P -> (N, L, C).
    take_along_dim broadcasts the index, so no shape is unpacked (fx-traceable)."""
    return torch.take_along_dim(t, perm.unsqueeze(-1), dim=1)
