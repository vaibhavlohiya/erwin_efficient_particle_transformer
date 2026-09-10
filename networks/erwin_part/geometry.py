"""Jet geometry for ErwinParTv2.

Everything the ball tree and the pair features need in order to see a jet the way
a physicist does: angular coordinates measured in the jet frame, with the
azimuthal angle wrapped, plus a fixed-length slot layout that carries an explicit
validity mask instead of relying on duplicated padding.

Layout convention: this module works in (N, P, ...) - batch, particle, channel.
The (N, 4, P) Lorentz tensor weaver hands to the model and the (P, N, C) feature
tensor ParT's Embed produces are converted at the call site, not here.

Two rules this module exists to enforce:

1.  phi is periodic. `atan2(py, px)` returns absolute phi in (-pi, pi], so a jet
    straddling the +/-pi boundary has constituents at both ends of that interval.
    Any mean, distance, or tree split taken on absolute phi is then wrong by up
    to 2*pi. All angular quantities here are differences against the jet axis,
    wrapped back into (-pi, pi].

2.  Padded slots are not particles. They are excluded from the jet axis, from
    ball centroids, and from attention. Every function that reduces over
    particles takes `valid` and honours it.
"""
from __future__ import annotations

import math

import torch

TWO_PI = 2.0 * math.pi
EPS = 1e-8


def wrap_phi(dphi: torch.Tensor) -> torch.Tensor:
    """Wrap an azimuthal *difference* into (-pi, pi]."""
    return (dphi + math.pi) % TWO_PI - math.pi


def eta_phi_pt(p4: torch.Tensor):
    """(..., 4) [px, py, pz, E] -> (eta, phi, pt), each (...,).

    pt is clamped away from zero so that a zero-momentum padded slot yields
    finite (but meaningless) angles rather than NaN; callers mask them out.
    """
    px, py, pz = p4[..., 0], p4[..., 1], p4[..., 2]
    pt = torch.sqrt(px * px + py * py).clamp(min=EPS)
    eta = torch.asinh(pz / pt)
    phi = torch.atan2(py, px)
    return eta, phi, pt


def jet_axis(p4: torch.Tensor, valid: torch.Tensor):
    """Jet axis as the direction of the summed 4-momentum of the real particles.

    Args:
        p4:    (N, P, 4) constituent 4-momenta.
        valid: (N, P) bool, True for real particles.
    Returns:
        (eta_jet, phi_jet), each (N,).

    Summing 4-momenta - rather than averaging angles - is what makes this the
    physical jet axis: it is linear in the constituents, dominated by the hard
    ones, and immune to the phi periodicity that breaks an angular mean.
    """
    p4_sum = (p4 * valid.unsqueeze(-1).to(p4.dtype)).sum(dim=1)   # (N, 4)
    eta_jet, phi_jet, _ = eta_phi_pt(p4_sum)
    return eta_jet, phi_jet


def relative_positions(p4: torch.Tensor, valid: torch.Tensor,
                       axis: tuple[torch.Tensor, torch.Tensor] | None = None,
                       eta_sign_convention: bool = True):
    """(N, P, 4) -> (N, P, 2) angular position (delta_eta, delta_phi) in the jet frame.

    This is the metric space the ball tree partitions in and the distance bias
    of Eq. 10 is measured in. Padded slots are zeroed so they cannot drag a
    centroid; they are still masked downstream.

    `eta_sign_convention` reproduces JetClass's own definition,

        part_deta = (part_eta - jet_eta) * sign(jet_eta)

    verified against the shipped branches to 2.4e-7. The flip folds in the
    forward/backward (eta -> -eta) symmetry of a pp collision, so a jet in the
    -eta hemisphere is represented identically to its mirror image in +eta.
    Keep it on: the model is fed `pf_points` = (part_deta, part_dphi), so a
    tree built on the unflipped sign would partition in a different frame from
    the one the features describe.
    """
    eta, phi, _ = eta_phi_pt(p4)
    eta_jet, phi_jet = axis if axis is not None else jet_axis(p4, valid)

    deta = eta - eta_jet.unsqueeze(1)
    if eta_sign_convention:
        # sign(0) is 0, which would erase delta_eta; treat eta_jet == 0 as +1.
        deta = deta * torch.where(eta_jet >= 0, 1.0, -1.0).unsqueeze(1).to(deta.dtype)
    dphi = wrap_phi(phi - phi_jet.unsqueeze(1))
    pos = torch.stack([deta, dphi], dim=-1)                        # (N, P, 2)
    return pos * valid.unsqueeze(-1).to(pos.dtype)


def delta_r(pos_a: torch.Tensor, pos_b: torch.Tensor) -> torch.Tensor:
    """Angular separation between two sets of (delta_eta, delta_phi) points.

    Both inputs are already jet-frame differences, so the phi component is a
    difference of differences and must be re-wrapped.
    """
    deta = pos_a[..., 0] - pos_b[..., 0]
    dphi = wrap_phi(pos_a[..., 1] - pos_b[..., 1])
    return torch.sqrt(deta * deta + dphi * dphi + EPS)


def sort_truncate_pad(valid: torch.Tensor, pt: torch.Tensor, length: int):
    """Fixed-length slot layout: keep the `length` hardest constituents per jet.

    Returns:
        idx:     (N, length) long - gather index into the original P axis.
        valid_L: (N, length) bool - True where the slot holds a real particle.

    Sorting by descending pt means truncation drops the softest constituents,
    which is what ParT already does at its own 128-slot limit. Padded slots keep
    a well-defined index (0) so gathers stay in bounds; `valid_L` is what makes
    them harmless.
    """
    key = torch.where(valid, pt, torch.full_like(pt, float("-inf")))
    order = torch.argsort(key, dim=1, descending=True)             # (N, P)

    P = valid.shape[1]
    if P >= length:
        idx = order[:, :length]
        valid_L = torch.gather(valid, 1, idx)
    else:
        pad = length - P
        idx = torch.cat([order, order.new_zeros(order.shape[0], pad)], dim=1)
        valid_L = torch.cat(
            [torch.gather(valid, 1, order),
             valid.new_zeros(valid.shape[0], pad)], dim=1)
    return idx, valid_L


def take_slots(t: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather (N, P, C) -> (N, L, C) along the particle axis using `idx` (N, L)."""
    return torch.gather(t, 1, idx.unsqueeze(-1).expand(-1, -1, t.shape[-1]))


def masked_centroid(pos: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Centroid c_B of Eq. 9, over real members only.

    Args:
        pos:   (..., m, D)
        valid: (..., m) bool
    Returns:
        (..., 1, D)

    An all-invalid group returns the zero vector rather than NaN; its rows are
    masked out of attention anyway.
    """
    w = valid.unsqueeze(-1).to(pos.dtype)
    return (pos * w).sum(dim=-2, keepdim=True) / w.sum(dim=-2, keepdim=True).clamp(min=1.0)
