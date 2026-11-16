"""Pair features for ErwinParTLite: exact (ParT's) or lookup-table (FPGA-friendly).

'exact' is weaver's `pairwise_lv_fts`, bit-identical to ParT and ErwinParTv2:
[ln kT, ln z, ln dR, ln m^2] in rapidity, from the 4-vectors.

'lut' computes the same four quantities with only additions, multiplications,
comparisons and one table lookup per logarithm - no atan2, no exp, no general
log or sqrt:

* positions come from the inputs' (delta_eta, delta_phi), which JetClass ships
  already wrapped about the jet axis, so no azimuth is recomputed;
* ln x = ln2 * (e + log2 m), with x = m * 2^e from the float's exponent and
  log2 m read from a 2^bits-entry table on m in [0.5, 1) (linear interpolation)
  - in hardware a leading-zero count and a small ROM;
* ln dR = 0.5 ln(dR^2), so no square root; pT enters as ln pT = 0.5 ln(px^2 + py^2);
* the pair mass needs no transcendental function at all: m^2 = (E_i + E_j)^2 -
  |p_i + p_j|^2 is multiply-adds, so it is computed exactly and only its log
  goes through the table. (A small-angle form pT_i pT_j dR^2 drops the
  particles' own masses and is off by a factor ~3 for soft collinear pairs.)

Everything here is torch-only and torch.fx-traceable.
"""
import math

import torch

EPS = 1e-8
LN2 = math.log(2.0)


def _log2_table(bits: int) -> torch.Tensor:
    """log2(m) at 2^bits + 1 evenly spaced m in [0.5, 1] (the last for interpolation)."""
    m = 0.5 + 0.5 * torch.arange(2 ** bits + 1, dtype=torch.float64) / 2 ** bits
    return torch.log2(m).to(torch.float32)


def _frexp(x: torch.Tensor):
    """x = m * 2^e with m in [0.5, 1), read from float32 bit fields (x > 0, normal).

    This is what the hardware does - the exponent is a shift and a mask, the
    mantissa the low 23 bits with the exponent forced to 2^-1 - and unlike
    torch.frexp it runs on every backend (MPS has no frexp) and traces in fx."""
    bits = x.float().contiguous().view(torch.int32)
    e = ((bits >> 23) & 0xFF) - 126
    m = ((bits & 0x007FFFFF) | (126 << 23)).view(torch.float32)
    return m, e


def lut_log(x: torch.Tensor, table: torch.Tensor, bits: int) -> torch.Tensor:
    """Natural log through the exponent and a mantissa table, x clamped to EPS."""
    m, e = _frexp(x.clamp(min=EPS))                      # x = m * 2^e, m in [0.5, 1)
    pos = (m - 0.5) * (2 * 2 ** bits)                    # 0 .. 2^bits
    i = pos.floor().clamp(0, 2 ** bits - 1)
    frac = pos - i
    i = i.long()
    lo, hi = table[i], table[i + 1]
    return LN2 * (e.to(x.dtype) + lo + frac * (hi - lo))


def lut_features(pos_i, pos_j, lnpt_i, lnpt_j, pt_i, pt_j, table, bits, p4_i, p4_j):
    """[ln kT, ln z, ln dR, ln m^2] from positions, pT and 4-momenta, with table logs."""
    d = pos_i - pos_j
    dr2 = (d * d).sum(-1)
    ln_dr = 0.5 * lut_log(dr2, table, bits)
    ln_min = torch.minimum(lnpt_i, lnpt_j)
    ln_sum = lut_log(pt_i + pt_j, table, bits)
    s = p4_i + p4_j                                      # (px, py, pz, E) of the pair
    m2 = s[..., 3] * s[..., 3] - s[..., 0] * s[..., 0] - s[..., 1] * s[..., 1] - s[..., 2] * s[..., 2]
    return torch.stack([ln_min + ln_dr,                  # ln kT = ln min(pT) + ln dR
                        ln_min - ln_sum,                 # ln z
                        ln_dr,
                        lut_log(m2, table, bits)],       # ln m^2, exact m^2 by multiply-adds
                       dim=-1)


def lut_pt(p4: torch.Tensor, table, bits):
    """(pT, ln pT) from (..., 4) momenta. pT is an input-side quantity (a Level-1
    trigger receives it with each candidate); only its log comes from the table."""
    pt2 = p4[..., 0] * p4[..., 0] + p4[..., 1] * p4[..., 1]
    return pt2.clamp(min=EPS).sqrt(), 0.5 * lut_log(pt2, table, bits)


def exact_features(p4_i: torch.Tensor, p4_j: torch.Tensor) -> torch.Tensor:
    """ParT's own [ln kT, ln z, ln dR, ln m^2] for (..., 4) pairs -> (..., 4)."""
    from weaver.nn.model.ParticleTransformer import pairwise_lv_fts
    p4_i, p4_j = p4_i + torch.zeros_like(p4_j), p4_j + torch.zeros_like(p4_i)   # broadcast
    xi = p4_i.reshape(-1, 4).t().unsqueeze(0)            # (1, 4, n)
    xj = p4_j.reshape(-1, 4).t().unsqueeze(0)
    f = pairwise_lv_fts(xi, xj, num_outputs=4)           # (1, 4, n)
    return f[0].t().reshape_as(p4_i)                     # 4 features <-> 4 components
