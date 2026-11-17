"""Fixed-point quantization-aware training for ErwinParTLite.

Emulates hls4ml's `ap_fixed<W, I>` (W bits in all, I of them integer bits
including the sign; `ap_ufixed` for unsigned): round to nearest, saturate at the
type's range, and let the gradient through unchanged inside the range (the
straight-through estimator) and not at all outside it.

What is quantized (`quantize_model`):
  * every Linear's weights (W bits, I from the weight's own max, so a power-of-
    two scale) and its input activations (W bits, I calibrated);
  * the residual stream after every block;
  * the attention path - Q, K, V, the assembled pair + distance bias U_ij, the
    scores and the softmax probabilities - at max(W, attn_floor) bits. As in a
    hardware softmax, each row's maximum is subtracted first, so the scores are
    <= 0 and take the fixed type ap_fixed<., 5> ([-16, 16)): a key 16 below the
    row's best has weight e^-16 and saturating it costs nothing. The
    probabilities are unsigned with no integer bits: ap_ufixed<W, 0> in [0, 1).
Biases stay at accumulator precision (in hardware they are added into the
wide accumulator, not stored as W-bit values). The pair features themselves,
the norms' arithmetic and the Morton keys are not quantized here; BatchNorm
folds into the neighbouring Linear at inference, so quantizing the Linear's
input after it is the deployed arrangement up to that fold.

The integer bits of each activation are set once by `calibrate`: the largest
per-batch 99.99th percentile of |x| over a few batches of real jets, so rare
outliers saturate rather than cost resolution everywhere - the usual way an
ap_fixed type is chosen for a layer. (So even at 24 bits a rare jet differs
from the float model; percentile=1.0 calibrates to the max instead.)

BOPs (bit operations, MACs x weight bits x activation bits) are counted by
`count_bops` on one jet: Linear layers and the two attention matmuls. Pair
features, norms and softmax are left out.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

NEG = -1e9


class QConfig:
    def __init__(self, bits: int, attn_floor: int = 0, percentile: float = 0.9999):
        self.bits = bits
        self.percentile = percentile          # calibration: 1.0 = the max, nothing saturates
        self.attn_bits = max(bits, attn_floor)
        self.enabled = True
        self.calibrating = False
        self.counting = False
        self.bops = 0.0

    def __repr__(self):
        return f"QConfig(bits={self.bits}, attn_bits={self.attn_bits})"


def int_bits_for(m: float, signed: bool) -> int:
    """Smallest I such that ap_(u)fixed<., I> holds magnitude m."""
    if m <= 0:
        return 1 if signed else 0
    return math.floor(math.log2(m)) + (2 if signed else 1)


def fake_fixed(x: torch.Tensor, bits: int, I: int, signed: bool = True) -> torch.Tensor:
    """Round x to ap_fixed<bits, I> (or ap_ufixed), saturating; straight-through gradient."""
    s = 2.0 ** (bits - I)
    if signed:
        lo, hi = -(2.0 ** (I - 1)), 2.0 ** (I - 1) - 1 / s
    else:
        lo, hi = 0.0, 2.0 ** I - 1 / s
    xc = x.clamp(lo, hi)
    return xc + (torch.round(xc * s) / s - xc).detach()


FIXED_INT_BITS = {"prob": 0, "score": 5}       # types set by construction, not calibrated


class FixedQ(nn.Module):
    """One activation quantizer. group 'default' uses cfg.bits; 'attn', 'score' and
    'prob' use cfg.attn_bits, the last two with fixed types (FIXED_INT_BITS)."""

    def __init__(self, cfg: QConfig, group: str = "default"):
        super().__init__()
        self.cfg, self.group = cfg, group
        self.signed = group != "prob"
        self.register_buffer("int_bits", torch.tensor(float(FIXED_INT_BITS.get(group, float("nan")))))
        self._seen: list[float] = []

    def bits(self) -> int:
        return self.cfg.bits if self.group == "default" else self.cfg.attn_bits

    def forward(self, x):
        c = self.cfg
        if not c.enabled:
            return x
        if c.calibrating:
            if self.group not in FIXED_INT_BITS:
                a = x.detach().abs().flatten().float().cpu()
                a = a[a < 1e8]                                   # masked scores are not data
                if a.numel():
                    if a.numel() > 500_000:          # the 99.99th percentile still rests on 50 values
                        a = a[torch.randint(0, a.numel(), (500_000,))]
                    k = max(1, int(round(c.percentile * a.numel())))
                    self._seen.append(float(a.kthvalue(k).values))
            return x
        return fake_fixed(x, self.bits(), int(self.int_bits), self.signed)

    def finish_calibration(self):
        if self.group not in FIXED_INT_BITS:
            self.int_bits.fill_(int_bits_for(max(self._seen) if self._seen else 0.0, True))
        self._seen = []


class QLinear(nn.Linear):
    """nn.Linear with fixed-point weights and input (installed by class swap, so a
    full-precision checkpoint loads first and the parameters are kept)."""

    qcfg: QConfig
    in_q: FixedQ

    def qweight(self):
        w = self.weight
        return fake_fixed(w, self.qcfg.bits, int_bits_for(float(w.detach().abs().max()), True))

    def forward(self, x):
        c = self.qcfg
        if not c.enabled:
            return F.linear(x, self.weight, self.bias)
        x = self.in_q(x)
        if c.counting:
            c.bops += (x.numel() / self.in_features) * self.in_features * self.out_features * c.bits * c.bits
        return F.linear(x, self.qweight(), self.bias)


class AttnQ(nn.Module):
    """The attention path's quantizers for one attention module."""

    def __init__(self, cfg: QConfig):
        super().__init__()
        self.cfg = cfg
        self.q, self.k, self.v = FixedQ(cfg, "attn"), FixedQ(cfg, "attn"), FixedQ(cfg, "attn")
        self.bias, self.score = FixedQ(cfg, "attn"), FixedQ(cfg, "score")
        self.prob = FixedQ(cfg, "prob")

    def attend(self, q, k, v, bias, key_valid, dropout_p, training):
        """softmax(QK^T/sqrt(d) + bias) V with every step at fixed point.
        q (B, H, m, d); k, v (B, H, K, d); bias (B, H, m, K) unmasked or None;
        key_valid bool broadcastable to (B, H, m, K). Masking is applied after the
        score is quantized: a masked key is skipped, never a saturated value."""
        c = self.cfg
        q, k, v = self.q(q), self.k(k), self.v(v)
        s = (q @ k.transpose(-1, -2)) * (q.shape[-1] ** -0.5)
        if bias is not None:
            s = s + self.bias(bias)
        s = s.masked_fill(~key_valid, NEG)
        s = s - s.amax(-1, keepdim=True).detach()       # the row max, as a hardware softmax does
        s = self.score(s).masked_fill(~key_valid, NEG)  # re-mask: a masked key saturated at -16
        p = self.prob(torch.softmax(s, -1))
        if training and dropout_p:
            p = F.dropout(p, dropout_p)
        if c.enabled and c.counting:
            macs = q.shape[0] * q.shape[1] * q.shape[2] * k.shape[2] * q.shape[3]
            c.bops += 2 * macs * c.attn_bits * c.attn_bits
        return p @ v


def quantize_model(model: nn.Module, bits: int, attn_floor: int = 0, percentile: float = 0.9999) -> QConfig:
    """Install fixed-point emulation in place; returns the shared QConfig."""
    from .lite_model import ClsBlock, LiteBlock, LiteMSA

    cfg = QConfig(bits, attn_floor, percentile)
    for mod in list(model.modules()):
        if type(mod) is nn.Linear:
            mod.__class__ = QLinear
            mod.qcfg = cfg
            mod.in_q = FixedQ(cfg)
        if isinstance(mod, (LiteMSA, ClsBlock)):
            mod.aq = AttnQ(cfg)
        if isinstance(mod, LiteBlock):
            mod.out_q = FixedQ(cfg)
    dev = next(model.parameters()).device
    model.to(dev)                                       # new buffers onto the model's device
    return cfg


def quantizers(model: nn.Module):
    return [m for m in model.modules() if isinstance(m, FixedQ)]


@torch.no_grad()
def calibrate(model: nn.Module, cfg: QConfig, batches) -> dict:
    """Set every activation's integer bits from `batches` (callables running one
    forward each). Returns a histogram of the chosen integer bits."""
    was = model.training
    model.eval()
    cfg.calibrating = True
    for run in batches:
        run()
    cfg.calibrating = False
    hist: dict = {}
    for q in quantizers(model):
        q.finish_calibration()
        k = f"{q.group}:{int(q.int_bits)}"
        hist[k] = hist.get(k, 0) + 1
    model.train(was)
    return dict(sorted(hist.items()))


@torch.no_grad()
def count_bops(model: nn.Module, cfg: QConfig, run) -> float:
    """BOPs of one forward (run() should score a single jet)."""
    was = model.training
    model.eval()
    cfg.counting, cfg.bops = True, 0.0
    run()
    cfg.counting = False
    model.train(was)
    return cfg.bops


def weight_bits(model: nn.Module, cfg: QConfig) -> dict:
    """On-chip storage: Linear weights at cfg.bits, everything else (biases, norms,
    tokens) at 32 bits."""
    wq = sum(m.weight.numel() for m in model.modules() if isinstance(m, QLinear))
    rest = sum(p.numel() for p in model.parameters()) - wq
    return {"weights_quantized": wq, "other_params": rest,
            "kbit_total": (wq * cfg.bits + rest * 32) / 1e3}
