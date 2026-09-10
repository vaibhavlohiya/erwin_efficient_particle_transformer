#!/usr/bin/env python
"""
Task 1 — Complexity benchmark for the Erwin / BallMSA attention block.

Isolates `ParticleErwinBlock` (which internally runs `ErwinTransformerBlock`)
and pushes dummy Lorentz vectors + features of increasing sequence length
n = 50, 100, 200, 400 (extendable) through it. For each n we measure:

  * forward-pass wall-clock time
  * peak resident memory attributable to the forward pass

A faithful full O(n^2) self-attention block is run on the exact same inputs as a
reference. It *explicitly* materialises the (batch, heads, n, n) score matrix and
softmaxes it -- this is what ParT's pairwise-interaction attention does in this
repo, and it is the cost the Erwin block is designed to avoid. (We deliberately
do NOT use the fused `scaled_dot_product_attention` path for the baseline, since
that kernel hides the n^2 memory and defeats the comparison.) Fitting a power law
to each curve shows the Erwin block scales ~ n^1 (cost is O(n * ball_size)) while
full attention scales ~ n^2. Results are printed as a table and saved to
`erwin_complexity.png`.

Run from the repo root with the weaver env:
    /opt/homebrew/Caskroom/miniconda/base/envs/weaver/bin/python benchmark_erwin_complexity.py
    # or, if `weaver` is your active env:
    python benchmark_erwin_complexity.py
"""
import argparse
import gc
import statistics
import time

import numpy as np
import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from networks.EfficientParticleTransformer import ParticleErwinBlock, _compute_positions


class FullAttention(nn.Module):
    """Reference O(n^2) self-attention that EXPLICITLY builds the n x n score
    matrix (batch, heads, n, n) and softmaxes it -- the same quadratic cost as
    ParT's pairwise-bias attention. Not fused, so the n^2 time and memory are
    real and measurable. Input/output are (P, N, C) like the Erwin block."""

    def __init__(self, dim, num_heads):
        super().__init__()
        self.h = num_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.scale = (dim // num_heads) ** -0.5

    def forward(self, x):
        P, N, C = x.shape
        qkv = self.qkv(x).reshape(P, N, 3, self.h, C // self.h)
        q, k, v = qkv.permute(2, 1, 3, 0, 4)          # each (N, h, P, C/h)
        scores = (q @ k.transpose(-2, -1)) * self.scale   # (N, h, P, P)  <-- O(n^2)
        attn = scores.softmax(dim=-1)                     # materialised n x n
        out = attn @ v                                    # (N, h, P, C/h)
        out = out.permute(2, 0, 1, 3).reshape(P, N, C)
        return self.proj(out)


# --------------------------------------------------------------------------- #
# Peak-memory measurement
#   CUDA : exact, from torch's caching-allocator peak counter.
#   CPU  : deterministic, reconstructed from the profiler's tensor-allocation
#          events (cumulative live bytes over time -> max). This avoids the
#          noise of RSS sampling, which the caching allocator hides.
# --------------------------------------------------------------------------- #
def peak_memory_mb(fn, device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        fn()
        torch.cuda.synchronize()
        return (torch.cuda.max_memory_allocated() - base) / 1e6

    with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
        fn()
    evs = [e for e in prof.events() if getattr(e, "cpu_memory_usage", 0)]
    evs.sort(key=lambda e: e.time_range.start)
    cur = peak = 0
    for e in evs:
        cur += e.cpu_memory_usage      # negative on free -> tracks live bytes
        peak = max(peak, cur)
    return peak / 1e6


def time_forward(fn, reps, warmup, device):
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)  # ms
    return statistics.median(ts)


def make_inputs(n, n_batch, embed_dim, device):
    """Dummy jet: features (P,N,C), 4-momentum v (N,4,P), all-real mask (N,P)."""
    torch.manual_seed(0)
    x = torch.randn(n, n_batch, embed_dim, device=device)
    # realistic-ish momenta so (eta, phi) are well spread; magnitude is irrelevant
    px, py = torch.randn(n_batch, n, device=device), torch.randn(n_batch, n, device=device)
    pz = torch.randn(n_batch, n, device=device)
    e = torch.sqrt(px ** 2 + py ** 2 + pz ** 2) + 1.0
    v = torch.stack([px, py, pz, e], dim=1)                       # (N,4,P)
    padding_mask = torch.zeros(n_batch, n, dtype=torch.bool, device=device)  # no padding
    pos = _compute_positions(v.float())                          # (N,P,2)
    return x, pos, padding_mask


def powerlaw_exponent(ns, ys):
    """Least-squares slope of log(y) vs log(n) == the scaling exponent."""
    lp = np.polyfit(np.log(np.asarray(ns)), np.log(np.asarray(ys)), 1)
    return lp[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[50, 100, 200, 400, 800])
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--embed-dim", type=int, default=128)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--ball-size", type=int, default=16)
    ap.add_argument("--mlp-ratio", type=int, default=4)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="erwin_complexity.png")
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(0)

    erwin = ParticleErwinBlock(
        args.embed_dim, args.heads, ball_size=args.ball_size, mlp_ratio=args.mlp_ratio
    ).to(device).eval()
    # Reference: full O(n^2) self-attention over the same (P,N,C) sequence.
    full_attn = FullAttention(args.embed_dim, args.heads).to(device).eval()

    rows = []
    print(f"device={device}  embed_dim={args.embed_dim}  heads={args.heads}  "
          f"ball_size={args.ball_size}  batch={args.batch}\n")
    header = f"{'n':>6} | {'erwin ms':>9} {'full ms':>9} | {'erwin MB':>9} {'full MB':>9}"
    print(header + "\n" + "-" * len(header))

    for n in args.sizes:
        x, pos, pmask = make_inputs(n, args.batch, args.embed_dim, device)

        def erwin_fn():
            with torch.no_grad():
                return erwin(x, pos, pmask)

        def full_fn():
            with torch.no_grad():
                return full_attn(x)

        e_ms = time_forward(erwin_fn, args.reps, args.warmup, device)
        f_ms = time_forward(full_fn, args.reps, args.warmup, device)
        e_mb = peak_memory_mb(erwin_fn, device)
        f_mb = peak_memory_mb(full_fn, device)

        rows.append((n, e_ms, f_ms, e_mb, f_mb))
        print(f"{n:>6} | {e_ms:>9.3f} {f_ms:>9.3f} | {e_mb:>9.2f} {f_mb:>9.2f}")

    ns = [r[0] for r in rows]
    e_ms = [r[1] for r in rows]
    f_ms = [r[2] for r in rows]
    e_mb = [r[3] for r in rows]
    f_mb = [r[4] for r in rows]

    print("\nfitted scaling exponent  y ~ n^p   (Erwin -> ~1 = O(n*B),  full -> ~2 = O(n^2)):")
    print(f"  time    Erwin p = {powerlaw_exponent(ns, e_ms):.2f} | full p = {powerlaw_exponent(ns, f_ms):.2f}")
    print(f"  memory  Erwin p = {powerlaw_exponent(ns, e_mb):.2f} | full p = {powerlaw_exponent(ns, f_mb):.2f}")
    # large-n regime is the cleanest (fixed overheads amortised): last two points
    if len(ns) >= 2:
        lt = powerlaw_exponent(ns[-2:], f_ms[-2:])
        lm = powerlaw_exponent(ns[-2:], f_mb[-2:])
        print(f"  full attention, largest-n slope only:  time p = {lt:.2f}, memory p = {lm:.2f}")

    # ---- plot -----------------------------------------------------------
    def guide(y0, exp):
        return [y0 * (n / ns[0]) ** exp for n in ns]

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for a, e, f, ylab, title in (
        (ax[0], e_ms, f_ms, "forward time (ms)", "Execution time"),
        (ax[1], e_mb, f_mb, "peak memory (MB)", "Peak memory"),
    ):
        a.plot(ns, e, "o-", color="tab:blue", label="Erwin / BallMSA")
        a.plot(ns, f, "s-", color="tab:red", label="full attention  O(n²)")
        a.plot(ns, guide(e[0], 1), "--", color="tab:blue", alpha=0.5, label="O(n) guide")
        a.plot(ns, guide(f[0], 2), "--", color="tab:red", alpha=0.5, label="O(n²) guide")
        a.set_xscale("log"); a.set_yscale("log")
        a.set_xlabel("sequence length  n (particles / jet)")
        a.set_ylabel(ylab); a.set_title(title)
        a.grid(True, which="both", ls=":", alpha=0.4)
        a.legend(fontsize=8)
    fig.suptitle(
        f"Erwin BallMSA vs full attention  "
        f"(C={args.embed_dim}, H={args.heads}, B={args.ball_size}, batch={args.batch}, {device.type})",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"\nsaved plot -> {args.out}")


if __name__ == "__main__":
    main()
