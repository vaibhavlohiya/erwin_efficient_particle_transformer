"""S8 - complexity sweep: ball attention vs ParT's dense pairwise attention.

ParT's cost is dominated not by the attention matrix but by `PairEmbed`: it runs
an MLP over every one of the L^2 constituent pairs, materialising an
(N, 64, L, L) intermediate. Ball attention evaluates that same MLP on only the
within-ball pairs, n_balls * m^2 = L * m of them, and attends block-diagonally.
The prediction is therefore ~L^1 against ParT's ~L^2, in both time and memory.

This measures the two attention stacks on identical inputs and fits a power law
to each. The dense baseline is weaver's real `PairEmbed` plus an explicit
(N, H, L, L) score matrix - not the fused SDPA kernel, which would hide exactly
the quadratic memory being measured.

A caveat this sweep makes visible: the architecture is held fixed at
ball_sizes=[8, 8, 16], strides=[2, 2] as L grows, so the number of balls grows
and the per-level cost stays linear. The production config at L=64 additionally
collapses the bottleneck to a *single* ball of 16 for global mixing; that ball is
quadratic in its own size, which is free at 16 nodes but would not be if L were
scaled up without adding coarsening levels.

Run from the weaver conda env:
    python tests/s8_complexity.py [--max-l 512] [--batch 4] [--plot]
"""
import argparse
import gc
import os
import statistics
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "networks"))
from erwin_part.ball_attention import BallBlock, assemble_bias, ball_distance_bias  # noqa: E402
from erwin_part.balltree_torch import build_ball_tree  # noqa: E402
from erwin_part.pair_features import PairFeatureEmbed  # noqa: E402
from weaver.nn.model.ParticleTransformer import PairEmbed  # noqa: E402

DIM, HEADS, BALL = 128, 8, 8
failures = []


def check(name, ok, detail=""):
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}{' - ' + detail if detail else ''}")
    if not ok:
        failures.append(name)


class DensePairAttention(nn.Module):
    """ParT's mechanism: dense pair bias + an explicitly materialised L x L score."""

    def __init__(self, dim=DIM, num_heads=HEADS):
        super().__init__()
        self.h = num_heads
        self.pair = PairEmbed(4, 0, [64, 64, num_heads], mode='concat',
                              use_pre_activation_pair=True, normalize_input=True)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.scale = (dim // num_heads) ** -0.5

    def forward(self, x, p4):
        N, L, C = x.shape
        u = self.pair(p4.permute(0, 2, 1))                    # (N, H, L, L)
        q, k, v = self.qkv(x).view(N, L, 3, self.h, C // self.h).permute(2, 0, 3, 1, 4)
        att = (q @ k.transpose(-1, -2)) * self.scale + u      # (N, H, L, L)
        att = att.softmax(-1)
        return self.proj((att @ v).transpose(1, 2).reshape(N, L, C))


class BallPairAttention(nn.Module):
    """The ErwinParTv2 stack: tree, within-ball pair MLP, block-diagonal attention."""

    def __init__(self, dim=DIM, num_heads=HEADS, ball=BALL):
        super().__init__()
        self.ball = ball
        self.pair = PairFeatureEmbed([64, 64, num_heads], level=0)
        self.block = BallBlock(dim, num_heads)

    def forward(self, x, p4, pos):
        N, L, C = x.shape
        valid = torch.ones(N, L, dtype=torch.bool, device=x.device)
        with torch.no_grad():
            perm = build_ball_tree(pos, valid, min_leaf=1)
        g = lambda t: torch.gather(t, 1, perm.unsqueeze(-1).expand(-1, -1, t.shape[-1]))
        nb = L // self.ball
        xb = g(x).view(N, nb, self.ball, C)
        pb = g(pos).view(N, nb, self.ball, 2)
        qb = g(p4).view(N, nb, self.ball, 4)
        vb = torch.ones(N, nb, self.ball, dtype=torch.bool, device=x.device)
        u = self.pair(qb)
        bias = assemble_bias(u, vb, ball_distance_bias(pb, vb), self.block.attn.sigma)
        return self.block(xb, pb, vb, bias).view(N, L, C)


def peak_memory_mb(fn):
    """Peak CPU bytes attributable to one call, from the profiler's allocation log."""
    gc.collect()
    with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
        fn()
    cur = peak = 0
    for e in prof.events():
        d = getattr(e, "cpu_memory_usage", 0) or 0
        cur += d
        peak = max(peak, cur)
    return peak / 1e6


def time_forward(fn, reps=5, warmup=2):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return statistics.median(ts)


def powerlaw(ns, ys):
    return float(np.polyfit(np.log(np.array(ns, float)), np.log(np.array(ys, float)), 1)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-l", type=int, default=512)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(0)
    torch.set_num_threads(4)
    lengths = [n for n in (64, 128, 256, 512, 1024) if n <= args.max_l]
    N = args.batch

    ball = BallPairAttention().eval()
    dense = DensePairAttention().eval()

    rows = []
    print(f"batch {N}, dim {DIM}, heads {HEADS}, ball {BALL}, threads 4\n")
    print(f"{'L':>6} {'ball ms':>9} {'dense ms':>9} {'speedup':>8} "
          f"{'ball MB':>9} {'dense MB':>9} {'saving':>8} {'pairs b/d':>11}")
    for L in lengths:
        x = torch.randn(N, L, DIM)
        pos = torch.randn(N, L, 2) * 0.3
        p4 = torch.randn(N, L, 4).abs() + 1.0
        p4[..., 3] = p4[..., :3].norm(dim=-1) + 0.5           # timelike

        fb = lambda: ball(x, p4, pos)
        fd = lambda: dense(x, p4)
        with torch.no_grad():
            tb, td = time_forward(fb), time_forward(fd)
            mb, md = peak_memory_mb(fb), peak_memory_mb(fd)
        pb, pd = (L // BALL) * BALL * BALL, L * L
        rows.append((L, tb, td, mb, md, pb, pd))
        print(f"{L:>6} {tb*1e3:>9.2f} {td*1e3:>9.2f} {td/tb:>7.1f}x "
              f"{mb:>9.1f} {md:>9.1f} {md/max(mb,1e-9):>7.1f}x {pd/pb:>10.0f}x")

    Ls = [r[0] for r in rows]
    e_tb, e_td = powerlaw(Ls, [r[1] for r in rows]), powerlaw(Ls, [r[2] for r in rows])
    e_mb, e_md = powerlaw(Ls, [r[3] for r in rows]), powerlaw(Ls, [r[4] for r in rows])

    print(f"\nfitted exponents over L in [{Ls[0]}, {Ls[-1]}]")
    print(f"  time   : ball L^{e_tb:.2f}   dense L^{e_td:.2f}")
    print(f"  memory : ball L^{e_mb:.2f}   dense L^{e_md:.2f}\n")

    check("dense pair attention scales ~quadratically in memory",
          e_md > 1.7, f"L^{e_md:.2f}")
    check("ball attention scales ~linearly in memory",
          e_mb < 1.35, f"L^{e_mb:.2f}")
    check("dense pair attention scales super-linearly in time",
          e_td > 1.6, f"L^{e_td:.2f}")
    check("ball attention scales ~linearly in time",
          e_tb < 1.35, f"L^{e_tb:.2f}")
    check("the gap widens with L (this is the whole point)",
          (rows[-1][2] / rows[-1][1]) > (rows[0][2] / rows[0][1]),
          f"speedup {rows[0][2]/rows[0][1]:.1f}x at L={Ls[0]} -> "
          f"{rows[-1][2]/rows[-1][1]:.1f}x at L={Ls[-1]}")
    check("pair count matches L*m vs L^2 exactly",
          all(r[5] == r[0] * BALL and r[6] == r[0] ** 2 for r in rows),
          f"at L={Ls[-1]}: {rows[-1][5]} vs {rows[-1][6]}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        BLUE, ORANGE, INK, GRID = "#2a78d6", "#eb6834", "#3f3f3f", "#e3e3e0"
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
        for ax, (bi, di, lab, unit) in zip(axes, [(1, 2, "forward time", "ms"),
                                                  (3, 4, "peak memory", "MB")]):
            yb = [r[bi] * (1e3 if unit == "ms" else 1) for r in rows]
            yd = [r[di] * (1e3 if unit == "ms" else 1) for r in rows]
            eb, ed = powerlaw(Ls, yb), powerlaw(Ls, yd)
            ax.plot(Ls, yd, "o-", color=ORANGE, lw=2, ms=6, label="ParT dense pair attention")
            ax.plot(Ls, yb, "o-", color=BLUE, lw=2, ms=6, label="ball attention (ErwinParTv2)")
            ax.annotate(f"$L^{{{ed:.2f}}}$", (Ls[-1], yd[-1]), textcoords="offset points",
                        xytext=(-10, 12), color=ORANGE, fontsize=11, ha="right")
            ax.annotate(f"$L^{{{eb:.2f}}}$", (Ls[-1], yb[-1]), textcoords="offset points",
                        xytext=(-10, -24), color=BLUE, fontsize=11, ha="right")
            ax.set_xscale("log", base=2); ax.set_yscale("log")
            ax.set_xlabel("constituents per jet  L"); ax.set_ylabel(f"{lab}  ({unit})")
            ax.set_title(lab, color=INK, fontsize=12, loc="left")
            ax.grid(True, which="both", color=GRID, lw=0.7)
            ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        axes[0].legend(frameon=False, fontsize=9, loc="upper left")
        fig.suptitle("Ball attention vs ParT's dense pairwise attention", fontsize=13,
                     x=0.02, ha="left", color=INK)
        fig.tight_layout()
        out = os.path.join(os.path.dirname(__file__), "..", "erwin_part_v2_complexity.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"wrote {os.path.normpath(out)}")

    print()
    if failures:
        print(f"S8 FAILED: {len(failures)} check(s) - {', '.join(failures)}")
        sys.exit(1)
    print("S8 PASSED")


if __name__ == "__main__":
    main()
