"""Forward-pass FLOPs: ErwinParTv2 vs ParT.

Counted with torch's dispatch-level FlopCounterMode rather than derived by hand,
so the numbers reflect the operations actually issued (matmul, conv, SDPA)
instead of an idealised formula. Hardware-independent, unlike the wall-clock
comparisons, which on this machine were unusable on MPS.

Two views, because they answer different questions:

  matched-L    both models see the same L constituents. This isolates the
               *architecture*: how each one scales.
  production   ErwinParTv2 truncates to its fixed 64 slots while ParT consumes
               all L. This is the deployed cost, and part of the gap is the
               truncation rather than the attention mechanism - stated plainly
               rather than folded into the headline.

    python tests/flops_compare.py [--plot]
"""
import argparse
import json
import os
import sys
import warnings

import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "networks"))
from torch.utils.flop_counter import FlopCounterMode  # noqa: E402

from erwin_part.model import ErwinParticleTransformerV2  # noqa: E402
from weaver.nn.model.ParticleTransformer import ParticleTransformer  # noqa: E402

# 32/64/128 span the real jet range (median 39, p95 69, max 136); 256 and 512
# are synthetic dense clouds far beyond any JetClass jet, kept only to fit an
# asymptotic exponent. L must be a power of two: the tree is a balanced split.
C_IN, LENGTHS = 17, [32, 64, 128, 256, 512]
PHYSICAL = [32, 64, 128]
OUT = os.path.join(os.path.dirname(__file__), "..", "plots")


def inputs(P, n=1):
    x = torch.randn(n, C_IN, P, requires_grad=True)
    v = torch.randn(n, 4, P).abs() + 1.0
    v[:, 3] = v[:, :3].norm(dim=1) + 0.5
    mask = torch.ones(n, 1, P, dtype=torch.bool)
    pts = torch.randn(n, 2, P) * 0.3
    return x, v, mask, pts


def erwin(seq_len, **kw):
    return ErwinParticleTransformerV2(
        input_dim=C_IN, num_classes=10, seq_len=seq_len,
        # level-2 ball = seq_len//4 keeps the bottleneck a single ball at every L
        ball_sizes=[8, 8, max(seq_len // 4, 8)], strides=[2, 2], depths=[2, 2, 2],
        pair_embed_dims=[64, 64], **kw).eval()


def part():
    return ParticleTransformer(
        input_dim=C_IN, num_classes=10, pair_input_dim=4,
        pair_embed_dims=[64, 64, 64], num_layers=8, num_cls_layers=2,
        block_params={"dropout": 0, "attn_dropout": 0, "activation_dropout": 0},
        cls_block_params={"dropout": 0, "attn_dropout": 0, "activation_dropout": 0},
        fc_params=[]).eval()


def count(model, args, breakdown=False):
    fc = FlopCounterMode(display=False)
    with fc:
        model(*args)
    total = fc.get_total_flops()
    if not breakdown:
        return total, None
    return total, {k: sum(v.values()) for k, v in fc.get_flop_counts().items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    res = {"matched": [], "production": [], "breakdown": {}}
    print(f"{'L':>6}{'ParT G':>11}{'Erwin matched G':>18}{'Erwin prod G':>15}"
          f"{'matched x':>11}{'prod x':>9}")
    for L in LENGTHS:
        x, v, mask, pts = inputs(L)
        f_part, _ = count(part(), (x, v, mask))
        f_match, _ = count(erwin(L), (x, v, mask, pts))
        f_prod, _ = count(erwin(64), (x, v, mask, pts))
        res["matched"].append({"L": L, "part": f_part, "erwin": f_match})
        res["production"].append({"L": L, "part": f_part, "erwin": f_prod})
        print(f"{L:>6}{f_part/1e9:>11.3f}{f_match/1e9:>18.3f}{f_prod/1e9:>15.3f}"
              f"{f_part/f_match:>10.1f}x{f_part/f_prod:>8.1f}x")

    # component breakdown at the production point
    x, v, mask, pts = inputs(128)
    _, bd_p = count(part(), (x, v, mask), breakdown=True)
    _, bd_e = count(erwin(64), (x, v, mask, pts), breakdown=True)

    def group(bd, root, prefix, depth):
        """Sum the FLOPs of every child module under `prefix` at exactly `depth` dots.

        Blocks live in a ModuleList, so they are depth-2 keys ('...blocks.0');
        summing depth-1 only would silently report zero for them.
        """
        return sum(v for k, v in bd.items()
                   if k.startswith(prefix) and k.count(".") == depth)

    res["breakdown"] = {
        "ParT": {
            "pair embed (dense U)": group(bd_p, None, "ParticleTransformer.pair_embed", 1),
            "particle blocks": group(bd_p, None, "ParticleTransformer.blocks.", 2),
            "class blocks": group(bd_p, None, "ParticleTransformer.cls_blocks.", 2),
            "input embed": group(bd_p, None, "ParticleTransformer.embed", 1),
            "total": bd_p["ParticleTransformer"],
        },
        "ErwinParTv2": {
            "pair embed (ball U)": group(bd_e, None, "ErwinParticleTransformerV2.pair_embeds.", 2),
            "ball blocks": group(bd_e, None, "ErwinParticleTransformerV2.levels.", 3),
            "class blocks": group(bd_e, None, "ErwinParticleTransformerV2.cls_blocks.", 2),
            "input embed": group(bd_e, None, "ErwinParticleTransformerV2.embed", 1),
            "pooling": group(bd_e, None, "ErwinParticleTransformerV2.pools.", 2),
            "total": bd_e["ErwinParticleTransformerV2"],
        },
    }
    print("\ncomponent breakdown at L=128 (ParT) / 64 slots (ErwinParTv2), batch 1")
    for name, d in res["breakdown"].items():
        tot = d["total"]
        print(f"  {name}")
        for k, val in d.items():
            pct = "" if k == "total" else f"   {100 * val / tot:5.1f}%"
            print(f"    {k:<22}{val/1e9:8.4f} G{pct}")

    import math
    print("\nfitted FLOP scaling")
    for key, lbl in (("matched", "matched-L"), ("production", "production")):
        for rng, rlbl in ((PHYSICAL, "physical 32-128"), (LENGTHS, "full 32-512")):
            rows = [r for r in res[key] if r["L"] in rng]
            ex = lambda fld: math.log(rows[-1][fld] / rows[0][fld]) / math.log(
                rows[-1]["L"] / rows[0]["L"])
            print(f"  {lbl:<11} {rlbl:<16} ParT L^{ex('part'):.2f}   "
                  f"ErwinParTv2 L^{ex('erwin'):.2f}")
    res["physical"] = PHYSICAL

    # what dropping the (ablation-null) pair bias saves
    f_full, _ = count(erwin(64), (x, v, mask, pts))
    f_noU, _ = count(erwin(64, use_pair_bias=False), (x, v, mask, pts))
    res["no_pair_bias"] = {"full": f_full, "no_u": f_noU}
    print(f"\ndropping the U bias (S11: p=0.823, no measurable accuracy cost):")
    print(f"  {f_full/1e9:.4f} G -> {f_noU/1e9:.4f} G   ({100*(1-f_noU/f_full):.1f}% fewer FLOPs)")

    with open(os.path.join(os.path.dirname(__file__), "s10_results", "flops.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print("\nwrote tests/s10_results/flops.json")


if __name__ == "__main__":
    main()
