"""Measure PAIR_FEATURE_NORM - the fixed input standardisation of the pair MLP.

Builds the slots, tree and pooled 4-momenta exactly as the model does for the
given geometry, then prints mean and std of [ln kT, ln z, ln deltaR, ln m^2]
over the within-ball pairs of every level, keyed by (pool factor, ball size) -
the two things the statistics depend on - ready to paste into
networks/erwin_part/pair_features.py.

    python tests/calibrate_pair_norm.py                                  # default geometry
    python tests/calibrate_pair_norm.py --ball-sizes 8,16,32,64 --strides 1,1,1

Torch + weaver (for pairwise_lv_fts); run in the weaver env.
"""
import argparse
import os
import sys

import numpy as np
import torch
import uproot

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "networks"))
from erwin_part.balltree_torch import build_levels  # noqa: E402
from erwin_part.geometry import sort_truncate_pad, take_slots  # noqa: E402
from erwin_part.pair_features import within_ball_pair_features  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--jets", type=int, default=4000)
ap.add_argument("--seq-len", type=int, default=64)
ap.add_argument("--ball-sizes", default="8,8,16")
ap.add_argument("--strides", default="2,2")
ap.add_argument("--pairs", default="valid", choices=["all", "valid", "offdiag"],
                help="valid (default): both members real, i==j included - this "
                     "reproduces the original PAIR_FEATURE_NORM to every digit; "
                     "all: padding pairs too; offdiag: valid and i!=j")
args = ap.parse_args()
ball_sizes = [int(b) for b in args.ball_sizes.split(",")]
strides = [int(s) for s in args.strides.split(",")] if args.strides else []

WORKSPACE = os.path.abspath(os.path.join(HERE, "..", ".."))
root = next(p for p in (
    os.path.join(WORKSPACE, "efficient_particle_transformer", "JetClass_example_100k.root"),
    os.path.join(WORKSPACE, "JetClass_example_100k.root")) if os.path.exists(p))
br = ["part_px", "part_py", "part_pz", "part_energy", "part_deta", "part_dphi"]
ev = uproot.open(root)["tree"].arrays(br, library="np", entry_stop=args.jets)

N = args.jets
P = max(len(a) for a in ev["part_px"])
p4 = torch.zeros(N, P, 4); pos = torch.zeros(N, P, 2)
valid = torch.zeros(N, P, dtype=torch.bool)
for n in range(N):
    k = len(ev["part_px"][n])
    p4[n, :k] = torch.from_numpy(np.stack([ev[b][n] for b in br[:4]], -1).astype(np.float32))
    pos[n, :k] = torch.from_numpy(np.stack([ev[b][n] for b in br[4:]], -1).astype(np.float32))
    valid[n, :k] = True

# The model's own pipeline: pt-sorted slots, tree, pooling by 4-momentum sum.
pt = torch.hypot(p4[..., 0], p4[..., 1])
idx, vL = sort_truncate_pad(valid, pt, args.seq_len)
posL = take_slots(pos, idx) * vL.unsqueeze(-1)
p4L = take_slots(p4, idx) * vL.unsqueeze(-1)
levels = build_levels(posL, vL, ball_sizes, strides, rotate=0.0)

print(f"# {N} jets, L={args.seq_len}, ball_sizes={ball_sizes}, strides={strides}, "
      f"pairs={args.pairs}")
cur_p4, cur_v, pool = p4L, vL, 1
for li, lv in enumerate(levels):
    perm, m = lv["perm"], ball_sizes[li]
    cur_p4 = torch.gather(cur_p4, 1, perm.unsqueeze(-1).expand(-1, -1, 4))
    cur_v = torch.gather(cur_v, 1, perm)
    L = cur_p4.shape[1]
    f = within_ball_pair_features(cur_p4.view(N, L // m, m, 4))       # (N, 4, nb*m*m)
    vb = cur_v.view(N, L // m, m)
    pv = (vb.unsqueeze(-1) & vb.unsqueeze(-2))
    if args.pairs == "offdiag":
        pv = pv & ~torch.eye(m, dtype=torch.bool)
    keep = pv.reshape(N, -1) if args.pairs != "all" else torch.ones_like(pv.reshape(N, -1))
    x = f.permute(0, 2, 1)[keep]                                      # (pairs, 4)
    mean = tuple(round(v, 3) for v in x.mean(0).tolist())
    std = tuple(round(v, 3) for v in x.std(0).tolist())
    print(f"    ({pool}, {m}): ({mean}, {std}),")
    if li < len(strides):
        s = strides[li]
        cur_p4 = cur_p4.view(N, L // s, s, 4).sum(2)
        cur_v = cur_v.view(N, L // s, s).any(2)
        pool *= s
