"""S13 - growing-ball layouts (strides of 1).

`ball_sizes=[8, 16, 32, 64], strides=[1, 1, 1]` keeps all 64 constituents and
widens the ball layer by layer. That is only sound if a ball of 2m really is two
balls of m - i.e. if every level is a node of the SAME median-split tree - and
if the pair MLP is standardised for the ball size it actually sees.

  1. tree    - stride 1 yields the identity permutation, which is also what a
               rebuilt tree gives; every ball of size m is exactly a node of
               the tree cut at m; the default geometry is untouched
  2. norm    - the geometry lookup reproduces the level table for the default
               layout, standardises every grow level to ~N(0, 1) on held-out
               jets, and warns on a geometry nobody measured
  3. model   - both arms run forward and backward, every parameter trains, no
               pooling layer exists, and Lund tokens still fit the top ball
  4. isolation - a jet scored alone matches the batch; particle order is
               irrelevant (eval mode)

Run from the weaver conda env:
    python tests/s13_grow.py
"""
import os
import sys
import warnings

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "networks"))
from erwin_part.balltree_torch import build_ball_tree, build_levels  # noqa: E402
from erwin_part.geometry import sort_truncate_pad, take_slots  # noqa: E402
from erwin_part.model import ErwinParticleTransformerV2  # noqa: E402
from erwin_part.pair_features import (PAIR_FEATURE_NORM,  # noqa: E402
                                      PAIR_FEATURE_NORM_BY_GEOMETRY,
                                      PairFeatureEmbed, within_ball_pair_features)

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
torch.manual_seed(0)
failures = []
ARMS = {"grow8": ([8, 16, 32, 64], [1, 1, 1], [2, 2, 1, 1]),
        "grow2": ([2, 4, 8, 16, 32, 64], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1])}


def check(name, ok, detail=""):
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}{' - ' + detail if detail else ''}")
    if not ok:
        failures.append(name)


import uproot  # noqa: E402
root = next((p for p in (
    os.path.join(WORKSPACE, "efficient_particle_transformer", "JetClass_example_100k.root"),
    os.path.join(WORKSPACE, "JetClass_example_100k.root")) if os.path.exists(p)), None)
assert root, "JetClass sample not found"

# Jets 5000-5600: disjoint from the 4000 the pair constants were measured on.
J0, NJ = 5000, 600
br = ["part_px", "part_py", "part_pz", "part_energy", "part_deta", "part_dphi",
      "part_charge", "part_d0val", "part_dzval"]
ev = uproot.open(root)["tree"].arrays(br, library="np", entry_start=J0, entry_stop=J0 + NJ)
P = max(len(a) for a in ev["part_px"])
v = torch.zeros(NJ, 4, P); pts = torch.zeros(NJ, 2, P)
x = torch.zeros(NJ, 7, P); mask = torch.zeros(NJ, 1, P, dtype=torch.bool)
for n in range(NJ):
    k = len(ev["part_px"][n])
    p = np.stack([ev[b][n] for b in br[:4]])
    v[n, :, :k] = torch.from_numpy(p.astype(np.float32))
    pts[n, :, :k] = torch.from_numpy(np.stack([ev["part_deta"][n], ev["part_dphi"][n]]).astype(np.float32))
    x[n, :, :k] = torch.from_numpy(np.stack([
        np.log(np.clip(np.hypot(p[0], p[1]), 1e-6, None)), np.log(np.clip(p[3], 1e-6, None)),
        ev["part_deta"][n], ev["part_dphi"][n], ev["part_charge"][n],
        np.tanh(ev["part_d0val"][n]), np.tanh(ev["part_dzval"][n])]).astype(np.float32))
    mask[n, 0, :k] = True

valid = mask.squeeze(1)
p4 = v.permute(0, 2, 1); pos = pts.permute(0, 2, 1)
idx, vL = sort_truncate_pad(valid, torch.hypot(p4[..., 0], p4[..., 1]), 64)
posL = take_slots(pos, idx) * vL.unsqueeze(-1)
p4L = take_slots(p4, idx) * vL.unsqueeze(-1)

# --------------------------------------------------------- 1. tree -----------
print("1. tree")
bs, st, _ = ARMS["grow2"]
levels = build_levels(posL, vL, bs, st, rotate=45.0)
ident = torch.arange(64).expand(NJ, 64)
check("stride 1 gives the identity permutation at every later level",
      all(torch.equal(lv["perm"], ident) for lv in levels[1:]))
ordered = torch.gather(posL, 1, levels[0]["perm"].unsqueeze(-1).expand(-1, -1, 2))
vo = torch.gather(vL, 1, levels[0]["perm"])
check("...which is exactly what rebuilding the tree on that order gives",
      torch.equal(build_ball_tree(ordered, vo), ident))

perm0 = levels[0]["perm"]
same = True
for m in (2, 4, 8, 16, 32, 64):
    cut = build_ball_tree(posL, vL, min_leaf=m).view(NJ, 64 // m, m).sort(-1).values
    ours = perm0.view(NJ, 64 // m, m).sort(-1).values
    same &= torch.equal(cut, ours)
check("every ball of size m is a node of the tree cut at m (m = 2..64)", same)

default = build_levels(posL, vL, [8, 8, 16], [2, 2], rotate=45.0)
check("the default [8,8,16]/[2,2] geometry still rebuilds a tree per level",
      not any(torch.equal(lv["perm"], ident[:, :lv["perm"].shape[1]]) for lv in default))

# --------------------------------------------------------- 2. norm -----------
print("\n2. pair-feature standardisation")
geo = [(1, 8), (2, 8), (4, 16)]
same = all(PAIR_FEATURE_NORM_BY_GEOMETRY[g] == PAIR_FEATURE_NORM[i] for i, g in enumerate(geo))
m_def = ErwinParticleTransformerV2(input_dim=7, num_classes=10)
bufs = all(torch.equal(m_def.pair_embeds[i].embed[0].running_mean,
                       torch.tensor(PAIR_FEATURE_NORM[i][0])) for i in range(3))
check("default model gets exactly the level-indexed constants it always had", same and bufs)

worst = 0.0
for m in (2, 4, 8, 16, 32, 64):
    pv = torch.gather(p4L, 1, perm0.unsqueeze(-1).expand(-1, -1, 4)).view(NJ, 64 // m, m, 4)
    vb = torch.gather(vL, 1, perm0).view(NJ, 64 // m, m)
    keep = (vb.unsqueeze(-1) & vb.unsqueeze(-2)).reshape(NJ, -1)
    f = within_ball_pair_features(pv).permute(0, 2, 1)[keep]
    mean, std = (torch.tensor(t) for t in PAIR_FEATURE_NORM_BY_GEOMETRY[(1, m)])
    z = (f - mean) / std
    worst = max(worst, float(z.mean(0).abs().max()), float((z.std(0) - 1).abs().max()))
check("every grow level standardises to ~N(0,1) on 600 held-out jets",
      worst < 0.15, f"worst |mean| or |std-1| = {worst:.3f}")

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    PairFeatureEmbed([64, 64, 8], level=0, geometry=(8, 8))
check("an unmeasured geometry warns and names the calibration script",
      any("calibrate_pair_norm" in str(x.message) for x in w))

# --------------------------------------------------------- 3. model ----------
print("\n3. model")
B = 24
xb, vb_, mb, pb = x[:B], v[:B], mask[:B], pts[:B]
n_def = sum(p.numel() for p in m_def.parameters())
models = {}
for arm, (bs, st, dp) in ARMS.items():
    with warnings.catch_warnings():
        warnings.simplefilter("error")       # a missing constant would warn
        m = ErwinParticleTransformerV2(input_dim=7, num_classes=10, ball_sizes=bs,
                                       strides=st, depths=dp, readout="coarse")
    models[arm] = m
    n = sum(p.numel() for p in m.parameters())
    check(f"{arm}: 6 blocks, no pooling layer, every pair constant measured",
          sum(len(l) for l in m.levels) == 6 and
          not any(type(p).__name__ == "BallPooling" for p in m.pools),
          f"{n:,} params vs {n_def:,} default ({(n - n_def) / n_def:+.1%})")
    m.train()
    out = m(xb, v=vb_, mask=mb, points=pb)
    out.sum().backward()
    # level_embed rows of levels the readout skips are legitimately unused.
    dead = [k for k, p in m.named_parameters()
            if p.requires_grad and k != "level_embed" and (p.grad is None or not p.grad.abs().sum())]
    check(f"{arm}: training forward finite, every parameter gets gradient",
          bool(torch.isfinite(out).all()) and not dead, f"dead: {dead[:4]}" if dead else "")

ml = ErwinParticleTransformerV2(input_dim=7, num_classes=10, ball_sizes=[8, 16, 32, 64],
                                strides=[1, 1, 1], depths=[2, 2, 1, 1], readout="coarse",
                                use_lund=True).eval()
with torch.no_grad():
    out = ml(xb, v=vb_, mask=mb, points=pb)
check("Lund tokens join the 64-wide top ball of grow8", bool(torch.isfinite(out).all()))

# --------------------------------------------------------- 4. isolation ------
print("\n4. isolation (eval mode)")
# Shuffle each jet's real particles among themselves, keeping padding last:
# weaver's SequenceTrimmer cuts every input to the longest real multiplicity
# and assumes real particles come first (as the data loader guarantees), so a
# shuffle that mixed padding in would drop real particles, in ParT too.
sh = torch.arange(P).expand(B, P).clone()
for i in range(B):
    k = int(mb[i].sum())
    sh[i, :k] = torch.randperm(k)
g = lambda t: torch.gather(t, 2, sh.unsqueeze(1).expand(-1, t.shape[1], -1))
for arm, m in models.items():
    m.eval()
    with torch.no_grad():
        batch = m(xb, v=vb_, mask=mb, points=pb)
        alone = torch.cat([m(xb[i:i + 1], v=vb_[i:i + 1], mask=mb[i:i + 1],
                             points=pb[i:i + 1]) for i in range(B)])
        perm_out = m(g(xb), v=g(vb_), mask=g(mb), points=g(pb))
    check(f"{arm}: each jet alone matches the batch",
          float((alone - batch).abs().max()) < 1e-5, f"max |diff| {float((alone - batch).abs().max()):.1e}")
    check(f"{arm}: particle order does not change the logits",
          float((perm_out - batch).abs().max()) < 1e-4, f"max |diff| {float((perm_out - batch).abs().max()):.1e}")

print()
print("S13 FAILED: " + ", ".join(failures) if failures else "S13 passed")
sys.exit(1 if failures else 0)
