"""S5 - ball-local U against ParT's dense PairEmbed.

The claim under test is an indexing claim: evaluating ParT's pair MLP on only
the within-ball (i, j) pairs gives *exactly* the entries a dense PairEmbed would
have produced at those same indices - just without materialising the other
16,000-odd pairs per jet.

If the gather is off by a transpose, a ball offset, or a reshape order, the model
still trains and still produces plausible loss curves; it just silently biases
attention with the wrong particles' kinematics. Hence a bit-level check against
the real weaver module rather than a smoke test.

Run from the weaver conda env:
    python tests/s5_pair_parity.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "networks"))
from erwin_part.balltree_torch import ball_view, build_ball_tree  # noqa: E402
from erwin_part.geometry import (relative_positions, sort_truncate_pad,  # noqa: E402
                                 take_slots)
from erwin_part.pair_features import (PAIR_FEATURE_NORM, PairFeatureEmbed,  # noqa: E402
                                      within_ball_pair_features)
from weaver.nn.model.ParticleTransformer import PairEmbed, pairwise_lv_fts  # noqa: E402

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
L, BALL, HEADS = 64, 8, 8
failures = []


def check(name, ok, detail=""):
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}{' - ' + detail if detail else ''}")
    if not ok:
        failures.append(name)


# ------------------------------------------------------------ real jets ------
import uproot  # noqa: E402
root = next((p for p in (
    os.path.join(WORKSPACE, "efficient_particle_transformer", "JetClass_example_100k.root"),
    os.path.join(WORKSPACE, "JetClass_example_100k.root")) if os.path.exists(p)), None)
assert root, "JetClass sample not found"

NJ = 64
ev = uproot.open(root)["tree"].arrays(
    ["part_px", "part_py", "part_pz", "part_energy"], library="np", entry_stop=4000)
sel = [i for i in range(len(ev["part_px"])) if len(ev["part_px"][i]) >= L][:NJ]
NJ = len(sel)
p4 = torch.zeros(NJ, L, 4)
for n, i in enumerate(sel):
    px, py = ev["part_px"][i], ev["part_py"][i]
    keep = np.argsort(-(px ** 2 + py ** 2))[:L]
    for c, k in enumerate(("part_px", "part_py", "part_pz", "part_energy")):
        p4[n, :, c] = torch.from_numpy(ev[k][i][keep].astype(np.float32))
valid = torch.ones(NJ, L, dtype=torch.bool)
pos = relative_positions(p4, valid)
perm = build_ball_tree(pos, valid, min_leaf=1)
balls = ball_view(perm, BALL)                                   # (N, nb, m)
nb = balls.shape[1]
print(f"{NJ} real jets, L={L}, ball_size={BALL}, {nb} balls/jet\n")

# ------------------------------------------------- A. feature correctness ----
print("A. pair features")
p4_balls = torch.gather(p4, 1, balls.reshape(NJ, -1).unsqueeze(-1)
                        .expand(NJ, nb * BALL, 4)).view(NJ, nb, BALL, 4)
feats = within_ball_pair_features(p4_balls, 4).view(NJ, 4, nb, BALL, BALL)

# Independent reference: one explicit call per (ball, i, j) triple, no reshapes.
ref = torch.zeros(NJ, 4, nb, BALL, BALL)
with torch.no_grad():
    for b in range(nb):
        for i in range(BALL):
            for j in range(BALL):
                xi = p4_balls[:, b, i, :].unsqueeze(-1)          # (N, 4, 1)
                xj = p4_balls[:, b, j, :].unsqueeze(-1)
                ref[:, :, b, i, j] = pairwise_lv_fts(xi, xj, num_outputs=4).squeeze(-1)
# Absolute agreement is limited by float32: these are logs of clamped ratios with
# magnitudes up to ~20, so a few ulp of drift between a batched reshape and a
# per-pair call is expected. Assert the relative error instead, then prove the
# two paths are mathematically identical by re-running in float64.
rel = (feats - ref).abs() / ref.abs().clamp(min=1e-3)
check("vectorised features match an explicit per-pair reference (float32, relative)",
      float(rel.max()) < 1e-4,
      f"max relative {float(rel.max()):.2e}, max absolute "
      f"{float((feats - ref).abs().max()):.2e} on |values| up to "
      f"{float(ref.abs().max()):.1f}")

f64 = within_ball_pair_features(p4_balls.double(), 4).view(NJ, 4, nb, BALL, BALL)
ref64 = torch.zeros(NJ, 4, nb, BALL, BALL, dtype=torch.float64)
with torch.no_grad():
    for b in range(nb):
        for i in range(BALL):
            for j in range(BALL):
                ref64[:, :, b, i, j] = pairwise_lv_fts(
                    p4_balls[:, b, i, :].double().unsqueeze(-1),
                    p4_balls[:, b, j, :].double().unsqueeze(-1),
                    num_outputs=4).squeeze(-1)
# Residual float64 drift bottoms out around 1e-9 relative, not 1e-16: ln m^2 goes
# through m^2 = E^2 - |p|^2, a cancellation of large near-equal numbers for the
# nearly-collinear massless pairs that dominate a ball. 1e-8 is the meaningful
# floor here; what proves the point is the ~3e4x gap against float32.
rel64 = (f64 - ref64).abs() / ref64.abs().clamp(min=1e-3)
check("the two paths are identical in float64 (so the drift is precision, not logic)",
      float(rel64.max()) < 1e-8,
      f"max relative {float(rel64.max()):.2e}, absolute "
      f"{float((f64 - ref64).abs().max()):.2e} "
      f"(float32 relative was {float(rel.max()):.2e} - ~{float(rel.max()) / max(float(rel64.max()), 1e-16):.0e}x larger)")
check("features are symmetric in (i, j)",
      torch.allclose(feats, feats.transpose(-1, -2), atol=1e-5),
      f"max |U - U^T| = {float((feats - feats.transpose(-1, -2)).abs().max()):.2e}")

# --------------------------------------------- B. parity with dense PairEmbed --
print("\nB. parity with weaver's dense PairEmbed")
torch.manual_seed(0)
dense = PairEmbed(4, 0, [64, 64, HEADS], remove_self_pair=False,
                  use_pre_activation_pair=True, mode='concat',
                  normalize_input=True).eval()
local = PairFeatureEmbed.from_part_pair_embed(dense, level=0,
                                              input_norm="batch").eval()

sd_d, sd_l = dense.embed.state_dict(), local.embed.state_dict()
same = all(torch.equal(sd_d[k], sd_l[k]) for k in sd_d if k in sd_l)
check("weights transferred verbatim from PairEmbed",
      same and set(sd_d) == set(sd_l),
      f"{len(sd_d)} tensors, keys identical: {set(sd_d) == set(sd_l)}")

with torch.no_grad():
    u_dense = dense(p4.permute(0, 2, 1))                        # (N, H, L, L)
    u_ball = local(p4_balls)                                    # (N, nb, H, m, m)

check("dense U has ParT's shape", tuple(u_dense.shape) == (NJ, HEADS, L, L),
      str(tuple(u_dense.shape)))

# Gather the dense bias at exactly the pairs each ball contains.
bi = balls.unsqueeze(-1).expand(NJ, nb, BALL, BALL)              # row index
bj = balls.unsqueeze(-2).expand(NJ, nb, BALL, BALL)              # col index
n_ix = torch.arange(NJ).view(NJ, 1, 1, 1, 1)
h_ix = torch.arange(HEADS).view(1, 1, HEADS, 1, 1)
gathered = u_dense[n_ix, h_ix,
                   bi.unsqueeze(2).expand(NJ, nb, HEADS, BALL, BALL),
                   bj.unsqueeze(2).expand(NJ, nb, HEADS, BALL, BALL)]

diff = (u_ball - gathered).abs()
check("ball-local U == dense U at the same (i, j)",
      float(diff.max()) < 1e-4,
      f"max |diff| = {float(diff.max()):.2e}, mean {float(diff.mean()):.2e}")

# A deliberately wrong gather must fail this test, or the test proves nothing.
wrong = u_dense[n_ix, h_ix,
                bj.unsqueeze(2).expand(NJ, nb, HEADS, BALL, BALL),
                torch.roll(bi, 1, dims=1).unsqueeze(2).expand(NJ, nb, HEADS, BALL, BALL)]
check("a mis-indexed gather is detectably different (control)",
      float((u_ball - wrong).abs().max()) > 1e-3,
      f"max |diff| = {float((u_ball - wrong).abs().max()):.2e}")

# ------------------------------------------------------------- C. the win ----
print("\nC. cost")
dense_pairs = L * L
ball_pairs = nb * BALL * BALL
check("ball-local evaluates far fewer pairs",
      ball_pairs * 8 <= dense_pairs,
      f"{ball_pairs} vs {dense_pairs} per jet ({dense_pairs / ball_pairs:.0f}x fewer); "
      f"vs ParT's L=128: {128 * 128 / ball_pairs:.0f}x")

mem_dense = NJ * 64 * L * L * 4 / 1e6
mem_ball = NJ * 64 * ball_pairs * 4 / 1e6
print(f"       pair-MLP activation at 64 ch, fp32, N={NJ}: "
      f"dense {mem_dense:.1f} MB -> ball-local {mem_ball:.1f} MB")

# ------------------------------------------------- D. the fixed standardiser --
print("\nD. fixed per-level standardisation")
fixed = PairFeatureEmbed([64, 64, HEADS], level=0, input_norm="fixed")
fixed.train()
check("input standardiser stays in eval mode while training",
      not fixed.embed[0].training)
check("standardiser is frozen", not fixed.embed[0].weight.requires_grad)
mu, sd = PAIR_FEATURE_NORM[0]
check("standardiser carries the measured level-0 constants",
      torch.allclose(fixed.embed[0].running_mean, torch.tensor(mu), atol=1e-4)
      and torch.allclose(fixed.embed[0].running_var, torch.tensor(sd) ** 2, atol=1e-3),
      f"mean {[round(v, 2) for v in fixed.embed[0].running_mean.tolist()]}")

z = fixed.embed[0](feats.permute(0, 1, 2, 3, 4).reshape(NJ, 4, -1))
check("standardised features are ~zero-mean, unit-scale",
      float(z.mean().abs()) < 0.35 and 0.6 < float(z.std()) < 1.6,
      f"mean {float(z.mean()):+.3f}, std {float(z.std()):.3f}")

lv = [PairFeatureEmbed([64, 64, HEADS], level=i).embed[0].running_mean[3].item()
      for i in range(3)]
check("per-level constants differ (ln m^2 climbs with coarsening)",
      lv[2] - lv[0] > 5.0,
      f"ln m^2 mean: L0 {lv[0]:+.2f} -> L1 {lv[1]:+.2f} -> L2 {lv[2]:+.2f}")

print()
if failures:
    print(f"S5 FAILED: {len(failures)} check(s) - {', '.join(failures)}")
    sys.exit(1)
print("S5 PASSED")
