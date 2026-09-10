"""S3 + S4 - end-to-end invariances of ErwinParTv2.

A jet is a *set* of constituents stored in an arbitrarily ordered, arbitrarily
padded array. Two things must therefore be true of the logits, and neither is
guaranteed by construction once a tree, a pt-sort and a padded slot layout are
involved:

  S3  permutation invariance - reordering a jet's constituents changes nothing.
  S4  padding invariance     - masked slots change nothing, whatever they hold,
                               including duplicates of real particles (weaver's
                               `pad_mode: wrap` writes exactly that).

Both are silent failures if broken: the model still trains, still converges, and
is simply wrong about which particles interact.

Also checked here: the geometry survives SequenceTrimmer's *training-mode*
random permutation, which reorders x/v/mask and would decouple pf_points from
the features if it were not carried through the same gather.

Run from the weaver conda env:
    python tests/s3_s4_invariance.py
"""
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "networks"))
from erwin_part.model import ErwinParticleTransformerV2  # noqa: E402

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
torch.manual_seed(0)
failures = []


def check(name, ok, detail=""):
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}{' - ' + detail if detail else ''}")
    if not ok:
        failures.append(name)


# --------------------------------------------------------------- real batch --
import uproot  # noqa: E402
root = next((p for p in (
    os.path.join(WORKSPACE, "efficient_particle_transformer", "JetClass_example_100k.root"),
    os.path.join(WORKSPACE, "JetClass_example_100k.root")) if os.path.exists(p)), None)
assert root, "JetClass sample not found"

N = 12
ev = uproot.open(root)["tree"].arrays(
    ["part_px", "part_py", "part_pz", "part_energy", "part_deta", "part_dphi",
     "part_charge", "part_d0val", "part_dzval"], library="np", entry_stop=600)
# Deliberately include jets above the L=64 cut so the truncation check has
# something to truncate; the rest are ordinary.
big = [i for i in range(len(ev["part_px"])) if len(ev["part_px"][i]) > 70][:4]
small = [i for i in range(len(ev["part_px"])) if len(ev["part_px"][i]) <= 64][:N - len(big)]
pick = big + small
N = len(pick)
P = max(len(ev["part_px"][i]) for i in pick)

C = 7
x = torch.zeros(N, C, P)
v = torch.zeros(N, 4, P)
pts = torch.zeros(N, 2, P)
mask = torch.zeros(N, 1, P, dtype=torch.bool)
for n, i in enumerate(pick):
    k = len(ev["part_px"][i])
    p = np.stack([ev[b][i] for b in ("part_px", "part_py", "part_pz", "part_energy")])
    v[n, :, :k] = torch.from_numpy(p.astype(np.float32))
    pt = np.hypot(p[0], p[1])
    x[n, :, :k] = torch.from_numpy(np.stack([
        np.log(np.clip(pt, 1e-6, None)), np.log(np.clip(p[3], 1e-6, None)),
        ev["part_deta"][i], ev["part_dphi"][i], ev["part_charge"][i],
        np.tanh(ev["part_d0val"][i]), np.tanh(ev["part_dzval"][i])]).astype(np.float32))
    pts[n, :, :k] = torch.from_numpy(np.stack(
        [ev["part_deta"][i], ev["part_dphi"][i]]).astype(np.float32))
    mask[n, 0, :k] = True

mult = mask.sum(-1).squeeze(-1)
print(f"batch of {N} real jets, P={P}, multiplicity {int(mult.min())}-{int(mult.max())}\n")

model = ErwinParticleTransformerV2(input_dim=C, num_classes=10, seq_len=64).eval()
with torch.no_grad():
    base = model(x, v=v, mask=mask, points=pts)
check("baseline logits are finite", bool(torch.isfinite(base).all()),
      f"shape {tuple(base.shape)}")


# ------------------------------------------------- S3 permutation invariance --
print("\nS3. permutation invariance")
for trial, seed in enumerate((1, 2, 3)):
    g = torch.Generator().manual_seed(seed)
    xs, vs, ps, ms = x.clone(), v.clone(), pts.clone(), mask.clone()
    for n in range(N):
        k = int(mult[n])
        pi = torch.randperm(k, generator=g)
        xs[n, :, :k] = x[n, :, :k][:, pi]
        vs[n, :, :k] = v[n, :, :k][:, pi]
        ps[n, :, :k] = pts[n, :, :k][:, pi]
    with torch.no_grad():
        out = model(xs, v=vs, mask=ms, points=ps)
    d = float((out - base).abs().max())
    check(f"shuffled constituents give identical logits (seed {seed})", d < 1e-5,
          f"max |diff| = {d:.2e}")

# Padded slots interleaved among real ones, not just appended.
xs, vs, ps, ms = x.clone(), v.clone(), pts.clone(), mask.clone()
g = torch.Generator().manual_seed(7)
for n in range(N):
    pi = torch.randperm(P, generator=g)
    xs[n], vs[n], ps[n], ms[n] = x[n][:, pi], v[n][:, pi], pts[n][:, pi], mask[n][:, pi]
with torch.no_grad():
    out = model(xs, v=vs, mask=ms, points=ps)
check("real and padded slots interleaved arbitrarily", float((out - base).abs().max()) < 1e-5,
      f"max |diff| = {float((out - base).abs().max()):.2e}")


# ----------------------------------------------------- S4 padding invariance --
print("\nS4. padding invariance")
EXTRA = 24
pad_x = torch.cat([x, torch.randn(N, C, EXTRA) * 5], dim=2)
pad_v = torch.cat([v, torch.randn(N, 4, EXTRA).abs() * 100], dim=2)
pad_p = torch.cat([pts, torch.randn(N, 2, EXTRA) * 5], dim=2)
pad_m = torch.cat([mask, torch.zeros(N, 1, EXTRA, dtype=torch.bool)], dim=2)
with torch.no_grad():
    out = model(pad_x, v=pad_v, mask=pad_m, points=pad_p)
check("appended masked slots holding garbage change nothing",
      float((out - base).abs().max()) < 1e-5,
      f"max |diff| = {float((out - base).abs().max()):.2e}")

# weaver's pad_mode: wrap writes *duplicates of real particles* into padded slots.
wrap_x, wrap_v, wrap_p = x.clone(), v.clone(), pts.clone()
wrap_m = mask.clone()
for n in range(N):
    k = int(mult[n])
    if k < P:
        fill = P - k
        src = torch.arange(fill) % k
        wrap_x[n, :, k:] = x[n, :, src]
        wrap_v[n, :, k:] = v[n, :, src]
        wrap_p[n, :, k:] = pts[n, :, src]
with torch.no_grad():
    out = model(wrap_x, v=wrap_v, mask=wrap_m, points=wrap_p)
check("wrap-padded duplicates (weaver's pad_mode) change nothing",
      float((out - base).abs().max()) < 1e-5,
      f"max |diff| = {float((out - base).abs().max()):.2e}")

# A jet with zero constituents must not produce NaN.
empty_m = mask.clone()
empty_m[0] = False
with torch.no_grad():
    out = model(x, v=v, mask=empty_m, points=pts)
check("an all-padding jet yields finite logits", bool(torch.isfinite(out).all()))
check("emptying jet 0 leaves the other jets untouched",
      float((out[1:] - base[1:]).abs().max()) < 1e-5,
      f"max |diff| on jets 1.. = {float((out[1:] - base[1:]).abs().max()):.2e}")

# Truncation: constituents past the L-th hardest must not matter.
soft_x, soft_v, soft_p, soft_m = (t.clone() for t in (x, v, pts, mask))
added = 0
for n in range(N):
    k = int(mult[n])
    if k >= 64 and k < P:
        soft_v[n, :, k] = v[n, :, k - 1] * 1e-4        # a very soft extra particle
        soft_x[n, :, k] = x[n, :, k - 1]
        soft_p[n, :, k] = pts[n, :, k - 1]
        soft_m[n, 0, k] = True
        added += 1
if added:
    with torch.no_grad():
        out = model(soft_x, v=soft_v, mask=soft_m, points=soft_p)
    check(f"a constituent beyond the 64 hardest is truncated away ({added} jets)",
          float((out - base).abs().max()) < 1e-5,
          f"max |diff| = {float((out - base).abs().max()):.2e}")
else:
    print("       (no jet in this batch exceeds 64 constituents; truncation untested)")


# ------------------------------------------- geometry survives the trimmer ----
print("\nTrimmer alignment (training mode)")
model.train()
for m_ in model.modules():                     # isolate from dropout
    if isinstance(m_, torch.nn.Dropout):
        m_.p = 0.0
model.trimmer._counter = 99                    # skip its 5-call warm-up


def _seed(n):
    # SequenceTrimmer draws its truncation quantile from Python's `random` and its
    # permutation from torch's RNG. Seeding only torch leaves the truncation
    # length free to vary, so "same seed" would be reproducible only by luck.
    torch.manual_seed(n)
    random.seed(n)


_seed(3)
with torch.no_grad():
    a = model(x, v=v, mask=mask, points=pts)
_seed(3)
with torch.no_grad():
    b = model(x, v=v, mask=mask, points=pts)
check("training-mode forward is reproducible under a fixed seed",
      float((a - b).abs().max()) < 1e-6,
      f"max |diff| = {float((a - b).abs().max()):.2e}")

# If points did not ride through the trimmer's permutation, passing pf_points
# would disagree wildly with recomputing positions from the (permuted) p4.
_seed(5)
with torch.no_grad():
    with_pts = model(x, v=v, mask=mask, points=pts)
_seed(5)
with torch.no_grad():
    no_pts = model(x, v=v, mask=mask)
rel = float((with_pts - no_pts).abs().max() / with_pts.abs().max())
check("pf_points and the p4 fallback agree after a training-mode permutation",
      rel < 0.05, f"max relative difference = {rel:.2e} "
                  f"(misaligned points would be O(1))")

# Control: deliberately misalign points and confirm the check above would catch it.
_seed(5)
with torch.no_grad():
    scrambled = model(x, v=v, mask=mask, points=torch.roll(pts, 3, dims=2))
ctl = float((scrambled - no_pts).abs().max() / no_pts.abs().max())
check("misaligned points are detectable (control)", ctl > 0.05,
      f"max relative difference = {ctl:.2e}")

print()
if failures:
    print(f"S3/S4 FAILED: {len(failures)} check(s) - {', '.join(failures)}")
    sys.exit(1)
print("S3 + S4 PASSED")
