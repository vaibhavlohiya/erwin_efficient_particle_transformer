"""S1 - geometry checks for ErwinParTv2.

Three things must hold before a ball tree is worth building on top of these
coordinates:

  A. phi wrapping. A jet straddling +/-pi must come out contiguous in
     delta_phi. This is where the current ErwinParT path is wrong: it partitions
     on absolute phi = atan2(py, px).
  B. Ground truth. Positions recomputed from part_px/py/pz must reproduce the
     part_deta / part_dphi branches JetClass ships, to float32 precision.
  C. Slot layout. pt-descending sort, truncation to L, and the validity mask
     must be exact and order-independent.

Run from the weaver conda env:
    python tests/s1_geometry.py
"""
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "networks"))
from erwin_part.geometry import (  # noqa: E402
    delta_r, eta_phi_pt, jet_axis, masked_centroid, relative_positions,
    sort_truncate_pad, take_slots, wrap_phi,
)

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
failures = []


def check(name, ok, detail=""):
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}{' - ' + detail if detail else ''}")
    if not ok:
        failures.append(name)


def p4_from_angles(pt, eta, phi):
    """Build [px, py, pz, E] for massless constituents."""
    px, py = pt * np.cos(phi), pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    e = np.sqrt(px ** 2 + py ** 2 + pz ** 2)
    return torch.tensor(np.stack([px, py, pz, e], axis=-1), dtype=torch.float32)


# --------------------------------------------------------------- A. wrapping --
print("A. azimuthal wraparound")

# A jet whose axis sits on the +/-pi seam: constituents at phi = pi - 0.1 and
# phi = -pi + 0.1 are 0.2 apart in reality, but 2*pi - 0.2 apart in raw atan2.
rng = np.random.default_rng(0)
n = 24
phi_true = math.pi + rng.uniform(-0.25, 0.25, n)          # straddles the seam
phi_true = (phi_true + math.pi) % (2 * math.pi) - math.pi  # what atan2 returns
eta_true = rng.uniform(-0.25, 0.25, n)
pt_true = rng.uniform(1.0, 60.0, n)

p4 = p4_from_angles(pt_true, eta_true, phi_true).unsqueeze(0)   # (1, n, 4)
valid = torch.ones(1, n, dtype=torch.bool)

pos = relative_positions(p4, valid)
dphi = pos[0, :, 1]
check("delta_phi is contiguous across the seam",
      bool(dphi.abs().max() < 0.6),
      f"max |delta_phi| = {dphi.abs().max():.4f} rad (spread was +/-0.25)")

_, phi_raw, _ = eta_phi_pt(p4)
raw_spread = float(phi_raw.max() - phi_raw.min())
check("raw atan2 phi would have been torn apart",
      raw_spread > 5.0,
      f"raw phi spread = {raw_spread:.3f} rad vs wrapped {float(dphi.max()-dphi.min()):.3f}")

# The bug in the current path, reproduced exactly: unweighted mean of absolute phi.
buggy_axis = float(phi_raw.mean())
true_axis = float(jet_axis(p4, valid)[1])
check("summed-p4 axis beats the unweighted absolute-phi mean",
      abs(wrap_phi(torch.tensor(true_axis - math.pi))) < 0.3
      and abs(wrap_phi(torch.tensor(buggy_axis - math.pi))) > 1.0,
      f"true axis {true_axis:+.3f} rad, buggy mean {buggy_axis:+.3f} rad, "
      f"jet really at {math.pi:+.3f}/-pi")

check("wrap_phi is idempotent on already-wrapped input",
      torch.allclose(wrap_phi(dphi), dphi, atol=1e-6))

# delta_r must re-wrap: two points either side of the seam in jet-frame phi
a = torch.tensor([[0.0, 3.10]])
b = torch.tensor([[0.0, -3.10]])
check("delta_r re-wraps the phi difference",
      float(delta_r(a, b)) < 0.1,
      f"delta_r = {float(delta_r(a, b)):.4f} (naive would be 6.20)")


# ------------------------------------------------------------ B. ground truth --
print("\nB. agreement with the JetClass part_deta / part_dphi branches")
import uproot  # noqa: E402

root = next((p for p in (
    os.path.join(WORKSPACE, "efficient_particle_transformer", "JetClass_example_100k.root"),
    os.path.join(WORKSPACE, "JetClass_example_100k.root")) if os.path.exists(p)), None)

if root is None:
    check("JetClass sample found", False, "cannot run ground-truth check")
else:
    NJ = 2000
    t = uproot.open(root)["tree"]
    ev = t.arrays(["part_px", "part_py", "part_pz", "part_energy",
                   "part_deta", "part_dphi", "jet_eta", "jet_phi"],
                  library="np", entry_stop=NJ)

    P = max(len(a) for a in ev["part_px"])
    p4 = torch.zeros(NJ, P, 4)
    valid = torch.zeros(NJ, P, dtype=torch.bool)
    ref = torch.zeros(NJ, P, 2)
    for i in range(NJ):
        k = len(ev["part_px"][i])
        p4[i, :k, 0] = torch.from_numpy(ev["part_px"][i].astype(np.float32))
        p4[i, :k, 1] = torch.from_numpy(ev["part_py"][i].astype(np.float32))
        p4[i, :k, 2] = torch.from_numpy(ev["part_pz"][i].astype(np.float32))
        p4[i, :k, 3] = torch.from_numpy(ev["part_energy"][i].astype(np.float32))
        ref[i, :k, 0] = torch.from_numpy(ev["part_deta"][i].astype(np.float32))
        ref[i, :k, 1] = torch.from_numpy(ev["part_dphi"][i].astype(np.float32))
        valid[i, :k] = True

    # (i) against the stored jet_eta / jet_phi -- isolates the wrapping itself
    axis_stored = (torch.from_numpy(ev["jet_eta"].astype(np.float32)),
                   torch.from_numpy(ev["jet_phi"].astype(np.float32)))
    pos_stored = relative_positions(p4, valid, axis=axis_stored)
    d_eta = (pos_stored[..., 0] - ref[..., 0])[valid].abs()
    d_phi = (pos_stored[..., 1] - ref[..., 1])[valid].abs()
    check("delta_phi vs part_dphi (stored jet axis)",
          float(d_phi.max()) < 1e-4, f"max |diff| = {float(d_phi.max()):.2e}")
    check("delta_eta vs part_deta (stored jet axis)",
          float(d_eta.max()) < 1e-4, f"max |diff| = {float(d_eta.max()):.2e}")

    # The sign convention is load-bearing: without it, exactly the negative-eta
    # hemisphere disagrees. Assert that failure mode explicitly so a future
    # change to the default cannot pass silently.
    pos_unflipped = relative_positions(p4, valid, axis=axis_stored,
                                       eta_sign_convention=False)
    neg = axis_stored[0] < 0
    bad = (pos_unflipped[..., 0] - ref[..., 0])[valid].abs().max()
    good_pos = (pos_unflipped[~neg][..., 0] - ref[~neg][..., 0])[valid[~neg]].abs().max()
    check("dropping the eta sign convention breaks exactly the -eta hemisphere",
          float(bad) > 1.0 and float(good_pos) < 1e-4,
          f"unflipped: max |diff| {float(bad):.2e} overall, "
          f"{float(good_pos):.2e} on +eta jets ({float(neg.float().mean()):.1%} are -eta)")

    # (ii) against our own summed-p4 axis -- the fallback used when pf_points is
    #      unavailable (ONNX export, the pf+sv Tagger variant). jet_pt matches the
    #      constituent sum to 1e-8, so no constituents are missing, but jet_eta is
    #      NOT exactly asinh(pz_sum/pt_sum): it disagrees by ~3e-4 (median).
    #
    #      That disagreement is harmless, and this is the check that proves it:
    #      the residual is a pure per-jet TRANSLATION. A constant offset shared by
    #      every constituent of a jet leaves all inter-particle distances, the
    #      median-split partition, and every ball-centroid-relative quantity
    #      exactly unchanged. Only the absolute delta_eta origin moves.
    pos_ours = relative_positions(p4, valid)
    o_phi = (pos_ours[..., 1] - ref[..., 1])[valid].abs()
    check("delta_phi vs part_dphi (summed-p4 axis)",
          float(o_phi.max()) < 5e-3,
          f"max {float(o_phi.max()):.2e}, median {float(o_phi.median()):.2e}")

    resid = pos_ours[..., 0] - ref[..., 0]
    cnt = valid.sum(1).clamp(min=1).float()
    mean = (resid * valid).sum(1) / cnt
    var = (((resid - mean.unsqueeze(1)) ** 2) * valid).sum(1) / cnt
    check("delta_eta residual is a pure per-jet translation",
          float(var.sqrt().max()) < 1e-5,
          f"per-jet std max {float(var.sqrt().max()):.2e}, "
          f"offset median {float(mean.abs().median()):.2e}")

    # The consequence that actually matters: pairwise geometry is untouched.
    k = 24
    dr_ref = delta_r(ref[:64, :k, None, :], ref[:64, None, :k, :])
    dr_ours = delta_r(pos_ours[:64, :k, None, :], pos_ours[:64, None, :k, :])
    pair_ok = valid[:64, :k, None] & valid[:64, None, :k]
    check("pairwise delta_R is identical under either axis",
          float((dr_ref - dr_ours)[pair_ok].abs().max()) < 1e-5,
          f"max |diff| = {float((dr_ref - dr_ours)[pair_ok].abs().max()):.2e}")

    n_seam = int((torch.from_numpy(np.abs(ev["jet_phi"])) > math.pi - 0.4).sum())
    print(f"       ({n_seam} of {NJ} jets sit within 0.4 rad of the +/-pi seam)")


# ------------------------------------------------------------- C. slot layout --
print("\nC. fixed-length slot layout")
L = 64
N, P = 8, 100
pt = torch.rand(N, P) * 100
valid = torch.rand(N, P) > 0.4
pt = pt * valid                                     # invalid slots carry no pt

idx, valid_L = sort_truncate_pad(valid, pt, L)
check("idx / valid_L shapes", idx.shape == (N, L) and valid_L.shape == (N, L))
check("kept count = min(n_real, L)",
      bool((valid_L.sum(1) == torch.minimum(valid.sum(1),
                                            torch.tensor(L))).all()))

kept_pt = torch.gather(pt, 1, idx)
check("valid slots are pt-descending",
      bool(all((kept_pt[i][valid_L[i]].diff() <= 1e-6).all() for i in range(N))))
check("all valid slots precede all invalid slots",
      bool(all(valid_L[i].long().diff().min() >= -1 and
               not (valid_L[i, 1:] & ~valid_L[i, :-1]).any() for i in range(N))))

feat = torch.randn(N, P, 5)
g1 = take_slots(feat, idx)
perm = torch.randperm(P)
idx2, valid_L2 = sort_truncate_pad(valid[:, perm], pt[:, perm], L)
g2 = take_slots(feat[:, perm], idx2)
sel1 = torch.where(valid_L.unsqueeze(-1), g1, torch.zeros_like(g1))
sel2 = torch.where(valid_L2.unsqueeze(-1), g2, torch.zeros_like(g2))
check("layout is invariant to input particle order",
      torch.allclose(sel1, sel2, atol=1e-6),
      f"max diff = {float((sel1 - sel2).abs().max()):.2e}")

# truncation must keep the hardest, not the first
pt_small = torch.tensor([[5.0, 100.0, 3.0, 50.0]])
v_small = torch.ones(1, 4, dtype=torch.bool)
i_s, _ = sort_truncate_pad(v_small, pt_small, 2)
check("truncation keeps the two hardest constituents",
      set(i_s[0].tolist()) == {1, 3}, f"kept slots {i_s[0].tolist()} (pt 100, 50)")

# centroid must ignore padded slots
pos = torch.tensor([[[0.0, 0.0], [2.0, 2.0], [99.0, 99.0]]])
vmask = torch.tensor([[True, True, False]])
c = masked_centroid(pos, vmask)
check("masked centroid ignores padded slots",
      torch.allclose(c, torch.tensor([[[1.0, 1.0]]]), atol=1e-6),
      f"c_B = {c.flatten().tolist()}")
check("all-invalid group gives zeros, not NaN",
      torch.isfinite(masked_centroid(pos, torch.zeros(1, 3, dtype=torch.bool))).all())

print()
if failures:
    print(f"S1 FAILED: {len(failures)} check(s) - {', '.join(failures)}")
    sys.exit(1)
print("S1 PASSED")
