"""S2 - ball-tree parity and invariants.

Two questions:

  A. Does our batched pure-PyTorch tree partition points the same way Erwin's
     compiled `balltree` package does? This is the oracle check, run on real
     jets with exactly 64 constituents so neither side pads.
  B. Does our tree hold the invariants the model depends on, whether or not it
     agrees with the oracle bit-for-bit?

(B) is the part that actually gates the model. (A) tells us how far we have
drifted from the reference implementation, which matters for reading the Erwin
paper's results across to ours.

Needs `balltree` for part A. That lives in the system python, not the weaver
env - the training path never imports it:
    python3 tests/s2_tree_parity.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "networks"))
from erwin_part.balltree_torch import (  # noqa: E402
    build_ball_tree, ball_view, build_levels, rotate_positions,
)
from erwin_part.geometry import delta_r, relative_positions, sort_truncate_pad, take_slots  # noqa: E402

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
L, BALL = 64, 8
failures = []


def check(name, ok, detail=""):
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}{' - ' + detail if detail else ''}")
    if not ok:
        failures.append(name)


def partition(perm, ball_size):
    """(N, L) permutation -> list of per-jet sets of frozenset ball memberships."""
    balls = ball_view(perm, ball_size)
    return [{frozenset(b.tolist()) for b in jet} for jet in balls]


# ------------------------------------------------------ load real 64-jets ----
root = next((p for p in (
    os.path.join(WORKSPACE, "efficient_particle_transformer", "JetClass_example_100k.root"),
    os.path.join(WORKSPACE, "JetClass_example_100k.root")) if os.path.exists(p)), None)
assert root, "JetClass sample not found"

import uproot  # noqa: E402
ev = uproot.open(root)["tree"].arrays(
    ["part_px", "part_py", "part_pz", "part_energy", "part_deta", "part_dphi"],
    library="np", entry_stop=20000)

sel = [i for i in range(len(ev["part_px"])) if len(ev["part_px"][i]) >= L][:512]
NJ = len(sel)
print(f"using {NJ} real jets with >= {L} constituents (top-{L} by pt, no padding)\n")

p4 = torch.zeros(NJ, L, 4)
pos = torch.zeros(NJ, L, 2)
for n, i in enumerate(sel):
    px, py = ev["part_px"][i], ev["part_py"][i]
    keep = np.argsort(-(px ** 2 + py ** 2))[:L]           # pt-descending
    for c, k in enumerate(("part_px", "part_py", "part_pz", "part_energy")):
        p4[n, :, c] = torch.from_numpy(ev[k][i][keep].astype(np.float32))
    pos[n, :, 0] = torch.from_numpy(ev["part_deta"][i][keep].astype(np.float32))
    pos[n, :, 1] = torch.from_numpy(ev["part_dphi"][i][keep].astype(np.float32))
valid = torch.ones(NJ, L, dtype=torch.bool)

ours = build_ball_tree(pos, valid, min_leaf=1)

# ------------------------------------------------------------ A. oracle ------
print("A. parity against the balltree package")
try:
    from balltree import build_balltree_with_rotations

    flat_pos = pos.reshape(-1, 2)
    batch_idx = torch.arange(NJ).repeat_interleave(L)
    tree_idx, tree_mask, _ = build_balltree_with_rotations(
        flat_pos, batch_idx, [], [BALL], 0.0)

    check("oracle padded nothing (64 is already a power of two)",
          int(tree_mask.sum()) == NJ * L and tree_idx.shape[0] == NJ * L,
          f"{int(tree_mask.sum())}/{tree_idx.shape[0]} slots real")

    # Each jet must occupy one contiguous block of tree slots.
    blocks = batch_idx[tree_idx].view(NJ, L)
    contiguous = bool((blocks == blocks[:, :1]).all())
    check("oracle keeps each jet contiguous in tree order", contiguous)

    if contiguous:
        # Map global -> per-jet local indices, jets in oracle block order.
        jet_of_block = blocks[:, 0]
        oracle_local = (tree_idx.view(NJ, L) - jet_of_block.view(NJ, 1) * L)
        # Reorder our jets to match the oracle's block order.
        ours_aligned = ours[jet_of_block]

        po, pt_ = partition(oracle_local, BALL), partition(ours_aligned, BALL)
        agree = sum(a == b for a, b in zip(po, pt_))
        check(f"level-0 partition matches oracle on all {NJ} jets",
              agree == NJ, f"{agree}/{NJ} jets identical ({agree/NJ:.1%})")

        if agree != NJ:
            # Quantify the drift: how many balls differ, and are ours as compact?
            shared = [len(a & b) for a, b in zip(po, pt_)]
            nb = L // BALL
            print(f"       balls identical per jet: mean {np.mean(shared):.2f}/{nb}")

            def compactness(perm_local):
                b = ball_view(perm_local, BALL)
                p = torch.gather(pos[jet_of_block], 1,
                                 perm_local.unsqueeze(-1).expand(NJ, L, 2)
                                 ).view(NJ, L // BALL, BALL, 2)
                d = delta_r(p.unsqueeze(3), p.unsqueeze(2))
                iu = torch.triu_indices(BALL, BALL, offset=1)
                return float(d[:, :, iu[0], iu[1]].mean())

            co, ct = compactness(oracle_local), compactness(ours_aligned)
            print(f"       mean intra-ball delta_R: oracle {co:.4f}  ours {ct:.4f}  "
                  f"({'ours tighter' if ct <= co else 'oracle tighter'})")
except ImportError:
    print("  [skip] balltree not importable in this interpreter - run with system python3")

# --------------------------------------------------------- B. invariants -----
print("\nB. invariants the model depends on")

check("perm is a permutation of every jet's slots",
      bool((torch.sort(ours, dim=1).values ==
            torch.arange(L).expand(NJ, L)).all()))

shift = torch.tensor([0.37, -0.21])
check("partition is translation-invariant",
      partition(build_ball_tree(pos + shift, valid), BALL) == partition(ours, BALL))

g = torch.Generator().manual_seed(0)
sh = torch.stack([torch.randperm(L, generator=g) for _ in range(NJ)])
pos_sh = torch.gather(pos, 1, sh.unsqueeze(-1).expand(NJ, L, 2))
perm_sh = build_ball_tree(pos_sh, valid)
back = torch.gather(sh, 1, perm_sh)                      # rotated back to original ids
check("partition is invariant to input particle order",
      partition(back, BALL) == partition(ours, BALL))

rot = build_ball_tree(rotate_positions(pos, 45.0), valid)
differing = sum(a != b for a, b in zip(partition(rot, BALL), partition(ours, BALL)))
check("rotation yields a genuinely different partition",
      differing > 0.9 * NJ, f"{differing}/{NJ} jets repartitioned ({differing/NJ:.1%})")

# Balls must be spatially tight, or ball-local attention is meaningless.
p = torch.gather(pos, 1, ours.unsqueeze(-1).expand(NJ, L, 2)).view(NJ, L // BALL, BALL, 2)
iu = torch.triu_indices(BALL, BALL, offset=1)
intra = float(delta_r(p.unsqueeze(3), p.unsqueeze(2))[:, :, iu[0], iu[1]].mean())
allp = torch.gather(pos, 1, ours.unsqueeze(-1).expand(NJ, L, 2))
iu2 = torch.triu_indices(L, L, offset=1)
overall = float(delta_r(allp.unsqueeze(2), allp.unsqueeze(1))[:, iu2[0], iu2[1]].mean())
check("balls are spatially compact",
      intra < 0.55 * overall,
      f"intra-ball delta_R {intra:.4f} vs all-pairs {overall:.4f} "
      f"({intra/overall:.2f}x)")

# Padding must sink to the last balls, not smear across every ball.
vmix = valid.clone()
vmix[:, 40:] = False                                     # 24 padded slots per jet
perm_mix = build_ball_tree(pos, vmix, min_leaf=1)
vt = torch.gather(vmix, 1, perm_mix).view(NJ, L // BALL, BALL)
partial = int(((vt.any(-1)) & (~vt.all(-1))).sum())
empty = int((~vt.any(-1)).sum())
check("padding concentrates in the trailing balls",
      partial <= NJ, f"{partial} partially-filled and {empty} all-padding balls "
                     f"across {NJ} jets ({NJ * L // BALL} balls total)")
check("every real particle is still present after padding sinks",
      bool((torch.gather(vmix, 1, perm_mix).sum(1) == vmix.sum(1)).all()))

# Multi-level build: shapes, validity propagation, single-ball bottleneck.
levels = build_levels(pos, valid, [8, 8, 16], [2, 2], rotate=45.0)
check("build_levels shapes coarsen 64 -> 32 -> 16",
      [l["perm"].shape[1] for l in levels] == [64, 32, 16],
      f"{[l['perm'].shape[1] for l in levels]}")
check("bottleneck is a single ball",
      levels[-1]["n_balls"] == 1, f"n_balls per level "
      f"{[l['n_balls'] for l in levels]}")

lv_mix = build_levels(pos, vmix, [8, 8, 16], [2, 2], rotate=45.0)
check("parent validity = any(child valid)",
      int(lv_mix[1]["valid"].sum(1).max()) <= 32
      and bool((lv_mix[1]["valid"].sum(1) >= (vmix.sum(1) / 2).ceil()).all()),
      f"level-1 valid nodes: {int(lv_mix[1]['valid'].sum(1).float().mean())} of 32 "
      f"for {int(vmix.sum(1).float().mean())} real particles")

print()
if failures:
    print(f"S2 FAILED: {len(failures)} check(s) - {', '.join(failures)}")
    sys.exit(1)
print("S2 PASSED")
