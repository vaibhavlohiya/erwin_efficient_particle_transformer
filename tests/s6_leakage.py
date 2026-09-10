"""S6 - cross-jet isolation.

ErwinParTv2 flattens a batch of jets into shared (N, L, ...) tensors and reshapes
them into balls. A single mis-shaped view - a reshape that folds the batch axis
into the ball axis, a mask broadcast one dimension off - silently lets one jet's
constituents attend to another's. The loss still falls; the model is just
learning from a physically meaningless object.

Three progressively stronger statements:

  1. perturbing one jet moves only that jet's logits
  2. d(logits of jet a) / d(embedded features of jet b) is exactly zero, a != b
  3. a jet scored alone gives the same logits as the same jet scored in a batch

(3) is the strongest and is the one that also catches normalisation leakage. It
holds in eval mode. In *training* mode the BatchNorms inside the pair MLP couple
the batch by construction - true of stock ParT as well - so that case is
asserted to differ rather than pretended away.

Run from the weaver conda env:
    python tests/s6_leakage.py
"""
import os
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


import uproot  # noqa: E402
root = next((p for p in (
    os.path.join(WORKSPACE, "efficient_particle_transformer", "JetClass_example_100k.root"),
    os.path.join(WORKSPACE, "JetClass_example_100k.root")) if os.path.exists(p)), None)
assert root, "JetClass sample not found"

N, C = 10, 7
ev = uproot.open(root)["tree"].arrays(
    ["part_px", "part_py", "part_pz", "part_energy", "part_deta", "part_dphi",
     "part_charge", "part_d0val", "part_dzval"], library="np", entry_stop=400)
pick = list(range(N))
P = max(len(ev["part_px"][i]) for i in pick)

x = torch.zeros(N, C, P); v = torch.zeros(N, 4, P)
pts = torch.zeros(N, 2, P); mask = torch.zeros(N, 1, P, dtype=torch.bool)
for n, i in enumerate(pick):
    k = len(ev["part_px"][i])
    p = np.stack([ev[b][i] for b in ("part_px", "part_py", "part_pz", "part_energy")])
    v[n, :, :k] = torch.from_numpy(p.astype(np.float32))
    x[n, :, :k] = torch.from_numpy(np.stack([
        np.log(np.clip(np.hypot(p[0], p[1]), 1e-6, None)),
        np.log(np.clip(p[3], 1e-6, None)), ev["part_deta"][i], ev["part_dphi"][i],
        ev["part_charge"][i], np.tanh(ev["part_d0val"][i]),
        np.tanh(ev["part_dzval"][i])]).astype(np.float32))
    pts[n, :, :k] = torch.from_numpy(np.stack(
        [ev["part_deta"][i], ev["part_dphi"][i]]).astype(np.float32))
    mask[n, 0, :k] = True
mult = mask.sum(-1).squeeze(-1)
print(f"batch of {N} real jets, P={P}, multiplicity {int(mult.min())}-{int(mult.max())}\n")

model = ErwinParticleTransformerV2(input_dim=C, num_classes=10, seq_len=64).eval()
with torch.no_grad():
    base = model(x, v=v, mask=mask, points=pts)

# ------------------------------------------------------ 1. perturbation ------
print("1. perturbing one jet")
TARGET = 4
xp = x.clone()
xp[TARGET, :, :int(mult[TARGET])] += torch.randn(C, int(mult[TARGET])) * 0.5
with torch.no_grad():
    out = model(xp, v=v, mask=mask, points=pts)
others = torch.cat([out[:TARGET], out[TARGET + 1:]]) - torch.cat([base[:TARGET], base[TARGET + 1:]])
check("the perturbed jet's logits move",
      float((out[TARGET] - base[TARGET]).abs().max()) > 1e-3,
      f"max |diff| = {float((out[TARGET] - base[TARGET]).abs().max()):.2e}")
check("every other jet is untouched", float(others.abs().max()) < 1e-6,
      f"max |diff| = {float(others.abs().max()):.2e}")

# Perturbing the *geometry* of one jet is the case that stresses the tree.
pp = pts.clone()
pp[TARGET, :, :int(mult[TARGET])] += torch.randn(2, int(mult[TARGET])) * 0.2
with torch.no_grad():
    out = model(x, v=v, mask=mask, points=pp)
others = torch.cat([out[:TARGET], out[TARGET + 1:]]) - torch.cat([base[:TARGET], base[TARGET + 1:]])
check("repartitioning one jet's tree leaves the others untouched",
      float(others.abs().max()) < 1e-6 and
      float((out[TARGET] - base[TARGET]).abs().max()) > 1e-3,
      f"target moved {float((out[TARGET] - base[TARGET]).abs().max()):.2e}, "
      f"others {float(others.abs().max()):.2e}")

# ------------------------------------------------------ 2. gradient paths ----
print("\n2. gradient isolation")
# The raw input carries no gradient: ParT (and this model) run SequenceTrimmer
# under no_grad, which detaches it. The first differentiable point is Embed's
# output, and isolation there covers the entire tree / ball-attention / readout
# stack - which is where a bad reshape would actually leak.
store = {}


def _capture(_mod, _inp, out):
    out.retain_grad()
    store["h"] = out


handle = model.embed.register_forward_hook(_capture)
out = model(x, v=v, mask=mask, points=pts)
out[TARGET].sum().backward()
handle.remove()

g = store["h"].grad                                   # (P, N, C): jets on dim 1
own = float(g[:, TARGET].abs().max())
leak = float(torch.cat([g[:, :TARGET], g[:, TARGET + 1:]], dim=1).abs().max())
check("d(logits_b)/d(features_b) is non-zero", own > 1e-8, f"max |grad| = {own:.2e}")
check("d(logits_b)/d(features_a) is exactly zero for a != b", leak == 0.0,
      f"max |grad| on other jets = {leak:.2e}")

# ------------------------------------------------ 3. batch-size independence --
print("\n3. batch composition (eval mode)")
with torch.no_grad():
    alone = torch.cat([model(x[i:i + 1], v=v[i:i + 1], mask=mask[i:i + 1],
                             points=pts[i:i + 1]) for i in range(N)])
check("scoring each jet alone matches scoring the batch",
      float((alone - base).abs().max()) < 1e-5,
      f"max |diff| = {float((alone - base).abs().max()):.2e}")

order = torch.randperm(N, generator=torch.Generator().manual_seed(11))
with torch.no_grad():
    shuffled = model(x[order], v=v[order], mask=mask[order], points=pts[order])
check("reordering jets permutes the logits and nothing else",
      float((shuffled - base[order]).abs().max()) < 1e-5,
      f"max |diff| = {float((shuffled - base[order]).abs().max()):.2e}")

sub = [0, 3, 7]
with torch.no_grad():
    part = model(x[sub], v=v[sub], mask=mask[sub], points=pts[sub])
check("a sub-batch reproduces its members' logits",
      float((part - base[sub]).abs().max()) < 1e-5,
      f"max |diff| = {float((part - base[sub]).abs().max()):.2e}")

# ---------------------------------------- 4. training mode couples on purpose --
print("\n4. training mode (documented coupling)")
model.train()
for m_ in model.modules():
    if isinstance(m_, torch.nn.Dropout):
        m_.p = 0.0
model.trimmer.enabled = False              # isolate BN from the random trimmer
with torch.no_grad():
    t_batch = model(x, v=v, mask=mask, points=pts)
    t_alone = torch.cat([model(x[i:i + 1], v=v[i:i + 1], mask=mask[i:i + 1],
                               points=pts[i:i + 1]) for i in range(N)])
delta = float((t_alone - t_batch).abs().max())
check("BatchNorm in the pair MLP couples the batch while training, as in ParT",
      delta > 1e-4,
      f"max |diff| = {delta:.2e} - expected, and why S6's isolation claims are "
      f"stated for eval mode")

bn = [m for m in model.pair_embeds.modules() if isinstance(m, torch.nn.BatchNorm1d)]
check("the fixed input standardiser is exempt (stays in eval)",
      not bn[0].training and sum(b.training for b in bn) == len(bn) - 3,
      f"{sum(b.training for b in bn)} of {len(bn)} BatchNorms live "
      f"(3 frozen input standardisers, one per level)")

print()
if failures:
    print(f"S6 FAILED: {len(failures)} check(s) - {', '.join(failures)}")
    sys.exit(1)
print("S6 PASSED")
