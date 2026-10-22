"""S12 - Lund splitting tokens inside ErwinParTv2.

S12b checks the clustering against FastJet in a weaver-free interpreter. This
checks everything around it, in the weaver env:

  1. splitting features - one splitting is bit-identical to weaver's
     `pairwise_lv_fts` for the same two particles (the pair features' source)
  2. token set - invariant to slot order and to extra padding; truncation keeps
     the hardest kT; the kT cut keeps only ln kT > cut
  3. model - both placements run on real jets, every Lund parameter is trained,
     the tokens actually move the logits, and the bottleneck guard fires
  4. isolation - a jet scored alone matches the batch, and particle order does
     not matter, with Lund tokens on

Run from the weaver conda env:
    python tests/s12_lund.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "networks"))
from erwin_part.lund import lund_splittings  # noqa: E402
from erwin_part.model import ErwinParticleTransformerV2  # noqa: E402
from weaver.nn.model.ParticleTransformer import pairwise_lv_fts  # noqa: E402

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

N, C = 12, 7
ev = uproot.open(root)["tree"].arrays(
    ["part_px", "part_py", "part_pz", "part_energy", "part_deta", "part_dphi",
     "part_charge", "part_d0val", "part_dzval"], library="np", entry_stop=400)
P = max(len(ev["part_px"][i]) for i in range(N))

x = torch.zeros(N, C, P); v = torch.zeros(N, 4, P)
pts = torch.zeros(N, 2, P); mask = torch.zeros(N, 1, P, dtype=torch.bool)
for n in range(N):
    k = len(ev["part_px"][n])
    p = np.stack([ev[b][n] for b in ("part_px", "part_py", "part_pz", "part_energy")])
    v[n, :, :k] = torch.from_numpy(p.astype(np.float32))
    x[n, :, :k] = torch.from_numpy(np.stack([
        np.log(np.clip(np.hypot(p[0], p[1]), 1e-6, None)),
        np.log(np.clip(p[3], 1e-6, None)), ev["part_deta"][n], ev["part_dphi"][n],
        ev["part_charge"][n], np.tanh(ev["part_d0val"][n]),
        np.tanh(ev["part_dzval"][n])]).astype(np.float32))
    pts[n, :, :k] = torch.from_numpy(np.stack(
        [ev["part_deta"][n], ev["part_dphi"][n]]).astype(np.float32))
    mask[n, 0, :k] = True

# Slot layout as the model builds it, in float64.
L = 64
p4 = torch.zeros(N, L, 4, dtype=torch.float64)
pos = torch.zeros(N, L, 2, dtype=torch.float64)
valid = torch.zeros(N, L, dtype=torch.bool)
for n in range(N):
    k = int(mask[n].sum())
    order = torch.argsort(torch.hypot(v[n, 0, :k], v[n, 1, :k]), descending=True)[:L]
    p4[n, :len(order)] = v[n, :, order].T.double()
    pos[n, :len(order)] = pts[n, :, order].T.double()
    valid[n, :len(order)] = True

# --------------------------------------------------------- 1. features -------
print("1. splitting features")
two = valid.clone(); two[:, 2:] = False
f2, _, t2 = lund_splittings(p4, two, max_splits=None)
ref = pairwise_lv_fts(p4[:, 0].unsqueeze(-1), p4[:, 1].unsqueeze(-1),
                      num_outputs=3).squeeze(-1)                      # (N, 3)
check("a single splitting equals pairwise_lv_fts [ln kT, ln z, ln dR]",
      bool(t2[:, 0].all()) and torch.allclose(f2[:, 0], ref, atol=1e-10),
      f"max |diff| {float((f2[:, 0] - ref).abs().max()):.1e}")
check("a two-particle jet yields exactly one token", int(t2.sum()) == N)

full, fpos, fv = lund_splittings(p4, valid, pos, max_splits=None)
k = valid.sum(1)
check("k constituents give k-1 tokens", bool((fv.sum(1) == k - 1).all()))
check("ln z <= ln(1/2) for every token (b is the softer branch)",
      bool((full[..., 1][fv] <= np.log(0.5) + 1e-12).all()))

# --------------------------------------------------------- 2. token set ------
print("\n2. token set")
perm = torch.stack([torch.randperm(L) for _ in range(N)])
g = lambda t: torch.gather(t, 1, perm.unsqueeze(-1).expand(-1, -1, t.shape[-1]))
fp, _, vp = lund_splittings(g(p4), torch.gather(valid, 1, perm), g(pos), max_splits=None)
check("invariant to slot order", torch.equal(vp, fv) and
      torch.allclose(fp, full, atol=1e-9), f"max |diff| {float((fp - full).abs().max()):.1e}")

pad = lambda t: torch.cat([t, t.new_zeros(N, 32, *t.shape[2:])], dim=1)
fq, _, vq = lund_splittings(pad(p4), pad(valid), pad(pos), max_splits=L - 1)
check("invariant to extra padding", torch.equal(vq, fv) and torch.equal(fq, full))

f10, _, v10 = lund_splittings(p4, valid, pos, max_splits=10)
check("max_splits keeps the hardest-kT tokens",
      torch.equal(f10, full[:, :10]) and bool(v10.all()))
check("tokens come out hardest first",
      bool((full[:, :-1, 0] >= full[:, 1:, 0])[fv[:, 1:]].all()))

fc, _, vc = lund_splittings(p4, valid, pos, max_splits=None, ktcut=0.0)
check("kT cut keeps exactly the ln kT > 0 tokens",
      int(vc.sum()) == int((full[..., 0][fv] > 0).sum()) and
      bool((fc[..., 0][vc] > 0).all()), f"{int(vc.sum())}/{int(fv.sum())} kept")

# The last merge is the whole jet: its position is the pt-weighted jet centroid.
ptw = torch.hypot(p4[..., 0], p4[..., 1]) * valid
centroid = (pos * ptw.unsqueeze(-1)).sum(1) / ptw.sum(1, keepdim=True)
_, rp, _ = lund_splittings(p4, valid, pos, alg="ca", max_splits=None)
# The root is not necessarily the hardest in kT, so look for it by position.
best = (rp - centroid.unsqueeze(1)).norm(dim=-1).min(1).values
check("one token per jet sits at the pt-weighted jet centroid (the root)",
      float(best.max()) < 1e-9, f"max offset {float(best.max()):.1e}")

# --------------------------------------------------------- 3. model ----------
print("\n3. model")
base = ErwinParticleTransformerV2(input_dim=C, num_classes=10)
n0 = sum(p.numel() for p in base.parameters())
models = {w: ErwinParticleTransformerV2(input_dim=C, num_classes=10, use_lund=True,
                                        lund_where=w) for w in ("bottleneck", "readout")}
n1 = sum(p.numel() for p in models["bottleneck"].parameters())
check("Lund tokens add < 3% parameters (PLuM: +2%)", (n1 - n0) / n0 < 0.03,
      f"{n0:,} -> {n1:,} (+{(n1 - n0) / n0:.1%})")

for where, model in models.items():
    model.train()
    out = model(x, v=v, mask=mask, points=pts)
    out.sum().backward()
    dead = [n for n, p in model.named_parameters()
            if "lund" in n and p.requires_grad and (p.grad is None or not p.grad.abs().sum())]
    check(f"{where}: training forward is finite, every Lund parameter gets gradient",
          bool(torch.isfinite(out).all()) and not dead, f"dead: {dead}" if dead else "")

    model.eval()
    # A random direction: a constant shift of every channel is erased by the
    # LayerNorm each token meets first, so it would prove nothing.
    with torch.no_grad():
        a = model(x, v=v, mask=mask, points=pts)
        delta = torch.randn_like(model.lund_token)
        model.lund_token.add_(delta)
        b = model(x, v=v, mask=mask, points=pts)
        model.lund_token.sub_(delta)
    check(f"{where}: the Lund tokens move the logits",
          float((a - b).abs().max()) > 1e-3, f"max |diff| {float((a - b).abs().max()):.2e}")

try:
    ErwinParticleTransformerV2(input_dim=C, num_classes=10, use_lund=True,
                               ball_sizes=[8, 8, 8])
    raised = False
except ValueError:
    raised = True
check("bottleneck placement refuses a top level that is not one ball", raised)

# --------------------------------------------------------- 4. isolation ------
print("\n4. isolation (eval mode)")
model = models["bottleneck"].eval()
with torch.no_grad():
    batch = model(x, v=v, mask=mask, points=pts)
    alone = torch.cat([model(x[i:i + 1], v=v[i:i + 1], mask=mask[i:i + 1],
                             points=pts[i:i + 1]) for i in range(N)])
check("scoring each jet alone matches the batch",
      float((alone - batch).abs().max()) < 1e-5,
      f"max |diff| {float((alone - batch).abs().max()):.2e}")

shuf = torch.randperm(P)
with torch.no_grad():
    sh = model(x[..., shuf], v=v[..., shuf], mask=mask[..., shuf], points=pts[..., shuf])
check("particle order does not change the logits",
      float((sh - batch).abs().max()) < 1e-4, f"max |diff| {float((sh - batch).abs().max()):.2e}")

if torch.backends.mps.is_available():
    # Compare the tokens, not the logits: the baseline model's MPS and CPU
    # logits already differ at the 1e-1 level, independent of Lund.
    lc = model._lund(pos.float(), p4.float(), valid)
    lm = model._lund(pos.float().to("mps"), p4.float().to("mps"), valid.to("mps"))
    same = all(torch.equal(a_, b_.cpu()) for a_, b_ in zip(lc, lm))
    check("MPS path (clustered on CPU in float64) gives the CPU tokens exactly", same)

print()
print("S12 FAILED: " + ", ".join(failures) if failures else "S12 passed")
sys.exit(1 if failures else 0)
