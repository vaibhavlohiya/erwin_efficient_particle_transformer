"""S7 - gradient flow and numerical behaviour.

Ball attention stacks three additive terms inside one softmax - ParT's U, Erwin's
distance bias, and a padded-key penalty - on top of a tree, a pooling step and a
multi-scale readout. Plenty of places for a term to be silently dead (a branch
that never receives gradient) or silently explosive (a bias that dominates the
logits before training starts).

Checks:
  1. every trainable parameter receives a finite, non-zero gradient
  2. the attention bias is well-scaled at initialisation - U and the distance
     term comparable, neither saturating the softmax
  3. sigma is negative-by-construction via softplus, at every level
  4. bf16 autocast forward and backward stay finite
  5. degenerate batches (an empty jet) do not produce NaN gradients
  6. a few optimiser steps actually reduce the loss

Run from the weaver conda env:
    python tests/s7_gradients.py
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "networks"))
from erwin_part.ball_attention import NEG  # noqa: E402
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

N, C = 16, 7
LABELS = ["label_QCD", "label_Hbb", "label_Hcc", "label_Hgg", "label_H4q",
          "label_Hqql", "label_Zqq", "label_Wqq", "label_Tbqq", "label_Tbl"]
# JetClass files are sorted by class, so a contiguous prefix is one class.
# Read the (flat, cheap) label branches over the whole file and stratify, or the
# "loss decreases" check below is trained on a single-class batch and vacuous.
_tree = uproot.open(root)["tree"]
_lab = np.stack([_tree[l].array(library="np") for l in LABELS], axis=1).argmax(1)
rng = np.random.default_rng(0)
pick = np.concatenate([rng.choice(np.flatnonzero(_lab == c), N // 10, replace=False)
                       for c in range(10)])
pick = np.sort(pick)[:N]
ev = _tree.arrays(
    ["part_px", "part_py", "part_pz", "part_energy", "part_deta", "part_dphi",
     "part_charge", "part_d0val", "part_dzval"] + LABELS, library="np",
    entry_start=int(pick.min()), entry_stop=int(pick.max()) + 1)
pick = pick - int(pick.min())
P = max(len(ev["part_px"][i]) for i in pick)

x = torch.zeros(N, C, P); v = torch.zeros(N, 4, P)
pts = torch.zeros(N, 2, P); mask = torch.zeros(N, 1, P, dtype=torch.bool)
y = torch.zeros(N, dtype=torch.long)
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
    y[n] = int(np.argmax([ev[l][i] for l in LABELS]))
print(f"batch of {N} class-mixed jets, P={P}, {len(set(y.tolist()))} classes present\n")

model = ErwinParticleTransformerV2(input_dim=C, num_classes=10, seq_len=64)
model.train()
model.trimmer.enabled = False

# --------------------------------------------------------- 1. gradient flow --
print("1. gradient flow (fp32)")
loss = F.cross_entropy(model(x, v=v, mask=mask, points=pts), y)
loss.backward()

trainable = [(n_, p_) for n_, p_ in model.named_parameters() if p_.requires_grad]
missing = [n_ for n_, p_ in trainable if p_.grad is None]
nonfinite = [n_ for n_, p_ in trainable if p_.grad is not None and not torch.isfinite(p_.grad).all()]
dead = [n_ for n_, p_ in trainable if p_.grad is not None and float(p_.grad.abs().max()) == 0.0]

check("initial loss is near ln(10) for 10 balanced classes",
      1.8 < loss.item() < 3.2, f"loss = {loss.item():.4f}, ln(10) = {np.log(10):.4f}")
check("every trainable parameter received a gradient", not missing,
      f"{len(trainable)} tensors, {len(missing)} missing" +
      (f": {missing[:3]}" if missing else ""))
check("no gradient is non-finite", not nonfinite, f"{len(nonfinite)} non-finite")
check("no gradient is identically zero", not dead,
      f"{len(dead)} dead" + (f": {dead[:3]}" if dead else ""))

gnorm = torch.norm(torch.stack([p_.grad.norm() for _, p_ in trainable if p_.grad is not None]))
check("global gradient norm is sane at initialisation", 1e-3 < float(gnorm) < 1e3,
      f"||g|| = {float(gnorm):.3f}")

# every level must be live, not just the first
for li in range(3):
    s = model.levels[li][0].attn.sigma
    pe = model.pair_embeds[li].embed[1].weight
    check(f"level {li}: sigma and pair MLP both receive gradient",
          s.grad is not None and float(s.grad.abs().max()) > 0
          and pe.grad is not None and float(pe.grad.abs().max()) > 0,
          f"|d sigma| = {float(s.grad.abs().max()):.2e}, "
          f"|d W_pair| = {float(pe.grad.abs().max()):.2e}")
check("the multi-scale level embedding is live",
      float(model.level_embed.grad.abs().max()) > 0,
      f"|d level_embed| = {float(model.level_embed.grad.abs().max()):.2e}")

# --------------------------------------------------------- 2. bias scaling ---
print("\n2. attention bias at initialisation")
captured = {}
orig = ErwinParticleTransformerV2._level_context


def spy(self, li, pos, p4, valid, order):
    out = orig(self, li, pos, p4, valid, order)
    if order is None:
        captured[li] = out
    return out


ErwinParticleTransformerV2._level_context = spy
with torch.no_grad():
    model.eval()
    model(x, v=v, mask=mask, points=pts)
ErwinParticleTransformerV2._level_context = orig

# Only real query / real key entries carry meaning: padded keys are masked to
# NEG and padded query rows are zeroed on output, so they are excluded here.
for li in range(3):
    _, vb, u, dist = captured[li]
    sig = model.levels[li][0].attn.sigma
    dterm = F.softplus(sig) * dist                     # (N, nb, 1, m, m)
    rr = (vb.unsqueeze(-1) & vb.unsqueeze(-2)).unsqueeze(2)     # (N, nb, 1, m, m)
    rr_u = rr.expand_as(u)

    u_mag = float(u[rr_u].abs().mean())
    d_mag = float(dterm[rr.expand_as(dterm)].abs().mean())
    check(f"level {li}: U and the distance term are comparable in scale",
          0.02 < u_mag / max(d_mag, 1e-9) < 50,
          f"|U| = {u_mag:.3f}, |softplus(sigma)*d| = {d_mag:.3f}")

    tot = (u - dterm.expand_as(u))[rr_u]
    check(f"level {li}: the bias does not saturate the softmax",
          float(tot.abs().max()) < 30,
          f"max |bias| = {float(tot.abs().max()):.2f} (softmax saturates ~>30)")

    dh = dist[rr]
    check(f"level {li}: normalised distance is scale-free (mean ~1)",
          0.7 < float(dh.mean()) < 2.5,
          f"mean d_hat = {float(dh.mean()):.2f}, p99 = "
          f"{float(dh.quantile(0.99)):.2f}, max = {float(dh.max()):.1f}")

sig_all = torch.cat([b.attn.sigma.flatten() for lv in model.levels for b in lv])
check("softplus keeps the distance bias negative at every head",
      bool((F.softplus(sig_all) > 0).all()),
      f"{sig_all.numel()} heads, softplus(sigma) in "
      f"[{float(F.softplus(sig_all).min()):.3f}, {float(F.softplus(sig_all).max()):.3f}]")
check("the padded-key penalty is finite (MPS-safe)", np.isfinite(NEG), f"NEG = {NEG:.0e}")

# ------------------------------------------------------------ 3. bf16 --------
print("\n3. bfloat16 autocast")
model.train()
model.zero_grad(set_to_none=True)
with torch.autocast("cpu", dtype=torch.bfloat16):
    out16 = model(x, v=v, mask=mask, points=pts)
    loss16 = F.cross_entropy(out16.float(), y)
loss16.backward()
bad16 = [n_ for n_, p_ in trainable if p_.grad is not None and not torch.isfinite(p_.grad).all()]
check("bf16 forward is finite", bool(torch.isfinite(out16).all()),
      f"dtype {out16.dtype}, loss {float(loss16):.4f}")
check("bf16 backward produces no non-finite gradient", not bad16,
      f"{len(bad16)} non-finite of {len(trainable)}")

model.eval()
with torch.no_grad():
    f32 = model(x, v=v, mask=mask, points=pts)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        b16 = model(x, v=v, mask=mask, points=pts)
rel = float((f32 - b16.float()).abs().max() / f32.abs().max())
check("bf16 tracks fp32 to bf16 precision", rel < 0.15,
      f"max relative difference = {rel:.3f} (bf16 has ~3 decimal digits)")

# ------------------------------------------------------ 4. degenerate batch --
print("\n4. degenerate inputs")
model.train()
model.zero_grad(set_to_none=True)
dmask = mask.clone()
dmask[0] = False                                       # empty jet
dmask[1, 0, 1:] = False                                # single-constituent jet
loss_d = F.cross_entropy(model(x, v=v, mask=dmask, points=pts), y)
loss_d.backward()
bad = [n_ for n_, p_ in trainable if p_.grad is not None and not torch.isfinite(p_.grad).all()]
check("an empty jet and a 1-particle jet give a finite loss",
      bool(torch.isfinite(loss_d)), f"loss = {float(loss_d):.4f}")
check("and finite gradients", not bad, f"{len(bad)} non-finite of {len(trainable)}")

# ------------------------------------------------------------ 5. it learns ---
print("\n5. optimisation actually descends")
torch.manual_seed(0)
m2 = ErwinParticleTransformerV2(input_dim=C, num_classes=10, seq_len=64)
m2.train(); m2.trimmer.enabled = False
opt = torch.optim.AdamW(m2.parameters(), lr=1e-3)
losses = []
for step in range(25):
    opt.zero_grad(set_to_none=True)
    l = F.cross_entropy(m2(x, v=v, mask=mask, points=pts), y)
    l.backward()
    torch.nn.utils.clip_grad_norm_(m2.parameters(), 1.0)
    opt.step()
    losses.append(float(l))
check("loss decreases over 25 steps on a fixed batch",
      losses[-1] < losses[0] - 0.3,
      f"{losses[0]:.3f} -> {losses[-1]:.3f} "
      f"(min {min(losses):.3f})")
check("no step produced a non-finite loss", all(np.isfinite(losses)))

print()
if failures:
    print(f"S7 FAILED: {len(failures)} check(s) - {', '.join(failures)}")
    sys.exit(1)
print("S7 PASSED")
