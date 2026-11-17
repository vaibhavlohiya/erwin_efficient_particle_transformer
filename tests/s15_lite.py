"""S15 - ErwinParTLite, the fixed-size FPGA-friendly ErwinParTv2.

  1. trace     torch.fx traces every switch combination (bar the 'tree' reference
               layout) and the traced graph reproduces the eager model
  2. train     forward finite, every parameter gets gradient, for each layout
  3. isolation each jet scored alone matches the batch; padding content and the
               order of real particles do not change the logits
  4. tiles     Morton tiles are spatial: smaller within-tile dR than pT-order tiles
  5. summary   with ball summaries a change in one tile reaches every other tile
               in the first block; without them it stays in its own tile; the
               summary keys of empty tiles are masked
  6. lut       table logarithms track torch.log; LUT pair features track ParT's
               exact ones (ln m^2 uses the small-angle form, so it is looser)
  7. norm      the masked BatchNorm's statistics ignore padded rows
  8. size      D = 128 matches ErwinParTv2's parameter count; the D = 64 / 32
               configurations land near the HLT / L1 budgets

Run from the weaver conda env:
    python tests/s15_lite.py
"""
import os
import sys
import warnings

import numpy as np
import torch
import torch.fx as fx

warnings.filterwarnings("ignore")
HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "networks"))
from erwin_part.fpga import ErwinParTLite, LiteConfig  # noqa: E402
from erwin_part.fpga.approx import _log2_table, exact_features, lut_features, lut_log, lut_pt  # noqa: E402
from erwin_part.fpga.norms import MaskedBatchNorm  # noqa: E402
from erwin_part.fpga.partition import gather_slots, morton_order  # noqa: E402
from erwin_part.model import ErwinParticleTransformerV2  # noqa: E402
from jetclass_data import NUM_FEATURES, find_root, load_all  # noqa: E402

torch.manual_seed(0)
fails = []


def check(name, ok, detail=""):
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
    if not ok:
        fails.append(name)


def run(m, b):
    pts, fts, vec, msk = b
    return m(fts, v=vec, mask=msk, points=pts)


WS = os.path.abspath(os.path.join(HERE, "..", ".."))
pts, fts, vec, msk, y = load_all(find_root(WS), seed=0)
B = 24
batch = tuple(t[:B, ..., :128] for t in (pts, fts, vec, msk))
real_n = batch[3].sum((1, 2))
C = NUM_FEATURES
FPGA = dict(D=32, ffn_ratio=2, pair_hidden=(16, 16), pair_features="lut", norm="affine",
            ball_summary="physics")

# ------------------------------------------------------------------ trace --
print("1. trace")
configs = [dict(), dict(ball_summary="physics"), dict(ball_summary="plain"),
           dict(pair_features="lut", norm="batchnorm"), dict(norm="none", pair_features="none"),
           dict(partition="pt", hierarchy=False, depths=(6,), ball_summary="plain"), FPGA]
for kw in configs:
    m = ErwinParTLite(LiteConfig(input_dim=C, **kw)).eval()
    try:
        g = fx.symbolic_trace(m)
        with torch.no_grad():
            diff = float((run(g, batch) - run(m, batch)).abs().max())
        check(f"traces and matches eager: {kw or 'default'}", diff < 1e-5, f"max |diff| {diff:.1e}")
    except Exception as e:                                   # noqa: BLE001
        check(f"traces: {kw or 'default'}", False, f"{type(e).__name__}: {str(e)[:80]}")

# ------------------------------------------------------------------ train --
print("\n2. train")
for kw in [dict(), dict(partition="tree"), dict(partition="pt", hierarchy=False, depths=(6,), ball_summary="plain"),
           dict(norm="batchnorm", pair_features="lut", ball_summary="physics"), FPGA]:
    m = ErwinParTLite(LiteConfig(input_dim=C, **kw)).train()
    out = run(m, batch)
    out.sum().backward()
    dead = [k for k, p in m.named_parameters() if p.requires_grad and (p.grad is None or not p.grad.abs().sum())]
    check(f"finite, every parameter gets gradient: {kw or 'default'}",
          bool(torch.isfinite(out).all()) and not dead, f"no gradient: {dead[:4]}" if dead else "")

# -------------------------------------------------------------- isolation --
print("\n3. isolation (eval)")
for kw in [dict(ball_summary="physics"), FPGA]:
    m = ErwinParTLite(LiteConfig(input_dim=C, **kw)).eval()
    with torch.no_grad():
        full = run(m, batch)
        alone = torch.cat([run(m, tuple(t[i:i + 1] for t in batch)) for i in range(B)])
        noisy = list(batch)
        pad = ~batch[3].bool()
        noisy[1] = batch[1] + torch.randn_like(batch[1]) * pad      # garbage in padded feature slots
        sh = torch.arange(128).expand(B, 128).clone()
        for i in range(B):
            k = int(real_n[i])
            sh[i, :k] = torch.randperm(k)
        perm = tuple(torch.gather(t, 2, sh.unsqueeze(1).expand(-1, t.shape[1], -1)) for t in batch)
        d_alone = float((alone - full).abs().max())
        d_pad = float((run(m, tuple(noisy)) - full).abs().max())
        d_perm = float((run(m, perm) - full).abs().max())
    tag = "physics" if kw.get("D") is None else "FPGA candidate"
    check(f"{tag}: each jet alone matches the batch", d_alone < 1e-4, f"{d_alone:.1e}")
    check(f"{tag}: padding content does not change the logits", d_pad < 1e-5, f"{d_pad:.1e}")
    check(f"{tag}: order of real particles does not change the logits", d_perm < 1e-4, f"{d_perm:.1e}")

# ------------------------------------------------------------------ tiles --
print("\n4. tiles")
N = 400
p4 = vec[:N, :, :128].permute(0, 2, 1)
pos = pts[:N, :, :128].permute(0, 2, 1)
valid = msk[:N, 0, :128].bool()
pt = (p4[..., 0] ** 2 + p4[..., 1] ** 2).sqrt()
top = torch.argsort(torch.where(valid, pt, -torch.ones_like(pt)), dim=1, descending=True, stable=True)[:, :64]
pos64, v64 = gather_slots(pos, top), torch.gather(valid, 1, top)


def tile_dr(order):
    p = gather_slots(pos64, order).view(N, 8, 8, 2)
    vv = torch.gather(v64, 1, order).view(N, 8, 8)
    d = (p.unsqueeze(3) - p.unsqueeze(2)).norm(dim=-1)
    pv = vv.unsqueeze(3) & vv.unsqueeze(2) & ~torch.eye(8, dtype=torch.bool)
    return float(d[pv].mean())


ident = torch.arange(64).expand(N, 64)
d_pt, d_mo = tile_dr(ident), tile_dr(morton_order(pos64, v64))
check("Morton tiles are spatially tighter than pT-order tiles", d_mo < 0.7 * d_pt,
      f"mean within-tile dR {d_mo:.3f} vs {d_pt:.3f}")

# ---------------------------------------------------------------- summary --
print("\n5. ball summaries")
for mode, expect_spread in (("off", False), ("physics", True)):
    m = ErwinParTLite(LiteConfig(input_dim=C, ball_summary=mode)).eval()
    caught = []
    hook = m.levels[0][0].register_forward_hook(lambda mod, a, o: caught.append(o))
    one = tuple(t[:1] for t in batch)                       # a jet with 64+ particles
    j = int(torch.nonzero(one[3][0, 0])[0])                 # its hardest particle
    with torch.no_grad():
        run(m, one)
        bumped = list(one)
        bumped[1] = one[1].clone()
        # A random direction: a constant shift of all features would be erased by
        # the embedding's LayerNorm. Features only, so the tiles stay the same.
        bumped[1][0, :, j] += torch.randn(one[1].shape[1], generator=torch.Generator().manual_seed(1))
        run(m, tuple(bumped))
    hook.remove()
    occupied = caught[0][0].abs().amax((-1, -2)) > 0                 # empty tiles are masked to 0
    delta = (caught[0] - caught[1])[0].abs().amax((-1, -2))
    # Absolute threshold well above float noise (~1e-7): with summaries off the
    # other tiles have no path at all and change by exactly 0.
    changed = int((delta[occupied] > 1e-5).sum())
    n_occ = int(occupied.sum())
    ok = changed == n_occ if expect_spread else changed == 1
    check(f"summaries {mode}: a change in one tile reaches {'every occupied tile' if expect_spread else 'only its own tile'}",
          ok, f"{changed} of {n_occ} occupied tiles changed; per tile "
              f"{[f'{x:.0e}' for x in delta[occupied].tolist()]}")
m = ErwinParTLite(LiteConfig(input_dim=C, ball_summary="physics")).eval()
few = tuple(t[:1].clone() for t in batch)
few[3][0, 0, 5:] = False                                    # a 5-particle jet: 7 empty tiles
with torch.no_grad():
    v5 = torch.zeros(1, 64, dtype=torch.bool); v5[0, :5] = True
    ctx = m._context(0, torch.randn(1, 64, 2) * 0.1, torch.rand(1, 64, 4) + 1, v5)
kv = ctx["key_valid"]                                       # (1, nb, m + nb)
check("summary keys of empty tiles are masked", bool((kv[0, :, 8:].sum(-1) == 1).all()),
      f"valid summary keys per tile: {kv[0, :, 8:].sum(-1).tolist()}")

# -------------------------------------------------------------------- lut --
print("\n6. lookup tables")
tab = _log2_table(6)
x = torch.logspace(-6, 4, 20000)
err = float((lut_log(x, tab, 6) - torch.log(x)).abs().max())
check("table log within 1e-3 of torch.log over 1e-6..1e4", err < 1e-3, f"max |err| {err:.1e}")
pp = gather_slots(p4, top)[:200].view(200, 8, 8, 4)
po = pos64[:200].view(200, 8, 8, 2)
vv = v64[:200].view(200, 8, 8)
pv = vv.unsqueeze(3) & vv.unsqueeze(2) & ~torch.eye(8, dtype=torch.bool)
ex = exact_features(pp.unsqueeze(3), pp.unsqueeze(2))[pv]
pti, lni = lut_pt(pp.unsqueeze(3), tab, 6)
ptj, lnj = lut_pt(pp.unsqueeze(2), tab, 6)
lu = lut_features(po.unsqueeze(3), po.unsqueeze(2), lni, lnj, pti, ptj, tab, 6,
                  p4_i=pp.unsqueeze(3), p4_j=pp.unsqueeze(2))[pv]
for i, (nm, tol) in enumerate([("ln kT", 0.1), ("ln z", 0.05), ("ln dR", 0.1), ("ln m^2", 0.05)]):
    q95 = float((lu[:, i] - ex[:, i]).abs().quantile(0.95))
    check(f"LUT {nm} vs ParT's exact: 95th percentile |diff| < {tol}", q95 < tol, f"{q95:.3f}")

# ------------------------------------------------------------------- norm --
print("\n7. masked BatchNorm")
bn1, bn2 = MaskedBatchNorm(8).train(), MaskedBatchNorm(8).train()
xr = torch.randn(4, 10, 8)
vm = torch.rand(4, 10) > 0.4
o1 = bn1(xr * vm.unsqueeze(-1), vm)
o2 = bn2(xr * vm.unsqueeze(-1) + 50 * torch.randn(4, 10, 8) * (~vm).unsqueeze(-1), vm)
check("statistics ignore padded rows", torch.allclose(bn1.running_mean, bn2.running_mean) and
      torch.allclose(o1[vm], o2[vm], atol=1e-5))

print("\n7b. self-pairs")
m = ErwinParTLite(LiteConfig(input_dim=C)).train()
pn = m.pair_nets[0]
seen = []
h = pn.in_norm.register_forward_hook(lambda mod, a, o: seen.append((a[0].detach(), a[1])))
with torch.no_grad():
    run(m, batch)
h.remove()
f0, st = seen[0]                                            # level-0 pair features, stats mask
diag = torch.eye(8, dtype=torch.bool).expand_as(st)
check("pair-norm statistics exclude a particle paired with itself",
      not bool((st & diag).any()) and bool(st.any()),
      f"ln dR of self-pairs {float(f0[..., 2][diag & (msk[:B, 0, :1] > 0).view(-1, 1, 1, 1)].max()):.1f}, "
      f"of real pairs median {float(f0[..., 2][st].median()):.1f}")

# ------------------------------------------------------------------- size --
print("\n8. size")
n_v2 = sum(p.numel() for p in ErwinParticleTransformerV2(input_dim=C, num_classes=10).parameters())
n = lambda kw: sum(p.numel() for p in ErwinParTLite(LiteConfig(input_dim=C, **kw)).parameters())
n128, n64, n32 = n({}), n(dict(D=64, ffn_ratio=2, pair_hidden=(16, 16))), n(dict(D=32, ffn_ratio=2, pair_hidden=(16, 16)))
check("D = 128 matches ErwinParTv2 within 1%", abs(n128 / n_v2 - 1) < 0.01, f"{n128:,} vs {n_v2:,}")
print(f"  D = 64, FFN 2, pair net 16: {n64:,} parameters;  D = 32: {n32:,}")

print()
print("S15 FAILED: " + ", ".join(fails) if fails else "S15 passed")
sys.exit(1 if fails else 0)
