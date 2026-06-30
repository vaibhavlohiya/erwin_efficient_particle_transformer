"""
Integration checks: Erwin ball attention + weaver class-attention (cls) layers.

Verifies that the ParticleErwinBlock stack feeds the CaiT-style cls readout
correctly: padded slots stay isolated, every real particle reaches the jet
logits of its own jet only, gradients flow back into every Erwin layer, and
an all-padded jet stays finite.

Run from the repo root with weaver-core 0.4.x installed:
    python test_cls_integration.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from networks.EfficientParticleTransformer import EfficientParticleTransformer

torch.manual_seed(0)

model = EfficientParticleTransformer(
    input_dim=7, num_classes=10, embed_dims=[128, 512, 128],
    num_heads=8, num_layers=4, num_cls_layers=2,
    block_params={'attn_type': 'erwin', 'ball_size': 16, 'ffn_ratio': 4},
    cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0},
    fc_params=[], trim=False, for_inference=False,
).eval()

N, P = 4, 60
x = torch.randn(N, 7, P)
v = torch.randn(N, 4, P)
mask = torch.ones(N, 1, P, dtype=torch.bool)
mask[0, 0, 40:] = False
mask[1, 0, 25:] = False
mask[2, 0, 55:] = False   # jet 3 stays full

with torch.no_grad():
    out = model(x, v=v, mask=mask)
assert out.shape == (N, 10) and torch.isfinite(out).all()
print(f"[1] forward through 4 erwin + 2 cls layers: out {tuple(out.shape)}, finite  OK")

# padded slots must not influence the cls readout
x2 = x.clone()
x2[0, :, 40:] += 100.0
x2[1, :, 25:] -= 50.0
v2 = v.clone()
v2[0, :, 40:] *= 10.0
with torch.no_grad():
    out2 = model(x2, v=v2, mask=mask)
assert torch.allclose(out, out2, atol=1e-5), (out - out2).abs().max()
print("[2] perturbing padded slots leaves all logits unchanged  OK")

# every real particle must reach the cls token (ball attn -> cls attn)
# (non-uniform noise: Embed's input LayerNorm cancels constant shifts)
x3 = x.clone()
x3[1, :, 0] += torch.randn(7) * 5.0   # one real particle in jet 1
with torch.no_grad():
    out3 = model(x3, v=v, mask=mask)
diff_per_jet = (out - out3).abs().max(dim=1).values
assert diff_per_jet[1] > 1e-4, "perturbed jet logits did not change"
assert diff_per_jet[[0, 2, 3]].max() < 1e-6, "leakage across jets"
print(f"[3] real particle reaches cls readout (jet1 dlogit={diff_per_jet[1]:.2e}), "
      f"no cross-jet leakage  OK")

# gradient path: loss -> cls blocks -> erwin blocks (sigma_att of every layer)
model.train()
out = model(x, v=v, mask=mask)
out.sum().backward()
grads = [b.block.BMSA.sigma_att.grad.abs().sum().item() for b in model.blocks]
cls_grad = model.cls_blocks[0].attn.in_proj_weight.grad.abs().sum().item()
assert all(g > 0 for g in grads) and cls_grad > 0, grads
print(f"[4] gradients flow: cls attn {cls_grad:.1e}; sigma_att per erwin layer "
      f"{['%.1e' % g for g in grads]}  OK")

# zero-particle jet edge case: cls token attends only to itself, no NaN
mask_z = mask.clone(); mask_z[3, 0, :] = False
model.eval()
with torch.no_grad():
    out_z = model(x, v=v, mask=mask_z)
assert torch.isfinite(out_z).all()
print("[5] all-padded jet: finite logits (cls attends to itself)  OK")

print("\nAll cls-integration checks passed.")
