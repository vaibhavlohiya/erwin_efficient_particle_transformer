"""Pre-flight check: does this plugin build, run and stay finite on this node?

Runs in seconds, before the dataset is touched, so a broken checkout or a
numerics fault fails the job immediately instead of after the loader has spent
twenty minutes opening ROOT files.

Three things are checked, in increasing order of what they would have caught:

1.  Versions. torch comes from the image, not requirements.txt, so the exact
    build is recorded here and in the run manifest rather than assumed.
2.  The weaver plugin contract - get_model() against the real data config.
3.  A forward AND backward pass under autocast on a deliberately awkward batch:
    jets with 3, 8, 39 and 128 constituents. The sparse ones matter. Padding is
    pushed to the end of the ball-tree order, so a 3-constituent jet at L=64
    leaves 7 of 8 level-0 balls entirely invalid, and a masked-key sentinel that
    overflows fp16 turns those rows into NaN. Constructing the model does not
    exercise that; one backward pass does.
"""
import importlib.util
import os
import sys

import torch

NETWORK = os.environ.get("NETWORK_CONFIG", "networks/example_ErwinParTv2.py")
DATA_CONFIG = os.environ.get("DATA_CONFIG", "data/JetClass/JetClass_full.yaml")

print("--- preflight -----------------------------------------------------")
print(f"torch        {torch.__version__}  (cuda {torch.version.cuda}, "
      f"available={torch.cuda.is_available()})")
try:
    import weaver
    print(f"weaver-core  {getattr(weaver, '__version__', '?')}  at {os.path.dirname(weaver.__file__)}")
except Exception as exc:                                    # noqa: BLE001
    sys.exit(f"FATAL: weaver-core not importable: {exc}")

from weaver.utils.data.config import DataConfig             # noqa: E402

dc = DataConfig.load(DATA_CONFIG, load_observers=False)
spec = importlib.util.spec_from_file_location("_plugin", NETWORK)
plugin = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plugin)

use_amp = torch.cuda.is_available()
model, _ = plugin.get_model(dc, **({"use_amp": True} if use_amp else {}))
n_par = sum(p.numel() for p in model.parameters())
print(f"plugin OK    {NETWORK} -> {n_par / 1e6:.2f}M params, {len(dc.label_value)} classes")

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(dev).train()

N, P = 4, 128
C = len(dc.input_dicts["pf_features"])
mult = torch.tensor([3, 8, 39, 128])
mask = (torch.arange(P)[None, :] < mult[:, None]).unsqueeze(1).to(dev)
x = (torch.randn(N, C, P, device=dev) * mask)
pt = torch.rand(N, P, device=dev) * 100 + 1
eta = torch.randn(N, P, device=dev) * 0.3
phi = torch.randn(N, P, device=dev) * 0.3
px, py, pz = pt * torch.cos(phi), pt * torch.sin(phi), pt * torch.sinh(eta)
v = torch.stack([px, py, pz, torch.sqrt(px ** 2 + py ** 2 + pz ** 2)], 1) * mask
points = torch.stack([eta, phi], 1) * mask

with torch.autocast("cuda", enabled=use_amp):
    out = model(points, x, v, mask)
if not torch.isfinite(out).all():
    sys.exit("FATAL: non-finite logits on the sparse-jet probe batch")

out.float().sum().backward()
bad = [n for n, p in model.named_parameters()
       if p.grad is not None and not torch.isfinite(p.grad).all()]
if bad:
    sys.exit(f"FATAL: non-finite gradients in {len(bad)} tensors, e.g. {bad[:3]}")

print(f"fwd/bwd OK   amp={use_amp} on {dev}, multiplicities {mult.tolist()}, "
      f"logits finite, {sum(1 for _ in model.parameters())} tensors with finite grads")
print("-------------------------------------------------------------------")
