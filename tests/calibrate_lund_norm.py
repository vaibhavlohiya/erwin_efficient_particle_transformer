"""Measure LUND_FEATURE_NORM - the fixed input standardisation of `LundEmbed`.

Clusters real jets exactly as the model does (hardest `--seq-len` constituents,
`--max-splits` tokens kept, hardest kT first, no kT cut) and prints the mean and
std of [ln kT, ln z, ln deltaR] over the kept tokens, ready to paste into
networks/erwin_part/lund.py. Rerun whenever seq_len or max_splits change.

    python tests/calibrate_lund_norm.py [--jets 4000]
"""
import argparse
import importlib.util
import os

import numpy as np
import torch
import uproot

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "lund", os.path.join(HERE, "..", "networks", "erwin_part", "lund.py"))
lund = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lund)

ap = argparse.ArgumentParser()
ap.add_argument("--jets", type=int, default=4000)
ap.add_argument("--seq-len", type=int, default=64)
ap.add_argument("--max-splits", type=int, default=48)
args = ap.parse_args()

WORKSPACE = os.path.abspath(os.path.join(HERE, "..", ".."))
root = next(p for p in (
    os.path.join(WORKSPACE, "efficient_particle_transformer", "JetClass_example_100k.root"),
    os.path.join(WORKSPACE, "JetClass_example_100k.root")) if os.path.exists(p))
ev = uproot.open(root)["tree"].arrays(
    ["part_px", "part_py", "part_pz", "part_energy"], library="np", entry_stop=args.jets)

L = args.seq_len
p4 = torch.zeros(args.jets, L, 4, dtype=torch.float64)
valid = torch.zeros(args.jets, L, dtype=torch.bool)
for n in range(args.jets):
    p = np.stack([ev[b][n] for b in ("part_px", "part_py", "part_pz", "part_energy")], -1)
    p = p[np.argsort(-np.hypot(p[:, 0], p[:, 1]), kind="stable")][:L]
    p4[n, :len(p)] = torch.from_numpy(p.astype(np.float64))
    valid[n, :len(p)] = True

print(f"# {args.jets} jets, L={L}, max_splits={args.max_splits}")
for alg in lund.ALGORITHMS:
    feats, _, tv = lund.lund_splittings(p4, valid, alg=alg, max_splits=args.max_splits)
    f = feats[tv]
    mean = tuple(round(x, 3) for x in f.mean(0).tolist())
    std = tuple(round(x, 3) for x in f.std(0).tolist())
    print(f'    "{alg}": ({mean}, {std}),   # {len(f)} tokens, '
          f'{tv.sum(1).float().mean():.1f}/jet')
