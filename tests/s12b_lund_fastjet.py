"""S12b - Lund splittings: our batched torch clustering vs FastJet.

`lund.lund_splittings` reclusters every jet in a batch at once with a masked
argmin per step. FastJet is the reference implementation of the same
sequential-recombination algorithms, so the two must produce the same tree. This
compares the full set of splittings - every merge, as (ln kT, ln z, ln deltaR) -
jet by jet, for both kT and C/A, on the same 64 hardest constituents the model
keeps.

FastJet's tree is walked from the root with `has_parents`, the textbook Lund
declustering; ours records merges as they happen. Agreement therefore also
confirms that the order of the walk does not change the token set.

Needs `fastjet` (pip install fastjet) and torch, and no weaver - `lund.py` is
torch-only for exactly this reason. Run with an interpreter that has fastjet:
    LUND_ORACLE_PY=/path/to/python tests/run_all.sh
    /path/to/python tests/s12b_lund_fastjet.py
"""
import importlib.util
import os
import sys

import numpy as np
import torch

try:
    import fastjet as fj
except ImportError:
    print("  [SKIP] fastjet not installed in this interpreter - "
          "`pip install fastjet` and rerun with it")
    sys.exit(0)

HERE = os.path.dirname(os.path.abspath(__file__))
# Load lund.py by path: importing the erwin_part package is fine (it is lazy),
# but going through it would still put networks/ on the path for nothing.
spec = importlib.util.spec_from_file_location(
    "lund", os.path.join(HERE, "..", "networks", "erwin_part", "lund.py"))
lund = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lund)

WORKSPACE = os.path.abspath(os.path.join(HERE, "..", ".."))
L, NJETS = 64, 300
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
ev = uproot.open(root)["tree"].arrays(
    ["part_px", "part_py", "part_pz", "part_energy"], library="np", entry_stop=NJETS)

# The model's slot layout: hardest L constituents, zero-padded.
p4 = torch.zeros(NJETS, L, 4, dtype=torch.float64)
valid = torch.zeros(NJETS, L, dtype=torch.bool)
for n in range(NJETS):
    p = np.stack([ev[b][n] for b in ("part_px", "part_py", "part_pz", "part_energy")], -1)
    p = p[np.argsort(-np.hypot(p[:, 0], p[:, 1]), kind="stable")][:L]
    p4[n, :len(p)] = torch.from_numpy(p.astype(np.float64))
    valid[n, :len(p)] = True


def kin(j, native):
    """pt, rapidity, phi of a PseudoJet.

    `native` uses FastJet's own rapidity, which treats a slightly tachyonic
    input (m^2 < 0 from float rounding in the stored 4-vectors - JetClass has
    them) as massless. Otherwise use weaver's `to_ptrapphim` formula, which
    `lund.py` matches on purpose so splitting tokens agree with the pair
    features. The two differ at the 1e-7 level in rapidity, which ln(deltaR)
    amplifies for near-collinear pairs.
    """
    if native:
        return j.pt(), j.rap(), j.phi()
    e, pz = j.e(), j.pz()
    return j.pt(), 0.5 * np.log(1 + 2 * pz / max(e - pz, 1e-20)), j.phi()


def fastjet_splittings(parts, algorithm, native=False):
    """Every splitting of the full tree, by recursive declustering from the root."""
    cs = fj.ClusterSequence([fj.PseudoJet(*map(float, x)) for x in parts],
                            fj.JetDefinition(algorithm, 1000.0))
    jets = cs.inclusive_jets()
    out, stack = [], list(jets)
    while stack:
        j = stack.pop()
        a, b = fj.PseudoJet(), fj.PseudoJet()
        if not j.has_parents(a, b):
            continue
        if b.pt() > a.pt():
            a, b = b, a
        pta, ya, phia = kin(a, native)
        ptb, yb, phib = kin(b, native)
        dphi = (phia - phib + np.pi) % (2 * np.pi) - np.pi
        delta = np.hypot(ya - yb, dphi)
        out.append((np.log(max(ptb * delta, 1e-8)),
                    np.log(max(ptb / (pta + ptb), 1e-8)),
                    np.log(max(delta, 1e-8))))
        stack += [a, b]
    return len(jets), np.array(sorted(out, key=lambda t: -t[0]))


for name, algorithm in (("kt", fj.kt_algorithm), ("ca", fj.cambridge_algorithm)):
    feats, _, tv = lund.lund_splittings(p4, valid, alg=name, max_splits=None)
    n_root, n_count, worst, bad, native_worst = 0, 0, 0.0, [], 0.0
    for n in range(NJETS):
        k = int(valid[n].sum())
        roots, ref = fastjet_splittings(p4[n, :k].numpy(), algorithm)
        n_root += roots != 1
        ours = feats[n][tv[n]].numpy()
        if len(ours) != len(ref) or len(ref) != k - 1:
            n_count += 1
            continue
        err = np.abs(ours - ref).max() if len(ref) else 0.0
        worst = max(worst, err)
        if err > 1e-6:
            bad.append(n)
        _, ref_native = fastjet_splittings(p4[n, :k].numpy(), algorithm, native=True)
        if len(ref_native):
            native_worst = max(native_worst, np.abs(ours - ref_native).max())
    check(f"{name}: FastJet with R=1000 leaves one jet (no beam merges)", n_root == 0,
          f"{n_root}/{NJETS} jets split")
    check(f"{name}: k-1 splittings for k constituents, same count as FastJet",
          n_count == 0, f"{n_count}/{NJETS} mismatched")
    check(f"{name}: splitting set identical to FastJet on {NJETS} real jets",
          not bad, f"max |diff| {worst:.2e}, {len(bad)} jets differ")
    check(f"{name}: within 1e-3 of FastJet's own rapidity convention", native_worst < 1e-3,
          f"max |diff| {native_worst:.2e}")

# Near-ties in d_ij reorder merges in float32, which is why the model clusters in
# float64 wherever the device supports it. This only bounds how often it happens.
feats32, _, tv32 = lund.lund_splittings(p4.float(), valid, alg="kt", max_splits=None)
feats64, _, tv64 = lund.lund_splittings(p4, valid, alg="kt", max_splits=None)
same = sum(int(torch.allclose(feats32[n][tv32[n]].double(), feats64[n][tv64[n]],
                              atol=1e-3)) for n in range(NJETS))
check("kt: float32 clustering reproduces float64 on >= 95% of jets",
      same >= 0.95 * NJETS, f"{same}/{NJETS}")

print()
print("S12b FAILED: " + ", ".join(failures) if failures else "S12b passed")
sys.exit(1 if failures else 0)
