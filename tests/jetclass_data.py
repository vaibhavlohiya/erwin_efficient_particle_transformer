"""Load JetClass jets in weaver's layout, with JetClass_full.yaml's preprocessing.

Reproduces the 17 `pf_features` of data/JetClass/JetClass_full.yaml exactly -
same derived variables, same (subtract, multiply) constants, same clipping - so
that anything measured here transfers to a real `weaver` run rather than being
an approximation of it.

Returns tensors shaped as weaver hands them to a network config:
    points   (N, 2, P)   (part_deta, part_dphi)
    features (N, 17, P)
    vectors  (N, 4, P)   (px, py, pz, energy)
    mask     (N, 1, P)   True = real constituent
    labels   (N,)        class index in [0, 10)

JetClass files are sorted by class, so `stratified=True` (the default) samples an
equal number per class; a contiguous read would otherwise give a single class.
Returned rows are shuffled, so a positional split (train/val, or a held-out tail)
is class-mixed rather than landing on whole classes.
"""
from __future__ import annotations

import os

import numpy as np
import torch

LABELS = ["label_QCD", "label_Hbb", "label_Hcc", "label_Hgg", "label_H4q",
          "label_Hqql", "label_Zqq", "label_Wqq", "label_Tbqq", "label_Tbl"]
CLASS_NAMES = ["QCD", "Hbb", "Hcc", "Hgg", "H4q", "Hqql", "Zqq", "Wqq", "Tbqq", "Tbl"]

# (name, subtract, multiply, clip_min, clip_max) - straight from JetClass_full.yaml
FEATURES = [
    ("part_pt_log", 1.7, 0.7, -5, 5),
    ("part_e_log", 2.0, 0.7, -5, 5),
    ("part_logptrel", -4.7, 0.7, -5, 5),
    ("part_logerel", -4.7, 0.7, -5, 5),
    ("part_deltaR", 0.2, 4.0, -5, 5),
    ("part_charge", 0, 1, -5, 5),
    ("part_isChargedHadron", 0, 1, -5, 5),
    ("part_isNeutralHadron", 0, 1, -5, 5),
    ("part_isPhoton", 0, 1, -5, 5),
    ("part_isElectron", 0, 1, -5, 5),
    ("part_isMuon", 0, 1, -5, 5),
    ("part_d0", 0, 1, -5, 5),
    ("part_d0err", 0, 1, 0, 1),
    ("part_dz", 0, 1, -5, 5),
    ("part_dzerr", 0, 1, 0, 1),
    ("part_deta", 0, 1, -5, 5),
    ("part_dphi", 0, 1, -5, 5),
]
NUM_FEATURES = len(FEATURES)

_BRANCHES = ["part_px", "part_py", "part_pz", "part_energy", "part_deta", "part_dphi",
             "part_charge", "part_isChargedHadron", "part_isNeutralHadron",
             "part_isPhoton", "part_isElectron", "part_isMuon",
             "part_d0val", "part_d0err", "part_dzval", "part_dzerr",
             "jet_pt", "jet_energy"]


def find_root(workspace: str) -> str:
    for p in (os.path.join(workspace, "efficient_particle_transformer",
                           "JetClass_example_100k.root"),
              os.path.join(workspace, "JetClass_example_100k.root")):
        if os.path.exists(p):
            return p
    raise FileNotFoundError("JetClass_example_100k.root not found")


def _derived(ev, i):
    px, py, pz, e = (ev[k][i].astype(np.float64)
                     for k in ("part_px", "part_py", "part_pz", "part_energy"))
    pt = np.hypot(px, py)
    deta, dphi = ev["part_deta"][i], ev["part_dphi"][i]
    return {
        "part_pt_log": np.log(np.clip(pt, 1e-8, None)),
        "part_e_log": np.log(np.clip(e, 1e-8, None)),
        "part_logptrel": np.log(np.clip(pt / max(ev["jet_pt"][i], 1e-8), 1e-8, None)),
        "part_logerel": np.log(np.clip(e / max(ev["jet_energy"][i], 1e-8), 1e-8, None)),
        "part_deltaR": np.hypot(deta, dphi),
        "part_charge": ev["part_charge"][i],
        "part_isChargedHadron": ev["part_isChargedHadron"][i],
        "part_isNeutralHadron": ev["part_isNeutralHadron"][i],
        "part_isPhoton": ev["part_isPhoton"][i],
        "part_isElectron": ev["part_isElectron"][i],
        "part_isMuon": ev["part_isMuon"][i],
        "part_d0": np.tanh(ev["part_d0val"][i]),
        "part_d0err": ev["part_d0err"][i],
        "part_dz": np.tanh(ev["part_dzval"][i]),
        "part_dzerr": ev["part_dzerr"][i],
        "part_deta": deta,
        "part_dphi": dphi,
    }, np.stack([px, py, pz, e])


def load_jets(root: str, n: int, stratified: bool = True, seed: int = 0,
              max_particles: int | None = None):
    import uproot

    tree = uproot.open(root)["tree"]
    lab = np.stack([tree[l].array(library="np") for l in LABELS], axis=1).argmax(1)
    rng = np.random.default_rng(seed)

    if stratified:
        per = max(1, n // len(LABELS))
        idx = np.concatenate([rng.choice(np.flatnonzero(lab == c), per, replace=False)
                              for c in range(len(LABELS))])
    else:
        idx = rng.choice(len(lab), n, replace=False)
    rng.shuffle(idx)
    idx = np.sort(idx[:n])

    ev = tree.arrays(_BRANCHES, library="np",
                     entry_start=int(idx.min()), entry_stop=int(idx.max()) + 1)
    local = idx - int(idx.min())

    P = max(len(ev["part_px"][i]) for i in local)
    if max_particles:
        P = min(P, max_particles)
    N = len(local)

    points = torch.zeros(N, 2, P)
    feats = torch.zeros(N, NUM_FEATURES, P)
    vectors = torch.zeros(N, 4, P)
    mask = torch.zeros(N, 1, P, dtype=torch.bool)
    labels = torch.from_numpy(lab[idx].astype(np.int64))

    for row, i in enumerate(local):
        d, p4 = _derived(ev, i)
        k = min(len(ev["part_px"][i]), P)
        for c, (name, sub, mul, lo, hi) in enumerate(FEATURES):
            feats[row, c, :k] = torch.from_numpy(
                np.clip((np.asarray(d[name], dtype=np.float64)[:k] - sub) * mul, lo, hi
                        ).astype(np.float32))
        vectors[row, :, :k] = torch.from_numpy(p4[:, :k].astype(np.float32))
        points[row, 0, :k] = torch.from_numpy(np.asarray(d["part_deta"])[:k].astype(np.float32))
        points[row, 1, :k] = torch.from_numpy(np.asarray(d["part_dphi"])[:k].astype(np.float32))
        mask[row, 0, :k] = True

    # The entry indices were sorted for an efficient contiguous uproot read, which
    # puts the rows back in class order (JetClass files are sorted by class). Undo
    # that before returning, or any positional split - train/val, or a held-out
    # tail - lands on whole classes the other side never sees.
    out = torch.from_numpy(rng.permutation(N))
    return (points[out], feats[out], vectors[out], mask[out], labels[out])


def load_all(root: str, max_particles: int = 128, seed: int = 0,
             entry_stop: int | None = None):
    """Whole-file load, vectorised through awkward - for S10-scale training.

    `load_jets` builds tensors with a per-jet Python loop, which is fine for a
    few hundred jets and far too slow for 100k. This pads with `ak.pad_none` and
    converts in one shot per branch instead.

    Rows are shuffled (JetClass files are class-sorted), so a positional
    train/val/test split is class-mixed. Tensors stay on CPU; move batches to the
    device in the training loop rather than holding ~1.2 GB on an accelerator.
    """
    import awkward as ak
    import uproot

    tree = uproot.open(root)["tree"]
    arr = tree.arrays(_BRANCHES + LABELS, library="ak", entry_stop=entry_stop)
    P = min(max_particles, int(ak.max(ak.num(arr["part_px"]))))
    n_jets = len(arr["part_px"])

    def pad(field, fill=0.0):
        return ak.to_numpy(ak.fill_none(
            ak.pad_none(arr[field], P, clip=True), fill)).astype(np.float32)

    px, py, pz, e = (pad(k) for k in ("part_px", "part_py", "part_pz", "part_energy"))
    deta, dphi = pad("part_deta"), pad("part_dphi")
    nreal = ak.to_numpy(ak.num(arr["part_px"]))
    mask = np.arange(P)[None, :] < np.minimum(nreal, P)[:, None]

    jet_pt = np.asarray(arr["jet_pt"])[:, None].astype(np.float32)
    jet_e = np.asarray(arr["jet_energy"])[:, None].astype(np.float32)
    pt = np.hypot(px, py)
    safe = lambda a: np.clip(a, 1e-8, None)

    derived = {
        "part_pt_log": np.log(safe(pt)),
        "part_e_log": np.log(safe(e)),
        "part_logptrel": np.log(safe(pt / safe(jet_pt))),
        "part_logerel": np.log(safe(e / safe(jet_e))),
        "part_deltaR": np.hypot(deta, dphi),
        "part_charge": pad("part_charge"),
        "part_isChargedHadron": pad("part_isChargedHadron"),
        "part_isNeutralHadron": pad("part_isNeutralHadron"),
        "part_isPhoton": pad("part_isPhoton"),
        "part_isElectron": pad("part_isElectron"),
        "part_isMuon": pad("part_isMuon"),
        "part_d0": np.tanh(pad("part_d0val")),
        "part_d0err": pad("part_d0err"),
        "part_dz": np.tanh(pad("part_dzval")),
        "part_dzerr": pad("part_dzerr"),
        "part_deta": deta,
        "part_dphi": dphi,
    }

    feats = np.zeros((n_jets, NUM_FEATURES, P), dtype=np.float32)
    for c, (name, sub, mul, lo, hi) in enumerate(FEATURES):
        feats[:, c] = np.clip((derived[name] - sub) * mul, lo, hi)
    feats *= mask[:, None, :]                      # padded slots carry nothing

    vectors = np.stack([px, py, pz, e], axis=1) * mask[:, None, :]
    points = np.stack([deta, dphi], axis=1) * mask[:, None, :]
    labels = np.stack([np.asarray(arr[l]) for l in LABELS], axis=1).argmax(1)

    order = np.random.default_rng(seed).permutation(n_jets)
    return (torch.from_numpy(points[order]), torch.from_numpy(feats[order]),
            torch.from_numpy(vectors[order]),
            torch.from_numpy(mask[order][:, None, :]),
            torch.from_numpy(labels[order].astype(np.int64)))
