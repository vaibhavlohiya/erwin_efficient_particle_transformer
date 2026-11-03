"""FLOPs of the models as the cluster trains them (weaver plugins, JetClass full).

tests/flops_compare.py measures matched toy configurations; this one builds each
model through its own networks/example_*.py get_model() against
data/JetClass/JetClass_full.yaml, exactly as weaver does, and feeds a batch of
real jets (multiplicities from JetClass_example_100k.root, so ParT's
SequenceTrimmer cuts padding the way it does in training). Feature VALUES do not
change a FLOP count, only shapes and masks, so they are not standardised here.

  gflops_per_jet          eval forward, FlopCounterMode, divided by the batch, with
                          SequenceTrimmer OFF: every jet uses all 128 padded slots.
                          This is the deployed cost - weaver builds inference/ONNX
                          models with trimming disabled (enabled=trim and not
                          for_inference) - and that of a fixed-size hardware pipeline.
  gflops_per_jet_trimmed  trimming ON, after the 5 calls SequenceTrimmer leaves
                          untrimmed: padding cut to the batch's longest jet, as in
                          training
  train_pflops            3 x trimmed forward (forward + backward) x jets seen

FlopCounterMode counts matmul/conv/attention kernels, not elementwise ops, so
the Lund clustering (distance updates, no matmuls) is estimated separately as
an upper bound of 10 ops per pair per merge: 64 merges x 64^2 pairs.

Run from the weaver conda env:
    python tests/flops_cluster_models.py           # -> s10_results/cluster_flops.json
"""
import importlib.util
import json
import os
import sys
import warnings

import torch
from torch.utils.flop_counter import FlopCounterMode

warnings.filterwarnings("ignore")
HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
from jetclass_data import find_root, load_all  # noqa: E402
from weaver.utils.data.config import DataConfig  # noqa: E402

# name -> (network config, get_model kwargs, jets seen in training)
SEEN_10PCT_20EP = 20 * 1_024_000
MODELS = {
    "ParT":        ("networks/example_ParticleTransformer.py", {}, SEEN_10PCT_20EP),
    "ErwinParTv2": ("networks/example_ErwinParTv2.py", {}, SEEN_10PCT_20EP),
    "ErwinParTv2+Lund": ("networks/example_ErwinParTv2.py", {"use_lund": True}, SEEN_10PCT_20EP),
    "ErwinParT v1": ("networks/example_ErwinParticleTransformer.py", {}, None),
}
# Published JetClass baselines, counted the same way for the frontier plot only
# (tests/plot_fpga_frontier.py); not part of the cluster comparison.
REFERENCE_MODELS = {
    "PFN":         ("networks/example_PFN.py", {}, None),
    "P-CNN":       ("networks/example_PCNN.py", {}, None),
    "ParticleNet": ("networks/example_ParticleNet.py", {}, None),
}
LUND_ELEMENTWISE = 64 * 64 * 64 * 10        # merges x pairs x ops, upper bound


def alias_particle_transformer():
    """v1's network file imports `particle_transformer.networks...`, i.e. assumes
    the checkout is named that (as on the cluster image). Alias it via a symlink."""
    import tempfile
    d = tempfile.mkdtemp()
    os.symlink(ROOT, os.path.join(d, "particle_transformer"))
    sys.path.insert(0, d)


def build(path, kw, dc):
    if "ErwinParticleTransformer" in path:
        alias_particle_transformer()
    spec = importlib.util.spec_from_file_location("_plugin_" + os.path.basename(path), os.path.join(ROOT, path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model, _ = mod.get_model(dc, **kw)
    return model.eval()


if __name__ == "__main__":
    B = int(os.environ.get("BATCH", 512))
    torch.manual_seed(0)
    dc = DataConfig.load(os.path.join(ROOT, "data/JetClass/JetClass_full.yaml"), load_observers=False)
    C = len(dc.input_dicts["pf_features"])
    pts, _, vec, msk, y = load_all(find_root(os.path.dirname(ROOT)), seed=0)
    idx = torch.randperm(len(y))[:B]             # the file is class-sorted: sample across it
    P = 128
    pts, vec, msk = pts[idx, :, :P], vec[idx, :, :P], msk[idx, :, :P]
    fts = torch.randn(B, C, P) * msk
    mult = msk.sum((1, 2)).float()
    out = {"batch": B, "seq_padded_to": P, "multiplicity_mean": float(mult.mean()),
           "multiplicity_max_in_batch": int(mult.max()), "models": {}}
    out["reference_models"] = {}
    for name, (path, kw, seen) in {**MODELS, **REFERENCE_MODELS}.items():
        model = build(path, kw, dc)
        # PFN/P-CNN/ParticleNet have no SequenceTrimmer: their cost is fixed.
        trimmer = getattr(getattr(model, "mod", model), "trimmer", None) or type("NoTrim", (), {"enabled": True})()
        with torch.no_grad():
            for _ in range(6):                   # past SequenceTrimmer's 5 untrimmed calls
                model(pts[:2], fts[:2], vec[:2], msk[:2])

        def count():
            fc = FlopCounterMode(display=False)  # grad on (its hooks fail under no_grad)
            with fc:
                model(pts, fts, vec, msk)
            return fc.get_total_flops() / B

        per_jet_trim = count()
        trimmer.enabled = False
        per_jet_fixed = count()
        trimmer.enabled = True
        per_jet = per_jet_trim
        rec = {"network": path, "kwargs": kw, "params": sum(p.numel() for p in model.parameters()),
               "gflops_per_jet": per_jet_fixed / 1e9, "gflops_per_jet_trimmed": per_jet_trim / 1e9}
        if "use_lund" in kw:
            rec["lund_clustering_gflops_upper"] = LUND_ELEMENTWISE / 1e9
        if seen:
            rec["jets_seen"] = seen
            rec["train_pflops"] = 3 * per_jet * seen / 1e15
        (out["models"] if name in MODELS else out["reference_models"])[name] = rec
        print(f"{name:<18} {rec['params'] / 1e6:5.2f}M params  fixed {rec['gflops_per_jet']:.4f}"
              f"  trimmed {rec['gflops_per_jet_trimmed']:.4f} GFLOPs/jet"
              + (f"  {rec['train_pflops']:.1f} PFLOPs to train" if seen else ""), flush=True)
    path = os.path.join(HERE, "s10_results", "cluster_flops.json")
    json.dump(out, open(path, "w"), indent=1)
    print("wrote", path)
