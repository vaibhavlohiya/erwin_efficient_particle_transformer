"""Batch-1 cost of the cluster models: the measurements behind ideas #4, #5, #7.

Each model is built by its own networks/example_*.py get_model() against
data/JetClass/JetClass_full.yaml (as weaver builds it, see
tests/flops_cluster_models.py), in eval mode, fp32, on CPU with ONE thread -
the setting closest to a fixed pipeline. Absolute numbers are this CPU's;
what transfers to an FPGA is the ratios and how each cost scales with N.

  scaling   synthetic jets with exactly N particles, N = 8..128:
            FLOPs (FlopCounterMode) and peak live activation memory
  memory    the activation peak split by top-level component, plus weights
  latency   batch-1 wall time for ~400 real jets spread over multiplicity
            (minimum of 7 runs each; 15 on the synthetic grid), so the spread
            with N is visible

Peak activation memory is tracked with a TorchDispatchMode: every tensor an op
creates is counted from creation until its last view is freed (storages, so
views are not double counted); parameters, buffers and the inputs are excluded.
At the peak it records which component created each live tensor. It counts
what PyTorch allocates, so it is an upper bound on what an FPGA dataflow design
would need to buffer, and it scales with bit width (fp32 here; 16-bit halves it).

Run from the weaver conda env:
    python tests/fpga_efficiency.py        # -> s10_results/fpga_efficiency.json
"""
import os
import statistics
import sys
import time
import warnings
import weakref
from collections import defaultdict

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten
from torch.utils.flop_counter import FlopCounterMode

warnings.filterwarnings("ignore")
HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
from flops_cluster_models import MODELS, ROOT, build  # noqa: E402
from jetclass_data import find_root, load_all  # noqa: E402
from weaver.utils.data.config import DataConfig  # noqa: E402

P = 128                                   # the data config pads every jet to 128
N_GRID = [8, 16, 24, 32, 40, 48, 64, 80, 96, 112, 128]
COMPONENTS = {   # top-level child -> component; anything outside them is "other"
    "embed": "embedding", "lund_embed": "Lund tokens",
    "pair_embed": "pair features", "pair_embeds": "pair features",
    "blocks": "particle attention", "levels": "particle attention",
    "pools": "pooling", "cls_blocks": "class attention + head",
    "norm": "class attention + head", "fc": "class attention + head",
}
OTHER = "geometry, tree & other"


class LiveMemory(TorchDispatchMode):
    """Peak bytes of live op outputs, attributed to the component that made them."""

    def __init__(self, exclude_ptrs, component):
        super().__init__()
        self.exclude, self.component = exclude_ptrs, component
        self.refs = {}                      # storage ptr -> [views alive, bytes, component]
        self.live, self.peak = 0, 0
        self.by_comp, self.peak_by_comp = defaultdict(int), {}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        for t in tree_flatten(out)[0]:
            if isinstance(t, torch.Tensor):
                self._track(t)
        return out

    def _track(self, t):
        st = t.untyped_storage()
        key = st.data_ptr()
        if key == 0 or key in self.exclude:
            return
        rec = self.refs.get(key)
        if rec is None:
            rec = self.refs[key] = [0, st.nbytes(), self.component()]
            self.live += rec[1]
            self.by_comp[rec[2]] += rec[1]
            if self.live > self.peak:
                self.peak, self.peak_by_comp = self.live, dict(self.by_comp)
        rec[0] += 1
        weakref.finalize(t, self._release, key)

    def _release(self, key):
        rec = self.refs.get(key)
        if rec is None:
            return
        rec[0] -= 1
        if rec[0] == 0:
            self.live -= rec[1]
            self.by_comp[rec[2]] -= rec[1]
            del self.refs[key]


def component_tracker(model):
    inner = getattr(model, "mod", model)
    stack = []

    def leave(m, a, o):          # a forward hook that returns non-None replaces the output
        stack.pop()

    def called(mod):
        """A ModuleList/ModuleDict is never called itself, only its items (which
        may be containers again, as v2's per-level block lists are): descend."""
        if isinstance(mod, (torch.nn.ModuleList, torch.nn.ModuleDict)):
            return [m for c in mod.children() for m in called(c)]
        return [mod]

    for name, child in inner.named_children():
        comp = COMPONENTS.get(name, OTHER)
        for t in called(child):
            t.register_forward_pre_hook(lambda m, a, c=comp: stack.append(c) and None)
            t.register_forward_hook(leave)
    return lambda: stack[-1] if stack else OTHER


def synthetic_jet(n, C, seed=0):
    g = torch.Generator().manual_seed(seed)
    pts = torch.zeros(1, 2, P); fts = torch.zeros(1, C, P)
    vec = torch.zeros(1, 4, P); msk = torch.zeros(1, 1, P, dtype=torch.bool)
    pts[0, :, :n] = torch.randn(2, n, generator=g) * 0.3
    fts[0, :, :n] = torch.randn(C, n, generator=g)
    p3 = torch.randn(3, n, generator=g).abs() * 10 + 1
    vec[0, :3, :n] = p3
    vec[0, 3, :n] = p3.norm(dim=0) + 0.1
    msk[0, 0, :n] = True
    return pts, fts, vec, msk


def flops_and_memory(model, inp, comp):
    # FlopCounterMode's module hooks fail under no_grad, so count with grad enabled
    # (the parameters carry it). A grad-carrying INPUT breaks too, once weaver's
    # SequenceTrimmer starts trimming (its 4th call on). The memory peak is
    # measured under no_grad, as at inference.
    fc = FlopCounterMode(display=False)
    with fc:
        model(*inp)
    with torch.no_grad():
        exclude = {p.untyped_storage().data_ptr() for p in model.parameters()}
        exclude |= {b.untyped_storage().data_ptr() for b in model.buffers()}
        exclude |= {t.untyped_storage().data_ptr() for t in inp}
        lm = LiveMemory(exclude, comp)
        with lm:
            model(*inp)
    return fc.get_total_flops(), lm.peak, {k: v for k, v in lm.peak_by_comp.items() if v > 0}


def latency_ms(model, inp, reps=7):
    """Minimum over repeats: for a fixed workload, scheduling noise only ever adds
    time, so the minimum is the noise-robust estimate of the model's own cost."""
    with torch.no_grad():
        times = []
        for _ in range(reps):
            t = time.perf_counter()
            model(*inp)
            times.append(time.perf_counter() - t)
    return min(times) * 1e3


if __name__ == "__main__":
    # REUSE_LATENCY=1 keeps the latency of the existing JSON and re-measures only
    # FLOPs and memory (latency takes most of the run).
    reuse = None
    if os.environ.get("REUSE_LATENCY") == "1":
        import json
        reuse = json.load(open(os.path.join(HERE, "s10_results", "fpga_efficiency.json")))["models"]
    torch.set_num_threads(1)
    torch.manual_seed(0)
    dc = DataConfig.load(os.path.join(ROOT, "data/JetClass/JetClass_full.yaml"), load_observers=False)
    C = len(dc.input_dicts["pf_features"])
    pts, fts, vec, msk, y = load_all(find_root(os.path.dirname(ROOT)), seed=0)
    pts, fts, vec, msk = pts[..., :P], fts[..., :P], vec[..., :P], msk[..., :P]
    mult = msk.sum((1, 2))
    # ~25 real jets per multiplicity bin of width 8, so the tails are covered
    g = torch.Generator().manual_seed(0)
    pick = []
    for lo in range(0, 144, 8):
        idx = torch.nonzero((mult >= lo) & (mult < lo + 8)).flatten()
        pick += idx[torch.randperm(len(idx), generator=g)[:25]].tolist()
    print(f"latency sample: {len(pick)} real jets, multiplicities {int(mult[pick].min())}-{int(mult[pick].max())}")

    out = {"device": "cpu", "threads": 1, "dtype": "float32", "batch": 1, "torch": torch.__version__,
           "n_grid": N_GRID, "models": {}}
    for name, (path, kw, _) in MODELS.items():
        model = build(path, kw, dc).eval()
        comp = component_tracker(model)
        trimmer = getattr(model, "mod", model).trimmer
        # weaver's SequenceTrimmer leaves its first 5 calls untrimmed; get past them.
        for _ in range(6):
            latency_ms(model, synthetic_jet(8, C), reps=1)
        rec = {"params": sum(p.numel() for p in model.parameters()),
               "weight_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
               "scaling": [], "latency": []}
        for n in N_GRID:
            jet = synthetic_jet(n, C)
            f, peak, by = flops_and_memory(model, jet, comp)
            old = reuse and next(x for x in reuse[name]["scaling"] if x["n"] == n)
            r = {"n": n, "flops": f, "peak_bytes": peak, "peak_by_component": by,
                 "latency_ms": old["latency_ms"] if old else latency_ms(model, jet, reps=15)}
            # Fixed-size variant: trimming off, so every jet uses all 128 padded slots, as
            # in a hardware pipeline sized for the largest jet. (v2 pads to its 64 anyway.)
            trimmer.enabled = False
            fp, pp, byp = flops_and_memory(model, jet, comp)
            r.update(flops_fixed=fp, peak_bytes_fixed=pp, peak_by_component_fixed=byp,
                     latency_ms_fixed=old["latency_ms_fixed"] if old else latency_ms(model, jet, reps=15))
            trimmer.enabled = True
            rec["scaling"].append(r)
        warm = (pts[pick[0]:pick[0] + 1], fts[pick[0]:pick[0] + 1], vec[pick[0]:pick[0] + 1], msk[pick[0]:pick[0] + 1])
        for _ in range(3):
            latency_ms(model, warm, reps=1)
        if reuse:
            rec["latency"] = reuse[name]["latency"]
        for i in ([] if reuse else pick):
            inp = (pts[i:i + 1], fts[i:i + 1], vec[i:i + 1], msk[i:i + 1])
            rec["latency"].append({"n": int(mult[i]), "ms": latency_ms(model, inp)})
        lat = sorted(r["ms"] for r in rec["latency"])
        s = rec["scaling"]
        print(f"{name:<18} FLOPs N=32 {s[3]['flops'] / 1e6:7.1f}M  N=128 {s[-1]['flops'] / 1e6:7.1f}M | "
              f"peak act N=32 {s[3]['peak_bytes'] / 1e6:6.2f} MB  N=128 {s[-1]['peak_bytes'] / 1e6:6.2f} MB | "
              f"latency p50 {statistics.median(lat):6.2f} ms  p99 {lat[int(0.99 * len(lat))]:6.2f}  "
              f"min {lat[0]:6.2f}  max {lat[-1]:6.2f}", flush=True)
        out["models"][name] = rec
    import json
    path = os.path.join(HERE, "s10_results", "fpga_efficiency.json")
    json.dump(out, open(path, "w"), indent=1)
    print("wrote", path)
