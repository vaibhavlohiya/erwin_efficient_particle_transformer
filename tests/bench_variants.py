"""FLOPs and inference latency for the ErwinParTv2 variants S11 trained.

Builds each arm exactly as tests/s10_compare.py does (same `build()`), feeds it
real jets from the 100k sample, and writes tests/s10_results/variants_bench.json
for tests/plot_variants.py. Weights do not matter for either number, so nothing
is loaded from the _best.pt checkpoints.

  flops_per_jet   FlopCounterMode over one eval forward, divided by the batch.
                  It counts matmul/conv/attention kernels only, so the Lund
                  clustering (elementwise distance updates, no matmuls) adds ~0
                  here - its cost shows up in the latency instead.
  latency         median wall time of an eval forward, per batch and per jet,
                  after warm-up. Lund clustering runs on CPU in float64 even
                  on MPS, so time the device you will quote.

Run from the weaver conda env:
    python tests/bench_variants.py                    # CPU, batch 256
    python tests/bench_variants.py --device mps --batch 512
"""
import argparse
import json
import os
import statistics
import sys
import time
import warnings

import torch
from torch.utils.flop_counter import FlopCounterMode

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from jetclass_data import find_root, load_all  # noqa: E402
from s10_compare import WORKSPACE, OUTDIR, build, call  # noqa: E402

# name -> build() kwargs, matching the tags in tests/s11_lund.sh and s11_grow.sh.
VARIANTS = {
    "base":  {},
    "lund":  dict(use_lund=True, lund_where="bottleneck", lund_alg="kt"),
    "grow8": dict(layout=([8, 16, 32], [1, 1], [2, 2, 2]), readout="coarse"),
    "grow2": dict(layout=([2, 4, 8, 16, 32], [1, 1, 1, 1], [1, 1, 1, 1, 2]), readout="coarse"),
}


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps":
        torch.mps.synchronize()


def bench(name, kw, batch, dev, reps, warmup):
    model = build("erwin", dev, **kw).eval()
    pts, fts, vec, msk = (t[:batch].to(dev) for t in batch_data[:4])
    args = ("erwin", pts, fts, vec, msk)

    # FlopCounterMode's module hooks need a grad-carrying input (as in
    # tests/flops_compare.py); counting the forward only, as at inference.
    fc = FlopCounterMode(display=False)
    with fc:
        call(model, "erwin", pts, fts.clone().requires_grad_(), vec, msk)
    flops = fc.get_total_flops() / batch
    return {"flops_per_jet": flops, "params": sum(p.numel() for p in model.parameters()),
            **timed(model, args, dev, batch, reps, warmup)}


@torch.no_grad()
def timed(model, args, dev, batch, reps, warmup):
    for _ in range(warmup):
        call(model, *args)
    times = []
    for _ in range(reps):
        sync(dev); t = time.perf_counter()
        call(model, *args)
        sync(dev); times.append(time.perf_counter() - t)
    ms = statistics.median(times) * 1e3
    return {"latency_ms_per_batch": ms, "latency_us_per_jet": ms * 1e3 / batch,
            "latency_iqr_ms": (statistics.quantiles(times, n=4)[2]
                               - statistics.quantiles(times, n=4)[0]) * 1e3}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    args = ap.parse_args()
    dev = torch.device(args.device)
    torch.manual_seed(0)

    batch_data = load_all(find_root(WORKSPACE), seed=0, entry_stop=args.batch)
    out = {"device": args.device, "batch": args.batch, "reps": args.reps,
           "torch": torch.__version__, "threads": torch.get_num_threads(), "variants": {}}
    for name in args.variants.split(","):
        r = bench(name, VARIANTS[name], args.batch, dev, args.reps, args.warmup)
        out["variants"][name] = r
        print(f"{name:<6} {r['flops_per_jet'] / 1e6:8.1f} MFLOP/jet  "
              f"{r['latency_ms_per_batch']:8.1f} ms/batch  {r['latency_us_per_jet']:7.1f} us/jet  "
              f"{r['params'] / 1e6:.2f}M params", flush=True)

    path = os.path.join(OUTDIR, "variants_bench.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"wrote {path}")
