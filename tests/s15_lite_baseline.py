"""Step 1 of the FPGA-compression study: audit the local baseline.

1. The local ErwinParTv2 checkpoints still load and reproduce the test accuracy
   and AUC they were saved with (same seed, same split, CPU) - so they are valid
   baselines and distillation teachers for the study.
2. The extended local metrics (lite_metrics.local_metrics) for those baselines,
   including which 50% working points are capped by the ~1,000 QCD test jets.
3. The deployed cost of the locally trained ErwinParTv2 (64 slots, trimming off,
   batch 1), against the cluster figures (2.21M parameters, ~96M MACs, 0.63 MB).

Writes tests/s10_results/lite_baseline.json. Run from the weaver conda env:
    python tests/s15_lite_baseline.py
"""
import json
import os
import sys
import warnings

import torch

warnings.filterwarnings("ignore")
HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
from jetclass_data import NUM_FEATURES, find_root, load_all  # noqa: E402
from lite_metrics import deployed_cost, local_metrics  # noqa: E402
from s10_compare import OUTDIR, WORKSPACE, _probs, build  # noqa: E402

CHECKPOINTS = [("erwin_m8", 8, 0), ("erwin_s1_base", 8, 1), ("erwin_m64", 64, 0)]   # (tag, ball, seed)
CLUSTER = {"params": 2.21e6, "macs": 0.1914e9 / 2, "peak_act_mb_fp32": 0.63}       # plugin, deployed
fails = []


def check(name, ok, detail=""):
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
    if not ok:
        fails.append(name)


if __name__ == "__main__":
    torch.manual_seed(0)
    out = {"checkpoints": {}}
    print("1. checkpoints reproduce their saved test metrics")
    for seed in sorted({s for _, _, s in CHECKPOINTS}):
        data = load_all(find_root(WORKSPACE), seed=seed)
        n = len(data[-1])
        test = torch.arange(int(0.9 * n), n)                 # s10_compare's split: 80/10/10
        for tag, ball, s in CHECKPOINTS:
            if s != seed:
                continue
            model = build("erwin", torch.device("cpu"), ball=ball)
            model.load_state_dict(torch.load(os.path.join(OUTDIR, f"{tag}_best.pt"), map_location="cpu"))
            probs, y = _probs(model, "erwin", data, test, torch.device("cpu"), 512)
            m = local_metrics(probs, y)
            saved = json.load(open(os.path.join(OUTDIR, f"{tag}_result.json")))
            check(f"{tag}: accuracy {m['accuracy']:.4f} vs saved {saved['accuracy']:.4f}, "
                  f"AUC {m['auc_macro']:.4f} vs {saved['auc_macro']:.4f}",
                  abs(m["accuracy"] - saved["accuracy"]) < 2e-4 and abs(m["auc_macro"] - saved["auc_macro"]) < 2e-4)
            out["checkpoints"][tag] = {"ball": ball, "seed": seed, **m}
        del data

    print("\n2. which working points the local test set can resolve (baseline, seed 0)")
    base = out["checkpoints"]["erwin_m8"]
    for key in ("rej50", "rej90"):
        capped = [c for c, v in base[key].items() if v["capped"]]
        few = [c for c, v in base[key].items() if not v["capped"] and v["qcd_pass"] < 10]
        print(f"  {key}: capped {capped or 'none'}; fewer than 10 QCD passing {few or 'none'}")
        out[f"{key}_capped_at_baseline"] = capped

    print("\n3. deployed cost of the locally trained ErwinParTv2 (ball 8, 64 slots)")
    model = build("erwin", torch.device("cpu"), ball=8)
    cost = deployed_cost(model, lambda m, p, f, v, k: m(f, v=v, mask=k, points=p), NUM_FEATURES)
    for k, v in cost.items():
        print(f"  {k:<24}{v:,.4g}")
    for k, ref in CLUSTER.items():
        check(f"{k} matches the cluster figure within 3%", abs(cost[k] / ref - 1) < 0.03,
              f"{cost[k]:,.4g} vs {ref:,.4g}")
    out["cost_baseline_ball8"] = cost
    json.dump(out, open(os.path.join(OUTDIR, "lite_baseline.json"), "w"), indent=1)
    print("\nwrote tests/s10_results/lite_baseline.json")
    print("S15 baseline FAILED: " + ", ".join(fails) if fails else "S15 baseline passed")
