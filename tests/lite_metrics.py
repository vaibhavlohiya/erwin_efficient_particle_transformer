"""Shared metrics and cost measurements for the FPGA-compression study.

Every variant in the study (baseline, compressed, distilled, quantized) is scored
by the functions here, so the comparison is like for like.

Metrics: the local test split holds ~1,000 QCD jets, so a 50% working point
whose cut no QCD jet passes reads as the ceiling ("> N"). For most JetClass
classes that already happens for the baseline, so 50% rejection cannot show a
loss there. `local_metrics` therefore also reports rejection at 90% signal
efficiency, where enough QCD passes to measure, and flags every capped value
with its QCD pass count.

Cost: parameters; multiply-accumulates per jet (FlopCounterMode / 2) and peak
live activation memory, both for the deployed model (SequenceTrimmer off, every
jet on all its slots, batch 1); batch-1 CPU latency (1 thread, minimum over
repeats). The int8 memory figure is the fp32 one / 4: what the same tensors
would need at 8 bits, not a measurement of a quantized model.
"""
import os
import sys
import time

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
from fpga_efficiency import LiveMemory  # noqa: E402

CLASS_NAMES = ["QCD", "Hbb", "Hcc", "Hgg", "H4q", "Hqql", "Zqq", "Wqq", "Tbqq", "Tbl"]
QCD = 0


def _auc(scores, positive):
    """One-vs-rest AUC by rank statistic, ties averaged."""
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    npos, nneg = int(positive.sum()), int((~positive).sum())
    return float((ranks[positive].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def _rejection(probs, y, sig, eff):
    """(1/eps_B, QCD jets passing, QCD jets) at signal efficiency `eff`, signal vs QCD,
    with ParT's discriminant s_sig / (s_sig + s_QCD). Zero passing -> rejection = N, capped."""
    keep = (y == sig) | (y == QCD)
    p, lab = probs[keep], y[keep]
    d = p[:, sig] / np.clip(p[:, sig] + p[:, QCD], 1e-12, None)
    s, b = d[lab == sig], d[lab == QCD]
    n_pass = int((b >= np.quantile(s, 1 - eff)).sum())
    return (len(b) / n_pass if n_pass else float(len(b))), n_pass, int(len(b))


def local_metrics(probs, y):
    """Accuracy, macro and per-class AUC, and per-class rejection at 50% and 90%."""
    out = {"accuracy": float((probs.argmax(1) == y).mean()), "n_test": int(len(y)), "auc": {}, "rej50": {},
           "rej90": {}}
    aucs = []
    for c, name in enumerate(CLASS_NAMES):
        a = _auc(probs[:, c], y == c)
        aucs.append(a)
        out["auc"][name] = a
        if c == QCD:
            continue
        for eff, key in ((0.5, "rej50"), (0.9, "rej90")):
            r, n, nb = _rejection(probs, y, c, eff)
            out[key][name] = {"rej": r, "qcd_pass": n, "n_qcd": nb, "capped": n == 0}
    out["auc_macro"] = float(np.mean(aucs))
    return out


# ------------------------------------------------------------------ cost --
def _synthetic_jet(n, C, P):
    g = torch.Generator().manual_seed(0)
    pts, fts = torch.zeros(1, 2, P), torch.zeros(1, C, P)
    vec, msk = torch.zeros(1, 4, P), torch.zeros(1, 1, P, dtype=torch.bool)
    pts[0, :, :n] = torch.randn(2, n, generator=g) * 0.3
    fts[0, :, :n] = torch.randn(C, n, generator=g)
    p3 = torch.randn(3, n, generator=g).abs() * 10 + 1
    vec[0, :3, :n] = p3
    vec[0, 3, :n] = p3.norm(dim=0) + 0.1
    msk[0, 0, :n] = True
    return pts, fts, vec, msk


def deployed_cost(model, call, C, P=128, n=64, latency_reps=15):
    """Cost of one jet with `n` real particles, padded to P, trimming off.

    `call(model, pts, fts, vec, msk)` runs the model. A model that keeps a fixed
    number of slots (ErwinParTv2's 64) costs the same for any n >= that number.
    """
    model = model.cpu().eval()
    trimmer = getattr(getattr(model, "mod", model), "trimmer", None)
    if trimmer is not None:
        with torch.no_grad():
            for _ in range(6):                 # past SequenceTrimmer's 5 untrimmed calls
                call(model, *_synthetic_jet(8, C, P))
        trimmer.enabled = False
    jet = _synthetic_jet(n, C, P)
    fc = FlopCounterMode(display=False)        # its module hooks need grad enabled
    with fc:
        call(model, *jet)
    exclude = {p.untyped_storage().data_ptr() for p in model.parameters()}
    exclude |= {b.untyped_storage().data_ptr() for b in model.buffers()}
    exclude |= {t.untyped_storage().data_ptr() for t in jet}
    lm = LiveMemory(exclude, lambda: "all")
    with torch.no_grad(), lm:
        call(model, *jet)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    times = []
    with torch.no_grad():
        for _ in range(3):
            call(model, *jet)
        for _ in range(latency_reps):
            t = time.perf_counter()
            call(model, *jet)
            times.append(time.perf_counter() - t)
    torch.set_num_threads(threads)
    if trimmer is not None:
        trimmer.enabled = True
    params = sum(p.numel() for p in model.parameters())
    return {"params": params, "weight_mb_fp32": params * 4 / 1e6,
            "macs": fc.get_total_flops() / 2, "peak_act_mb_fp32": lm.peak / 1e6,
            "peak_act_mb_int8_equiv": lm.peak / 4 / 1e6, "latency_ms_cpu": min(times) * 1e3}
