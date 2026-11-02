"""Evaluate finished cluster runs from their weaver prediction files.

Runs where the predictions live (a pod with the output PVC mounted read-only),
so nothing large leaves the cluster: it prints ONE JSON document to stdout.

For every /output/runs/<run>/ with all 10 predict_output/pred_*.root files:
  accuracy, macro one-vs-rest AUC, per-class AUC
  per signal class vs QCD, with ParT's discriminant s_i / (s_i + s_QCD):
    AUC, a background-rejection curve on a fixed signal-efficiency grid, and
    the rejection at ParT's published working points (50%; 99% for H->lvqq,
    99.5% for t->blv), with the QCD count so a zero pass is a lower bound
  training/validation minutes per epoch, parsed from train.log (the last
  occurrence of each epoch, so resumed epochs count once)

    python Cluster/scripts/eval_runs.py /output/runs/<run> [...] > eval.json
"""
import datetime as dt
import glob
import json
import os
import re
import sys

import numpy as np
import uproot
from sklearn.metrics import roc_auc_score

WORKING_POINT = {"Hqql": 0.99, "Tbl": 0.995}          # everything else: 50%
EFF_GRID = np.round(np.concatenate([np.linspace(0.02, 0.9, 45), np.linspace(0.91, 0.995, 18)]), 4)
TS = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+\] INFO: (.*)")


def epoch_minutes(log):
    """{epoch: {"train": min, "val": min}} from weaver's train.log, last occurrence wins."""
    out, cur, t0, phase = {}, None, None, None
    for line in open(log, errors="replace"):
        m = TS.match(line)
        if not m:
            continue
        t = dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
        msg = m.group(2)
        if (e := re.match(r"Epoch #(\d+) (training|validating)", msg)):
            cur, phase, t0 = int(e.group(1)), e.group(2), t
        elif phase == "training" and msg.startswith("Train AvgLoss") and cur is not None:
            out.setdefault(cur, {})["train"] = (t - t0).total_seconds() / 60
        elif phase == "validating" and "Current validation metric" in msg and cur is not None:
            out.setdefault(cur, {})["val"] = (t - t0).total_seconds() / 60
            vm = re.search(r"metric: ([0-9.]+)", msg)
            out[cur]["val_acc"] = float(vm.group(1)) if vm else None
    return out


def evaluate(run_dir):
    files = sorted(glob.glob(os.path.join(run_dir, "predict_output", "pred_*.root")))
    if len(files) < 10:
        return None
    arrs = [uproot.open(f)["Events"].arrays(library="np") for f in files]
    lab = [k for k in arrs[0] if k.startswith("label_")]
    names = [k[len("label_"):] for k in lab]
    Y = np.concatenate([np.stack([a[k] for k in lab], 1) for a in arrs]).astype(np.float64)
    S = np.concatenate([np.stack([a["score_" + k] for k in lab], 1) for a in arrs]).astype(np.float64)
    y = Y.argmax(1)
    q = names.index("QCD")
    res = {"n_test": int(len(y)), "classes": names,
           "accuracy": float((S.argmax(1) == y).mean()),
           "auc_ovr": {n: float(roc_auc_score(Y[:, i], S[:, i])) for i, n in enumerate(names)},
           "per_class_acc": {n: float((S.argmax(1)[y == i] == i).mean()) for i, n in enumerate(names)},
           "vs_qcd": {}}
    res["auc_macro"] = float(np.mean(list(res["auc_ovr"].values())))
    for i, n in enumerate(names):
        if i == q:
            continue
        m = (y == i) | (y == q)
        d = S[m, i] / (S[m, i] + S[m, q] + 1e-12)
        t = y[m] == i
        sig, bkg = np.sort(d[t]), d[~t]
        cuts = np.quantile(sig, 1 - EFF_GRID)
        fb = np.array([(bkg >= c).mean() for c in cuts])
        wp = WORKING_POINT.get(n, 0.5)
        fb_wp = float((bkg >= np.quantile(sig, 1 - wp)).mean())
        res["vs_qcd"][n] = {
            "auc": float(roc_auc_score(t, d)), "n_sig": int(t.sum()), "n_bkg": int((~t).sum()),
            "eff": EFF_GRID.tolist(), "rej": [None if f == 0 else 1 / f for f in fb],
            "wp": wp, "rej_wp": None if fb_wp == 0 else 1 / fb_wp,
            "n_bkg_pass_wp": int(round(fb_wp * (~t).sum()))}
    return res


if __name__ == "__main__":
    out = {}
    for run_dir in sys.argv[1:]:
        name = os.path.basename(run_dir.rstrip("/"))
        rec = {"run_dir": run_dir}
        log = os.path.join(run_dir, "train.log")
        if os.path.exists(log):
            rec["epochs"] = epoch_minutes(log)
        manifests = sorted(glob.glob(os.path.join(run_dir, "run_manifest*.json")))
        if manifests:
            first = json.load(open(manifests[0]))
            rec["model"] = first.get("model")
            rec["attempts"] = len(manifests)
        ev = evaluate(run_dir)
        if ev:
            rec.update(ev)
        out[name] = rec
        print(f"evaluated {name}: {'complete' if ev else 'no full predictions'}", file=sys.stderr, flush=True)
    json.dump(out, sys.stdout)
