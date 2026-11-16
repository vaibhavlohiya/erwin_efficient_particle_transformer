"""Per-epoch training and validation loss/accuracy from runs' TensorBoard events.

weaver writes "Loss/train (epoch)", "Acc/train (epoch)" and, for validation,
"Loss/eval (epoch)", "Acc/eval (epoch)" with the epoch as the step; train.log
only has the training loss. A resumed run leaves one event file per attempt, so
for each (tag, epoch) the most recent write wins - for a run that was rolled
back, that is the epoch it finally used. Epochs >= --epochs are dropped.

    python Cluster/scripts/read_losses.py --epochs 20 /output/runs/<run> [...] > losses.json
"""
import argparse
import glob
import json
import os
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=20)
ap.add_argument("runs", nargs="+")
args = ap.parse_args()

out = {}
for run in args.runs:
    files = sorted(glob.glob(os.path.join(run, "tensorboard", "**", "events.out.tfevents*"), recursive=True))
    best = {}                                   # (tag, epoch) -> (wall_time, value)
    for f in files:
        acc = EventAccumulator(f, size_guidance={"scalars": 0})
        acc.Reload()
        for tag in acc.Tags().get("scalars", []):
            if "(epoch)" not in tag:
                continue
            for e in acc.Scalars(tag):
                if e.step >= args.epochs:
                    continue
                key = (tag, e.step)
                if key not in best or e.wall_time > best[key][0]:
                    best[key] = (e.wall_time, e.value)
    rec = {}
    for (tag, step), (_, v) in best.items():
        rec.setdefault(tag, {})[step] = v
    out[os.path.basename(run.rstrip("/"))] = {
        "event_files": len(files),
        "series": {t: [rec[t][s] for s in sorted(rec[t])] for t in rec},
        "epochs": {t: sorted(rec[t]) for t in rec}}
    print(f"{os.path.basename(run)}: {len(files)} event files, tags {sorted(rec)}", file=sys.stderr, flush=True)
json.dump(out, sys.stdout)
