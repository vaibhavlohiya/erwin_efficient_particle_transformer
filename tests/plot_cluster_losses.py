"""Training and validation loss per epoch for the cluster 10% runs.

Reads tests/s10_results/cluster_losses.json, written by
Cluster/scripts/read_losses.py from each run's TensorBoard events (weaver puts
only the training loss in train.log). Writes plots/new_comparison_plots/loss_curves.png.

A point is missing where a pod was killed before TensorBoard flushed that
epoch's event: the epoch was trained (train.log and the checkpoints have it),
but its scalar was lost. Validation loss sits below training loss because
training runs with dropout and averages over an epoch whose weights keep moving.

    python tests/plot_cluster_losses.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "s10_results")
OUT = os.path.join(HERE, "..", "plots", "new_comparison_plots")

INK, MUTED, GRID = "#1f1f22", "#6b6a74", "#e2e1e6"
P, Q = "20261029-35e96f7-", "20261102-04fe81e-"
# (run, label, colour, line style). Ball size m is ordered: one blue, light -> dark.
RUNS = [
    (P + "v2-10pct-a", "ErwinParTv2, m = 8 (run a)", "#9cc3ef", "-"),
    (P + "v2-10pct-b", "ErwinParTv2, m = 8 (run b)", "#9cc3ef", "--"),
    (Q + "m16-10pct", "ErwinParTv2, m = 16", "#5b9be3", "-"),
    (Q + "m32-10pct", "ErwinParTv2, m = 32", "#2a78d6", "-"),
    (Q + "m64-10pct", "ErwinParTv2, m = 64 (dense)", "#174a8c", "-"),
    (P + "lund-10pct-a", "ErwinParTv2 + Lund (run a)", "#eb6834", "-"),
    (P + "lund-10pct-b", "ErwinParTv2 + Lund (run b)", "#eb6834", "--"),
    (P + "part-10pct", "ParT", "#1baf7a", "-"),
]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "axes.formatter.use_mathtext": True,
    "font.size": 10, "axes.titlesize": 11, "legend.fontsize": 8, "legend.frameon": False,
    "axes.labelcolor": INK, "text.color": INK, "axes.edgecolor": MUTED,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "figure.facecolor": "white", "savefig.facecolor": "white", "savefig.dpi": 200, "savefig.bbox": "tight",
})


def titled(ax, main, sub=None):
    ax.set_title(main, loc="left", pad=16 if sub else 6)
    if sub:
        ax.text(0, 1.015, sub, transform=ax.transAxes, fontsize=8, color=MUTED, va="bottom")


def series(rec, tag):
    """Values on a 0..19 epoch axis, NaN where the event was lost."""
    y = np.full(20, np.nan)
    for e, v in zip(rec["epochs"].get(tag, []), rec["series"].get(tag, [])):
        y[e] = v
    return y


if __name__ == "__main__":
    d = json.load(open(os.path.join(RES, "cluster_losses.json")))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    x = np.arange(20)
    for ax, tag, title in ((axes[0], "Loss/train (epoch)", "Training loss"),
                           (axes[1], "Loss/eval (epoch)", "Validation loss")):
        for run, label, c, ls in RUNS:
            if run not in d:
                continue
            y = series(d[run], tag)
            ax.plot(x, y, ls, color=c, lw=1.6, marker="o", ms=2.8, label=label)
        ax.axvspan(13.5, 19.5, color=GRID, alpha=0.45, lw=0, zorder=0)
        ax.set_xlim(-0.5, 19.5)
        ax.set_xticks(range(0, 20, 2))
        ax.set_xlabel("epoch")
        ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
        titled(ax, title, "shaded: learning-rate decay (epochs 14-19)")
    axes[0].set_ylabel("cross-entropy loss")
    lo = np.nanmin([np.nanmin(series(d[r], "Loss/eval (epoch)")) for r, *_ in RUNS if r in d])
    axes[0].set_ylim(lo - 0.01, 0.62)
    for ax in axes:
        ax.text(0.98, 0.97, "earliest epochs above the range", transform=ax.transAxes, fontsize=7.5,
                color=MUTED, va="top", ha="right")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.12), fontsize=8)
    fig.suptitle("10% of JetClass, 20 epochs, batch 512; validation on 128k jets per epoch. Gaps: events lost "
                 "when a pod was killed (the epoch was trained).", x=0.01, ha="left", y=1.02, fontsize=9, color=MUTED)
    fig.tight_layout()
    path = os.path.join(OUT, "loss_curves.png")
    fig.savefig(path)
    print("  wrote", os.path.relpath(path, os.path.join(HERE, "..")))
    print(f"  {'run':<32}{'final train':>12}{'final val':>11}{'best val':>10}{'at epoch':>10}")
    for run, label, *_ in RUNS:
        if run in d:
            tr, va = series(d[run], "Loss/train (epoch)"), series(d[run], "Loss/eval (epoch)")
            print(f"  {label:<32}{tr[19]:>12.4f}{va[19]:>11.4f}{np.nanmin(va):>10.4f}{int(np.nanargmin(va)):>10}")
