"""Relative trade-off and per-class rejection ratio (ideas/visualization_ideas.md #2, #9).

Reads, never measures:
  tests/s10_results/cluster_eval.json      test-set metrics of the cluster 10% runs
  tests/s10_results/cluster_flops.json     deployed FLOPs and parameters
  tests/s10_results/fpga_efficiency.json   batch-1 memory and CPU latency, deployed
  tests/s10_results/test_multiplicity.json truncation at L = 64 per class

Writes plots/fpga_tradeoff/{relative_tradeoff,rejection_ratio}.png.

    python tests/plot_fpga_relative.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "s10_results")
OUT = os.path.join(HERE, "..", "plots", "fpga_tradeoff")
PREFIX = "20261029-35e96f7-"

# The target zone of the relative plot: an EXAMPLE budget until an FPGA target is
# chosen - at least TARGET_ACC % of ParT's accuracy at no more than TARGET_COST %
# of its cost.
TARGET_ACC, TARGET_COST = 98.0, 30.0

INK, MUTED, GRID = "#1f1f22", "#6b6a74", "#e2e1e6"
ZONE = "#e8f5ec"
MODELS = {"v2":   ("ErwinParTv2", "#2a78d6", "o", ["v2-10pct-a", "v2-10pct-b"], "ErwinParTv2"),
          "lund": ("ErwinParTv2 + Lund", "#eb6834", "s", ["lund-10pct-a", "lund-10pct-b"], "ErwinParTv2+Lund")}
PART = ("ParT", "#1baf7a", "^")
CLASS_TEX = {"Hbb": r"$H\to b\bar b$", "Hcc": r"$H\to c\bar c$", "Hgg": r"$H\to gg$",
             "H4q": r"$H\to 4q$", "Hqql": r"$H\to \ell\nu qq'$", "Zqq": r"$Z\to q\bar q$",
             "Wqq": r"$W\to qq'$", "Tbqq": r"$t\to bqq'$", "Tbl": r"$t\to b\ell\nu$"}

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


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path)
    plt.close(fig)
    print("  wrote", os.path.relpath(path, os.path.join(HERE, "..")))


def load():
    ev = json.load(open(os.path.join(RES, "cluster_eval.json")))
    runs = {k: [ev[PREFIX + t] for t in m[3] if "accuracy" in ev.get(PREFIX + t, {})] for k, m in MODELS.items()}
    part = ev[PREFIX + "part-10pct"]
    fl = json.load(open(os.path.join(RES, "cluster_flops.json")))["models"]
    fp = json.load(open(os.path.join(RES, "fpga_efficiency.json")))["models"]
    return runs, part, fl, fp


def costs(fl, fp, name):
    """Deployed cost measures of one model: every jet uses all 128 slots."""
    return {"flops": fl[name]["gflops_per_jet"],
            "memory": next(r for r in fp[name]["scaling"] if r["n"] == 128)["peak_bytes_fixed"],
            "latency": float(np.median([r["latency_ms_fixed"] for r in fp[name]["scaling"]])),
            "params": fl[name]["params"]}


# ------------------------------------------------------------- #2 relative --
def plot_relative(runs, part, fl, fp):
    panels = [("flops", "forward FLOPs per jet"), ("memory", "peak activation memory per jet"),
              ("latency", "latency per jet (CPU, batch 1)"), ("params", "parameters")]
    ref = costs(fl, fp, "ParT")
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.9), sharey=True)
    for ax, (key, label) in zip(axes, panels):
        ax.add_patch(plt.Rectangle((0, TARGET_ACC), TARGET_COST, 101.5 - TARGET_ACC, color=ZONE, lw=0, zorder=0))
        ax.axhline(100, color=MUTED, lw=0.7, ls="--"); ax.axvline(100, color=MUTED, lw=0.7, ls="--")
        ax.scatter([100], [100], s=80, marker=PART[2], color=PART[1], edgecolors="white", lw=1.4, zorder=4)
        ax.annotate("ParT", (100, 100), xytext=(-6, 7), textcoords="offset points", ha="right", fontsize=8.5)
        for k, (name, c, mk, _, fkey) in MODELS.items():
            if not runs[k]:
                continue
            x = costs(fl, fp, fkey)[key] / ref[key] * 100
            ys = [r["accuracy"] / part["accuracy"] * 100 for r in runs[k]]
            if len(ys) > 1:
                ax.scatter([x] * len(ys), ys, s=22, facecolors="white", edgecolors=c, lw=1.2, zorder=3)
            ax.scatter([x], [np.mean(ys)], s=80, marker=mk, color=c, edgecolors="white", lw=1.4, zorder=4,
                       label=name if key == "flops" else None)
            ax.annotate(f"{x:.0f}%", (x, np.mean(ys)), xytext=(7, -12 if k == "lund" else 6),
                        textcoords="offset points", fontsize=8, color=INK)
        ax.set_xlim(0, 135)
        ax.set_xlabel(f"% of ParT's {label}")
        ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
        titled(ax, label[0].upper() + label[1:])
    axes[0].set_ylim(97.0, 100.8)
    axes[0].set_ylabel("% of ParT's test accuracy")
    axes[0].legend(loc="lower right", fontsize=7.5)
    axes[0].text(2, TARGET_ACC + 0.12, f"example target:\n$\\geq${TARGET_ACC:g}% accuracy,\n"
                 f"$\\leq${TARGET_COST:g}% cost", fontsize=7.5, color="#1a7f37", va="bottom")
    fig.suptitle("Relative to ParT: same training budget (10% of JetClass, 20 epochs); costs when deployed "
                 "(all 128 slots); left of 100% = cheaper", x=0.01, ha="left", y=1.0, fontsize=10, color=MUTED)
    fig.tight_layout()
    save(fig, "relative_tradeoff.png")


# ---------------------------------------------------------- #9 per class --
def ratio_and_error(recs, part, cls):
    """Geometric-mean rejection over runs / ParT's, and its Poisson error from the
    numbers of QCD jets that pass each cut (runs pooled)."""
    rej = float(np.exp(np.mean([np.log(r["vs_qcd"][cls]["rej_wp"]) for r in recs])))
    n_ours = sum(r["vs_qcd"][cls]["n_bkg_pass_wp"] for r in recs)
    n_part = part["vs_qcd"][cls]["n_bkg_pass_wp"]
    ratio = rej / part["vs_qcd"][cls]["rej_wp"]
    return ratio, ratio * np.sqrt(1 / n_ours + 1 / n_part), n_ours / len(recs)


def plot_rejection_ratio(runs, part):
    classes = list(CLASS_TEX)
    counts = json.load(open(os.path.join(RES, "test_multiplicity.json")))["counts"]
    trunc = {c: np.array(v)[65:].sum() / np.sum(v) * 100 for c, v in counts.items()}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axes[0]
    y = np.arange(len(classes))[::-1]
    off = {"v2": 0.13, "lund": -0.13}
    for k, (name, c, mk, _, _) in MODELS.items():
        if not runs[k]:
            continue
        for yi, cls in zip(y, classes):
            r, e, _ = ratio_and_error(runs[k], part, cls)
            ax.errorbar([r], [yi + off[k]], xerr=[[e], [e]], fmt=mk, color=c, ms=7, mec="white", mew=1.2,
                        elinewidth=1.3, capsize=2.5, label=f"{name} ({len(runs[k])} run{'s' if len(runs[k]) > 1 else ''})"
                        if cls == classes[0] else None, zorder=3)
            if k == "v2":
                ax.annotate(f"{r:.2f}", (r + e, yi + off[k]), xytext=(5, -3), textcoords="offset points", fontsize=8)
    wp = {c: part["vs_qcd"][c]["wp"] for c in classes}
    ax.axvline(1, color=INK, lw=1)
    ax.set_xscale("log")
    ax.set_xlim(0.55, 1.45)
    ax.set_xticks([0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4], ["0.6", "0.7", "0.8", "0.9", "1", "1.2", "1.4"])
    ax.minorticks_off()
    ax.set_yticks(y, [f"{CLASS_TEX[c]}  @{wp[c] * 100:g}%" for c in classes]); ax.tick_params(axis="y", length=0)
    ax.set_xlabel("background rejection relative to ParT  (< 1: ParT rejects more QCD)")
    ax.grid(True, axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=8)
    titled(ax, "Where the accuracy loss sits",
           "same training budget; error bars: counting error of the QCD jets passing each cut")

    ax = axes[1]
    k = "v2"
    for cls in classes:
        r, e, _ = ratio_and_error(runs[k], part, cls)
        ax.errorbar([trunc[cls]], [r], yerr=[[e], [e]], fmt="o", color=MODELS[k][1], ms=6, mec="white", mew=1.1,
                    elinewidth=1.1, capsize=2.5, zorder=3)
        ax.annotate(CLASS_TEX[cls], (trunc[cls], r), xytext=(6, -3),
                    textcoords="offset points", fontsize=8.5)
    ax.axhline(1, color=INK, lw=1)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlim(-0.05, 60)
    ax.set_xticks([0, 1, 5, 10, 25], ["0", "1", "5", "10", "25"])
    ax.set_ylim(0.6, 1.15)
    ax.set_xlabel("test jets of this class truncated at L = 64 [%]")
    ax.set_ylabel("ErwinParTv2 rejection / ParT's")
    ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    titled(ax, "Does it follow truncation?",
           "no clear trend: the well-measured losses are H→4q (8% truncated) and H→gg (25%)")
    fig.tight_layout()
    save(fig, "rejection_ratio.png")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    runs, part, fl, fp = load()
    plot_relative(runs, part, fl, fp)
    plot_rejection_ratio(runs, part)
    ref = costs(fl, fp, "ParT")
    for k, m in MODELS.items():
        if runs[k]:
            c = costs(fl, fp, m[4])
            acc = np.mean([r["accuracy"] for r in runs[k]]) / part["accuracy"] * 100
            print(f"  {m[0]:<20} accuracy {acc:.1f}% of ParT | " +
                  " ".join(f"{q} {c[q] / ref[q] * 100:.0f}%" for q in c))
    for cls in CLASS_TEX:
        r, e, n = ratio_and_error(runs["v2"], part, cls)
        print(f"  {cls:<5} ratio {r:.2f} +- {e:.2f}  (QCD passing: ours {n:.0f}/run, ParT "
              f"{part['vs_qcd'][cls]['n_bkg_pass_wp']})")
