"""Efficiency figures for the FPGA case (ideas/visualization_ideas.md #4, #5, #7).

Reads, never measures:
  tests/s10_results/fpga_efficiency.json   tests/fpga_efficiency.py (batch 1, CPU, 1 thread)
  tests/s10_results/test_multiplicity.json the 2M test jets' constituent counts

Writes plots/fpga_tradeoff/{scaling,memory_breakdown,latency}.png.

    python tests/plot_fpga_efficiency.py
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

INK, MUTED, GRID = "#1f1f22", "#6b6a74", "#e2e1e6"
# Same model identity as plots/new_comparison_plots (slots 1-4, fixed order).
MODELS = {"ParT": ("ParT", "#1baf7a", "^"),
          "ErwinParTv2": ("ErwinParTv2", "#2a78d6", "o"),
          "ErwinParTv2+Lund": ("ErwinParTv2 + Lund", "#eb6834", "s"),
          "ErwinParT v1": ("ErwinParT v1", "#eda100", "D")}
# Components take the palette in fixed order too; only this figure colours by component.
COMPONENTS = [("embedding", "#2a78d6"), ("Lund tokens", "#6250d6"), ("pair features", "#eb6834"),
              ("particle attention", "#1baf7a"), ("pooling", "#eda100"),
              ("class attention + head", "#e87ba4"), ("geometry, tree & other", "#8a8994")]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "axes.formatter.use_mathtext": True,
    "font.size": 10, "axes.titlesize": 11, "legend.fontsize": 8, "legend.frameon": False,
    "axes.labelcolor": INK, "text.color": INK, "axes.edgecolor": MUTED,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "lines.linewidth": 1.8, "figure.facecolor": "white", "savefig.facecolor": "white",
    "savefig.dpi": 200, "savefig.bbox": "tight",
})


def titled(ax, main, sub=None):
    ax.set_title(main, loc="left", pad=16 if sub else 6)
    if sub:
        ax.text(0, 1.015, sub, transform=ax.transAxes, fontsize=8, color=MUTED, va="bottom")


def grid(ax, axis="both"):
    ax.grid(True, axis=axis, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path)
    plt.close(fig)
    print("  wrote", os.path.relpath(path, os.path.join(HERE, "..")))


def multiplicity():
    counts = json.load(open(os.path.join(RES, "test_multiplicity.json")))["counts"]
    return np.sum([np.array(c) for c in counts.values()], axis=0)


# --------------------------------------------------------------- #4 scaling --
def plot_scaling(d, hist):
    fig = plt.figure(figsize=(11, 5.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[4, 1], hspace=0.08, wspace=0.28)
    panels = [("flops", 1e9, "forward GFLOPs per jet", "Compute vs particles per jet"),
              ("peak_bytes", 1e6, "peak activation memory per jet [MB, fp32]", "Memory vs particles per jet")]
    n_ref = 40                                     # close to the test-set median (38)
    for col, (key, scale, ylab, title) in enumerate(panels):
        ax = fig.add_subplot(gs[0, col])
        for name, (label, c, mk) in MODELS.items():
            s = d["models"][name]["scaling"]
            ax.plot([r["n"] for r in s], [r[key] / scale for r in s], "-", color=c, marker=mk, ms=5,
                    label=label + (" (padding trimmed per jet)" if name in ("ParT", "ErwinParT v1") else ""))
            if name in ("ParT", "ErwinParT v1"):        # v2 fills its 64 slots either way
                ax.plot([r["n"] for r in s], np.array([r[key + "_fixed"] for r in s]) / scale, "--", color=c, lw=1.4,
                        label=label + " deployed (all 128 slots)")
        part = {r["n"]: r[key] for r in d["models"]["ParT"]["scaling"]}
        v2 = {r["n"]: r[key] for r in d["models"]["ErwinParTv2"]["scaling"]}
        fixed = d["models"]["ParT"]["scaling"][-1][key] / d["models"]["ErwinParTv2"]["scaling"][-1][key]
        r40 = part[n_ref] / v2[n_ref]
        ax.text(0.98, 0.04, f"ParT / ErwinParTv2:\n  deployed {fixed:.1f}$\\times$\n"
                f"  trimmed, N = {n_ref}: {r40:.2f}$\\times$", transform=ax.transAxes, ha="right",
                va="bottom", fontsize=8.5, color=INK)
        ax.axvline(64, color=MUTED, ls="--", lw=0.9)
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xticks(d["n_grid"], [])
        ax.set_ylabel(ylab)
        grid(ax)
        titled(ax, title, "batch 1, synthetic jets with exactly N particles; grey dashed line: v2 keeps 64")

        hx = fig.add_subplot(gs[1, col], sharex=ax)
        n = np.arange(len(hist))
        keep = n >= 1
        hx.fill_between(n[keep], hist[keep] / hist.sum() * 100, step="mid", color=GRID, lw=0)
        hx.axvline(64, color=MUTED, ls="--", lw=0.9)
        hx.set_xscale("log", base=2)
        hx.set_xticks(d["n_grid"], [str(v) for v in d["n_grid"]])
        hx.set_xlim(7, 140)
        hx.set_yticks([])
        hx.set_xlabel("particles per jet N")
        hx.text(0.01, 0.62, "2M test jets", transform=hx.transAxes, fontsize=7.5, color=MUTED)
        for s in ("top", "right", "left"):
            hx.spines[s].set_visible(False)
    h, l = fig.axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.1), fontsize=8)
    save(fig, "scaling.png")


# ---------------------------------------------------------------- #5 memory --
def plot_memory(d):
    order = ["ParT", "ErwinParT v1", "ErwinParTv2", "ErwinParTv2+Lund"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.6), sharey=True)
    y = np.arange(len(order))[::-1]
    for ax, (n, key, title) in zip(axes, ((40, "peak_by_component", "Median-size jet, padding trimmed (N = 40)"),
                                           (40, "peak_by_component_fixed", "Deployed: every jet uses all 128 slots"))):
        tot = []
        for yi, name in zip(y, order):
            rec = next(r for r in d["models"][name]["scaling"] if r["n"] == n)
            left = 0.0
            for comp, c in COMPONENTS:
                v = rec[key].get(comp, 0) / 1e6
                if v <= 0:
                    continue
                ax.barh(yi, v, left=left, height=0.62, color=c, edgecolor="white", lw=1.5,
                        label="tree build, Lund clustering & other" if comp == "geometry, tree & other" else comp)
                left += v
            tot.append(left)
            w = d["models"][name]["weight_bytes"] / 1e6
            ax.text(left, yi, f"  {left:.2f} MB   (+ {w:.1f} MB weights)", va="center", fontsize=8)
        ax.set_xlim(0, max(tot) * 1.6)
        ax.set_xlabel("activation memory live at the peak [MB, fp32]")
        grid(ax, "x"); ax.tick_params(axis="y", length=0)
        titled(ax, title, "batch 1; coloured by the component that created each live tensor")
    axes[0].set_yticks(y, [MODELS[n][0] for n in order])
    h, l = [], []
    for ax in axes:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh); l.append(ll)
    shown = {"geometry, tree & other": "tree build, Lund clustering & other"}
    keep = [shown.get(c, c) for c, _ in COMPONENTS if shown.get(c, c) in l]
    fig.legend([h[l.index(c)] for c in keep], keep, loc="lower center", ncol=len(keep),
               bbox_to_anchor=(0.5, -0.1), fontsize=8)
    fig.suptitle("Memory footprint by component", x=0.01, ha="left", y=1.04, fontsize=11)
    save(fig, "memory_breakdown.png")


# --------------------------------------------------------------- #7 latency --
def plot_latency(d):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    rng = np.random.default_rng(0)
    for name, (label, c, mk) in MODELS.items():
        lat = d["models"][name]["latency"]
        n = np.array([r["n"] for r in lat]) + rng.uniform(-0.4, 0.4, len(lat))
        ax.scatter(n, [r["ms"] for r in lat], s=9, color=c, marker=mk, alpha=0.45, lw=0, label=label)
        s = d["models"][name]["scaling"]
        ax.plot([r["n"] for r in s], [r["latency_ms_fixed"] for r in s], "--", color=c, lw=1.4)
    ax.axvline(64, color=MUTED, ls="--", lw=0.9)
    ax.set_yscale("log")
    ax.set_xlabel("particles per jet N"); ax.set_ylabel("latency per jet [ms]")
    grid(ax)
    ax.legend(loc="upper left", markerscale=2)
    titled(ax, "Batch-1 latency vs jet size",
           f"dots: {len(d['models']['ParT']['latency'])} real jets, trimmed per jet (min of 7); dashed: deployed, "
           "128 slots (min of 15); CPU, 1 thread")
    ax = axes[1]
    order = ["ParT", "ErwinParT v1", "ErwinParTv2", "ErwinParTv2+Lund"]
    y = np.arange(len(order))[::-1]
    for yi, name in zip(y, order):
        label, c, mk = MODELS[name]
        v = np.array([r["ms"] for r in d["models"][name]["latency"]])
        p5, p50, p95, p99 = np.percentile(v, [5, 50, 95, 99])
        ax.plot([v.min(), v.max()], [yi, yi], color=c, lw=1, alpha=0.5)
        ax.plot([p5, p95], [yi, yi], color=c, lw=5, alpha=0.6, solid_capstyle="butt")
        ax.scatter([p50], [yi], s=60, marker=mk, color=c, edgecolors="white", lw=1.5, zorder=3)
        ax.annotate(f"p50 {p50:.1f} ms, max/min {v.max() / v.min():.1f}$\\times$",
                    (v.max(), yi), xytext=(6, -3), textcoords="offset points", fontsize=8)
    ax.set_xscale("log")
    ax.set_yticks(y, [MODELS[n][0] for n in order]); ax.tick_params(axis="y", length=0)
    lo, hi = ax.get_xlim(); ax.set_xlim(lo, hi * 12)
    ax.set_xlabel("latency per jet [ms]")
    grid(ax, "x")
    titled(ax, "Spread", "marker: median; thick: 5th-95th percentile; thin: min-max")
    fig.tight_layout()
    save(fig, "latency.png")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    d = json.load(open(os.path.join(RES, "fpga_efficiency.json")))
    plot_scaling(d, multiplicity())
    plot_memory(d)
    plot_latency(d)
