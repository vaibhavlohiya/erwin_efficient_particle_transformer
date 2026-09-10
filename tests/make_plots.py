"""Figures for the ErwinParTv2 run on JetClass_example_100k.

Reads the artefacts S10 wrote (tests/s10_results/) and emits three PNGs to
plots/. Nothing here re-trains or re-evaluates; if a number is not in those
files it is not plotted.

  1. training_curves.png   loss / accuracy / AUC per epoch
  2. roc_curves.png        background rejection vs signal efficiency, per class
  3. efficiency.png        memory and throughput against the dense O(n^2) baseline
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "s10_results")
OUT = os.path.join(HERE, "..", "plots")

# Validated categorical slots 1 & 2 (see the dataviz reference palette).
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#1f1f22", "#6b6a74", "#e2e1e6"

plt.rcParams.update({
    "font.size": 9, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": GRID,
    "axes.titlesize": 10, "figure.facecolor": "white", "savefig.facecolor": "white",
})


def tidy(ax, grid_axis="both"):
    ax.grid(True, which="major", axis=grid_axis, color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ---------------------------------------------------------------- 1. curves --
def training_curves():
    h = json.load(open(os.path.join(RES, "erwin_history.json")))
    ep = [e["epoch"] for e in h]
    best = max(h, key=lambda e: e["val_acc"])["epoch"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    panels = [("loss", "training loss", "cross-entropy", False),
              ("val_acc", "validation accuracy", "accuracy", True),
              ("val_auc", "validation AUC", "macro one-vs-rest AUC", True)]
    for ax, (key, title, ylab, mark) in zip(axes, panels):
        ax.plot(ep, [e[key] for e in h], "-o", color=BLUE, lw=2, ms=3.5)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("epoch"); ax.set_ylabel(ylab)
        ax.set_xticks([1, 5, 10, 15, 20])
        if mark:
            ax.axvline(best, color=MUTED, ls="--", lw=1)
            ax.annotate(f"best (ep {best})", (best, ax.get_ylim()[0]),
                        xytext=(4, 8), textcoords="offset points",
                        color=MUTED, fontsize=8)
        tidy(ax)
    axes[0].annotate(f"{h[0]['loss']:.2f} → {h[-1]['loss']:.2f}",
                     (0.97, 0.86), xycoords="axes fraction", ha="right",
                     color=BLUE, fontsize=9, fontweight="bold")
    fig.suptitle("ErwinParTv2 · 80k train / 10k val · 20 epochs",
                 x=0.008, ha="left", fontsize=11, fontweight="bold")
    fig.text(0.008, -0.04, "Validation points were evaluated on MPS, which is "
             "batch-size sensitive (±0.01); trends are reliable, the final test "
             "number is the deterministic CPU one.", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = os.path.join(OUT, "training_curves.png")
    fig.savefig(p, dpi=160, bbox_inches="tight"); plt.close(fig)
    return p


# ------------------------------------------------------------------- 2. ROC --
def roc_curves():
    r = json.load(open(os.path.join(RES, "erwin_roc.json")))
    cur = r["curves"]
    names = list(cur)
    fig, axes = plt.subplots(3, 3, figsize=(11, 9.2), sharex=True)
    for ax, name in zip(axes.ravel(), names):
        c = cur[name]
        eff = np.array(c["eff"]); eps = np.array(c["eps_bkg"])
        ceil = c["n_bkg"]
        rej = np.where(eps > 0, 1.0 / np.clip(eps, 1e-12, None), ceil)
        sat = eps == 0
        ax.plot(eff[~sat], rej[~sat], color=BLUE, lw=2.2)
        if sat.any():
            ax.plot(eff[sat], rej[sat], color=BLUE, lw=2.2, ls=":", alpha=0.55)
        ax.axhline(ceil, color=MUTED, ls="--", lw=1)
        ax.set_yscale("log"); ax.set_ylim(1, ceil * 2.2); ax.set_xlim(0, 1)
        ax.set_title(f"{name}  vs QCD", loc="left", fontweight="bold")
        i50 = int(np.argmin(np.abs(eff - 0.5)))
        r50 = ceil if eps[i50] == 0 else 1 / eps[i50]
        ax.annotate(f"AUC {c['auc']:.4f}\nrej@50%  {'≥' if eps[i50]==0 else ''}{r50:.0f}",
                    (0.03, 0.03), xycoords="axes fraction", fontsize=8.5,
                    va="bottom", color=INK,
                    bbox=dict(fc="white", ec=GRID, lw=0.7, pad=3))
        tidy(ax)
    for ax in axes[-1]:
        ax.set_xlabel("signal efficiency  $\\varepsilon_S$")
    for ax in axes[:, 0]:
        ax.set_ylabel("background rejection  $1/\\varepsilon_B$")
    axes[0, 2].annotate(f"ceiling {cur[names[0]]['n_bkg']}\n(no QCD jet survives)",
                        (0.97, cur[names[0]]["n_bkg"]), xycoords=("axes fraction", "data"),
                        xytext=(0, 9), textcoords="offset points", ha="right",
                        fontsize=7.5, color=MUTED)
    fig.suptitle(f"ErwinParTv2 · ROC per signal class · {r['n_test']:,} test jets "
                 f"· accuracy {r['accuracy']:.4f}, macro AUC {r['auc_macro']:.4f}",
                 x=0.008, ha="left", fontsize=11, fontweight="bold")
    fig.text(0.008, -0.015, "Dotted segments and the dashed line mark the measurement "
             "ceiling: with only ~1,000 QCD test jets, a rejection at the ceiling means "
             "‘at least this’, not a measured value. No ParT baseline or ball-size "
             "sweep is shown — those runs do not exist yet.",
             fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(OUT, "roc_curves.png")
    fig.savefig(p, dpi=160, bbox_inches="tight"); plt.close(fig)
    return p


# ------------------------------------------------------------ 3. efficiency --
def efficiency():
    # Measured: S8 sweep (CPU), the batch-256/P=128 memory arithmetic, and the
    # three-run CPU throughput comparison.
    L = [64, 128, 256, 512]
    ball_ms = [1.96, 4.61, 7.25, 16.10]
    dense_ms = [3.80, 10.05, 42.23, 173.49]
    ball_mb = [16.0, 30.8, 63.9, 128.6]
    dense_mb = [47.6, 174.6, 686.1, 2857.2]

    fig = plt.figure(figsize=(13, 4.4))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1.3, 1.3], wspace=0.46,
                          left=0.055, right=0.99, top=0.70, bottom=0.16)

    ax = fig.add_subplot(gs[0])
    vals = [0.034, 6.42]
    bars = ax.bar(["ErwinParTv2", "ParT"], vals, color=[BLUE, ORANGE], width=0.6)
    ax.set_yscale("log"); ax.set_ylabel("GB"); ax.set_ylim(0.01, 30)
    ax.set_title("pair-MLP activations\nbatch 256, P=128", loc="left", fontweight="bold", pad=8)
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.3g} GB", (b.get_x() + b.get_width() / 2, v),
                    xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8.5)
    ax.annotate("189× less", (0.5, 0.75), xycoords="axes fraction", ha="center",
                fontsize=9, color=INK, fontweight="bold")
    tidy(ax, "y")

    ax = fig.add_subplot(gs[1])
    vals = [83.6, 46.7]
    bars = ax.bar(["ErwinParTv2", "ParT"], vals, color=[BLUE, ORANGE], width=0.6)
    ax.set_ylabel("jets / s"); ax.set_ylim(0, 105)
    ax.set_title("training throughput\nCPU, P=76, batch 64", loc="left", fontweight="bold", pad=8)
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.1f}", (b.get_x() + b.get_width() / 2, v),
                    xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8.5)
    ax.annotate("1.8× faster", (0.5, 0.86), xycoords="axes fraction", ha="center",
                fontsize=9, color=INK, fontweight="bold")
    tidy(ax, "y")

    for gi, (yb, yd, lab, unit, eb, ed) in enumerate([
            (ball_ms, dense_ms, "forward time", "ms", 0.98, 1.86),
            (ball_mb, dense_mb, "peak memory", "MB", 1.01, 1.97)]):
        ax = fig.add_subplot(gs[2 + gi])
        ax.plot(L, yd, "-o", color=ORANGE, lw=2, ms=5, label="dense pair attention")
        ax.plot(L, yb, "-o", color=BLUE, lw=2, ms=5, label="ball attention")
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xlabel("constituents per jet  L"); ax.set_ylabel(unit)
        ax.set_title(f"{lab} scaling", loc="left", fontweight="bold", pad=8)
        ax.annotate(f"$L^{{{ed:.2f}}}$", (0.97, 0.60), xycoords="axes fraction",
                    ha="right", color=ORANGE, fontsize=10, fontweight="bold")
        ax.annotate(f"$L^{{{eb:.2f}}}$", (0.97, 0.12), xycoords="axes fraction",
                    ha="right", color=BLUE, fontsize=10, fontweight="bold")
        tidy(ax)
        if gi == 0:
            ax.legend(frameon=False, fontsize=8, loc="upper left")

    fig.suptitle("ErwinParTv2 vs ParT's dense O(n\u00b2) pairwise attention",
                 x=0.008, y=0.965, ha="left", fontsize=12, fontweight="bold")
    fig.text(0.008, 0.015, "Memory and throughput are measured; the 189\u00d7 figure is "
             "the activation arithmetic that made ParT unrunnable at batch 256 on a "
             "16 GB machine while ErwinParTv2 trained in 41 min.",
             fontsize=7.5, color=MUTED)
    p = os.path.join(OUT, "efficiency.png")
    fig.savefig(p, dpi=160, bbox_inches="tight"); plt.close(fig)
    return p




# ------------------------------------------------------- 4. ball-size sweep --
# Ball size is an ordered quantity, so it gets a sequential ramp (light = small)
# rather than categorical hues; every curve is also direct-labelled, so identity
# never rests on colour alone.
RAMP = ["#9ecae1", "#4292c6", "#2171b5", "#08306b"]
MS = [8, 16, 32, 64]


def _sweep():
    return [json.load(open(os.path.join(RES, f"erwin_m{m}_result.json"))) for m in MS]


def ballsize_summary():
    rs = _sweep()
    acc = [r["accuracy"] for r in rs]
    auc = [r["auc_macro"] for r in rs]
    mins = [r["train_minutes"] for r in rs]
    n = rs[0]["n_test"] if "n_test" in rs[0] else 10000
    err = [np.sqrt(a * (1 - a) / n) for a in acc]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    x = np.arange(len(MS))

    ax = axes[0]
    ax.errorbar(x, acc, yerr=err, fmt="o-", color=RAMP[2], lw=2, ms=7,
                capsize=4, ecolor=MUTED, elinewidth=1)
    ax.set_ylabel("test accuracy"); ax.set_title("accuracy", loc="left", fontweight="bold")
    for xi, a in zip(x, acc):
        ax.annotate(f"{a:.4f}", (xi, a), xytext=(0, 11), textcoords="offset points",
                    ha="center", fontsize=8.5)
    ax.set_ylim(min(acc) - 0.006, max(acc) + 0.007)

    ax = axes[1]
    ax.plot(x, auc, "o-", color=RAMP[2], lw=2, ms=7)
    ax.set_ylabel("macro one-vs-rest AUC"); ax.set_title("AUC", loc="left", fontweight="bold")
    for xi, a in zip(x, auc):
        ax.annotate(f"{a:.4f}", (xi, a), xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=8.5)

    ax = axes[2]
    ax.bar(x, mins, color=RAMP, width=0.6)
    ax.set_ylabel("minutes / 12 epochs")
    ax.set_title("training cost", loc="left", fontweight="bold")
    for xi, v in zip(x, mins):
        ax.annotate(f"{v:.0f}", (xi, v), xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=8.5)
    ax.set_ylim(0, max(mins) * 1.42)
    ax.annotate("2.1× the cost of m=8,\nno significant gain",
                (0.5, 0.93), xycoords="axes fraction", ha="center", va="top",
                fontsize=8.5, color=INK, fontweight="bold")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([f"m={m}" + ("\n(dense)" if m == 64 else f"\n{64//m} balls")
                            for m in MS])
        ax.set_xlabel("level-0 ball size")
        tidy(ax, "y")

    fig.suptitle("S11 · level-0 ball size · 80k train / 10k test · 12 epochs · "
                 "2.21M params at every m",
                 x=0.008, ha="left", fontsize=11, fontweight="bold")
    fig.text(0.008, -0.06, "Error bars are binomial on 10,000 test jets. Paired McNemar: "
             "m=64 vs m=8 p=0.295 (not significant); m=32 is the only arm that differs "
             "significantly from the rest (p=0.044 vs m=8, p=0.002 vs m=64).",
             fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    p = os.path.join(OUT, "ballsize_summary.png")
    fig.savefig(p, dpi=160, bbox_inches="tight"); plt.close(fig)
    return p


def ballsize_roc():
    rs = _sweep()
    names = list(rs[0]["roc"])
    fig, axes = plt.subplots(3, 3, figsize=(11, 9.2), sharex=True)
    for ax, name in zip(axes.ravel(), names):
        ceil = rs[0]["roc"][name]["n_bkg"]
        for r, m, col in zip(rs, MS, RAMP):
            c = r["roc"][name]
            eff = np.array(c["eff"]); eps = np.array(c["eps_bkg"])
            rej = np.where(eps > 0, 1.0 / np.clip(eps, 1e-12, None), ceil)
            ax.plot(eff, rej, color=col, lw=1.9, label=f"m={m}")
        ax.axhline(ceil, color=MUTED, ls="--", lw=1)
        ax.set_yscale("log"); ax.set_ylim(1, ceil * 2.2); ax.set_xlim(0, 1)
        aucs = "  ".join(f"{r['roc'][name]['auc']:.4f}" for r in rs)
        ax.set_title(f"{name} vs QCD", loc="left", fontweight="bold")
        ax.annotate(f"AUC  {aucs}", (0.03, 0.04), xycoords="axes fraction",
                    fontsize=7.5, family="monospace", color=INK,
                    bbox=dict(fc="white", ec=GRID, lw=0.7, pad=2.5))
        tidy(ax)
    for ax in axes[-1]:
        ax.set_xlabel("signal efficiency  $\\varepsilon_S$")
    for ax in axes[:, 0]:
        ax.set_ylabel("background rejection  $1/\\varepsilon_B$")
    axes[0, 0].legend(frameon=False, fontsize=8, loc="lower left",
                      bbox_to_anchor=(0.0, 0.12), ncol=2)
    fig.suptitle("S11 · ROC by level-0 ball size · m=64 is a single ball, "
                 "i.e. full dense attention",
                 x=0.008, ha="left", fontsize=11, fontweight="bold")
    fig.text(0.008, -0.015, "AUC values in each panel are listed m=8, 16, 32, 64. The dashed "
             "line is the ~1,000-jet measurement ceiling. Curves are largely "
             "indistinguishable: locality costs essentially nothing in tagging power.",
             fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(OUT, "ballsize_roc.png")
    fig.savefig(p, dpi=160, bbox_inches="tight"); plt.close(fig)
    return p



# ---------------------------------------------------------------- 5. S11 ----
ABL = [("logpolar", "log-polar tree", 0.192),
       ("noD", "no distance bias (Eq. 10)", 0.580),
       ("noU", "no ParT U bias", 0.823),
       ("coarse", "readout: bottleneck only", 0.612),
       ("rot0", "no tree rotation", 0.381),
       ("minimal", "no U + no dist + no rotation", 0.282),
       ("fine", "readout: constituents only", 0.024)]


def ablations():
    base = json.load(open(os.path.join(RES, "erwin_m8_result.json")))
    rows = []
    for tag, label, p in ABL:
        r = json.load(open(os.path.join(RES, f"erwin_{tag}_result.json")))
        rows.append((label, r["accuracy"] - base["accuracy"], p,
                     r["train_minutes"] / base["train_minutes"] - 1))
    MDE = 0.006

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.4),
                                  gridspec_kw={"width_ratios": [1.65, 1]})
    ypos = np.arange(len(rows))
    ax.axvspan(-MDE, MDE, color=GRID, alpha=0.75, zorder=0)
    ax.axvline(0, color=MUTED, lw=1.2, zorder=1)
    for i, (label, d, p, _) in enumerate(rows):
        sig = p < 0.05
        ax.barh(i, d, color=ORANGE if sig else BLUE, height=0.62, zorder=2)
        ax.annotate(f"{d:+.4f}   p={p:.3f}" + ("  *" if sig else ""),
                    (d, i), xytext=(7 if d >= 0 else -7, 0),
                    textcoords="offset points", va="center",
                    ha="left" if d >= 0 else "right", fontsize=8.5,
                    fontweight="bold" if sig else "normal")
    ax.set_yticks(ypos); ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("change in test accuracy vs baseline")
    ax.set_xlim(-0.016, 0.012)
    ax.set_title("S11 ablations — paired McNemar, 10,000 test jets\n"
                 "shaded band = below the ±0.006 detection limit",
                 loc="left", fontweight="bold", fontsize=9.5)
    tidy(ax, "x")

    ax2.barh(ypos, [r[3] * 100 for r in rows], color=RAMP[1], height=0.62)
    ax2.axvline(0, color=MUTED, lw=1.2)
    for i, r in enumerate(rows):
        ax2.annotate(f"{r[3]*100:+.0f}%", (r[3] * 100, i),
                     xytext=(6 if r[3] >= 0 else -6, 0), textcoords="offset points",
                     va="center", ha="left" if r[3] >= 0 else "right", fontsize=8.5)
    ax2.set_yticks(ypos); ax2.set_yticklabels([])
    ax2.set_xlabel("change in training time")
    ax2.set_xlim(-32, 18)
    ax2.set_title("cost", loc="left", fontweight="bold")
    tidy(ax2, "x")

    fig.text(0.008, -0.03, "Only the constituents-only readout differs significantly from the "
             "baseline. Everything else — including ParT's interaction bias — sits inside the "
             "noise floor, individually and combined.", fontsize=7.5, color=MUTED)
    fig.tight_layout()
    p_ = os.path.join(OUT, "ablations.png")
    fig.savefig(p_, dpi=160, bbox_inches="tight"); plt.close(fig)
    return p_



# --------------------------------------------------------------- 6. FLOPs ---
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]


def flops():
    d = json.load(open(os.path.join(RES, "flops.json")))
    phys = set(d.get("physical", [32, 64, 128]))
    L = [r["L"] for r in d["matched"]]
    part = [r["part"] / 1e9 for r in d["matched"]]
    match = [r["erwin"] / 1e9 for r in d["matched"]]
    prod = [r["erwin"] / 1e9 for r in d["production"]]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.5),
                                  gridspec_kw={"width_ratios": [1.35, 1]})

    # Shade the region real JetClass jets actually occupy.
    ax.axvspan(32, 128, color=GRID, alpha=0.55, zorder=0)
    ax.axvline(39, color=MUTED, ls=":", lw=1.2, zorder=1)
    ax.annotate("median jet\n39 constituents", (39, 3.6), fontsize=7.5, color=MUTED,
                ha="center", va="top")
    ax.annotate("real jets live here", (64, 6.4), fontsize=8.5, color=INK,
                ha="center", fontweight="bold")
    ax.annotate("synthetic — no JetClass jet\nis this large", (300, 6.4), fontsize=8,
                color=MUTED, ha="center")

    ax.plot(L, part, "-o", color=ORANGE, lw=2.2, ms=6, label="ParT", zorder=3)
    ax.plot(L, match, "-o", color=BLUE, lw=2.2, ms=6, zorder=3,
            label="ErwinParTv2 (matched L)")
    ax.plot(L, prod, "--o", color=BLUE, lw=1.8, ms=4.5, alpha=0.6, zorder=3,
            label="ErwinParTv2 (production, fixed 64 slots)")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("constituents per jet  L")
    ax.set_ylabel("GFLOPs  (forward, batch 1)")
    ax.set_ylim(0.06, 9)
    ax.set_title("forward FLOPs — ratios over the physical range, not the tail",
                 loc="left", fontweight="bold", fontsize=10)
    for xi, a, b in zip(L, part, match):
        ax.annotate(f"{a/b:.1f}×", (xi, (a * b) ** 0.5), ha="center", fontsize=8,
                    color=INK if xi in phys else MUTED,
                    fontweight="bold" if xi in phys else "normal")
    ax.annotate("$L^{1.19}$", (L[-1], part[-1]), xytext=(-8, 9), textcoords="offset points",
                ha="right", color=ORANGE, fontsize=9.5, fontweight="bold")
    ax.annotate("$L^{1.03}$", (L[-1], match[-1]), xytext=(-8, -17),
                textcoords="offset points", ha="right", color=BLUE, fontsize=9.5,
                fontweight="bold")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    tidy(ax)

    bd = d["breakdown"]
    labels = ["attention blocks", "pair embed (U)", "input embed", "class blocks", "pooling"]
    series = [
        [bd["ParT"]["particle blocks"] / 1e9, bd["ErwinParTv2"]["ball blocks"] / 1e9],
        [bd["ParT"]["pair embed (dense U)"] / 1e9, bd["ErwinParTv2"]["pair embed (ball U)"] / 1e9],
        [bd["ParT"]["input embed"] / 1e9, bd["ErwinParTv2"]["input embed"] / 1e9],
        [bd["ParT"]["class blocks"] / 1e9, bd["ErwinParTv2"]["class blocks"] / 1e9],
        [0.0, bd["ErwinParTv2"]["pooling"] / 1e9],
    ]
    x = np.arange(2); bottom = np.zeros(2)
    tot = [bd["ParT"]["total"] / 1e9, bd["ErwinParTv2"]["total"] / 1e9]
    for vals, lab, col in zip(series, labels, CAT):
        ax2.bar(x, vals, bottom=bottom, color=col, width=0.55, label=lab)
        for xi, (v, b0) in enumerate(zip(vals, bottom)):
            if v / tot[xi] > 0.14:
                ax2.annotate(f"{100*v/tot[xi]:.0f}%", (xi, b0 + v / 2), ha="center",
                             va="center", fontsize=8, color="white", fontweight="bold")
        bottom += np.array(vals)
    for xi, t in enumerate(bottom):
        ax2.annotate(f"{t:.3f} G", (xi, t), xytext=(0, 5), textcoords="offset points",
                     ha="center", fontsize=9, fontweight="bold")
    ax2.set_xticks(x); ax2.set_xticklabels(["ParT\nL=128", "ErwinParTv2\n64 slots"])
    ax2.set_ylabel("GFLOPs"); ax2.set_ylim(0, bottom.max() * 1.2)
    ax2.set_title("where the FLOPs go", loc="left", fontweight="bold")
    ax2.legend(frameon=False, fontsize=8, loc="upper right")
    tidy(ax2, "y")

    fig.text(0.008, -0.10,
             "Bold ratios are inside the range real jets occupy; grey ones are extrapolation.\n"
             "Over 32–128 the exponents are ParT L$^{1.19}$ vs ErwinParTv2 L$^{1.03}$ — near-linear "
             "against mildly superlinear, worth 1.5–1.9×, not an order of magnitude.\n"
             "The dashed production curve sits ABOVE matched-L at L=32: a 32-particle jet padded to "
             "64 slots does more work than it needs — the 36.7% cost of a fixed sequence length.",
             fontsize=7.5, color=MUTED, linespacing=1.6)
    fig.tight_layout()
    p_ = os.path.join(OUT, "flops.png")
    fig.savefig(p_, dpi=160, bbox_inches="tight"); plt.close(fig)
    return p_



# ------------------------------------------------- 7. why L=64 (multiplicity) --
def multiplicity():
    d = json.load(open(os.path.join(RES, "multiplicity.json")))
    tr = json.load(open(os.path.join(RES, "truncation.json")))
    edges = np.array(d["edges"]); counts = np.array(d["counts"])
    L = 64

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.2),
                                  gridspec_kw={"width_ratios": [1.5, 1]})

    lo = edges[:-1]
    keep = (edges[1:] <= L)
    ax.bar(lo[keep], counts[keep], width=7, align="edge", color=BLUE, label=f"kept (L={L})")
    ax.bar(lo[~keep], counts[~keep], width=7, align="edge", color=BLUE, alpha=0.32,
           label="truncated to the 64 hardest")
    ax.axvline(L, color=INK, ls="--", lw=1.5)
    ax.annotate(f"L = {L}", (L + 3, counts.max() * 0.93), fontsize=11, fontweight="bold")
    ax.annotate(f"{d['frac_le_64']:.1%} of jets fit", (L + 3, counts.max() * 0.85),
                fontsize=9, color=MUTED)
    ax.axvline(d["median"], color=MUTED, ls=":", lw=1.2)
    ax.annotate(f"median {d['median']}", (d["median"], counts.max() * 1.06),
                fontsize=9, color=MUTED, ha="center")
    ax.set_ylim(0, counts.max() * 1.16)
    ax.set_xlabel("constituents per jet")
    ax.set_ylabel("jets")
    ax.set_xlim(0, 144)
    ax.set_title(f"jet multiplicity — all {d['n_jets']:,} jets, class-balanced",
                 loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    tidy(ax, "y")

    r64, r128 = tr["rows"]["64"], tr["rows"]["128"]
    ax.annotate(f"at L=64:  {r64['pt']:.3%} of jet $p_T$ lost,  {r64['pad']:.0%} padded slots\n"
                f"at L=128: {r128['pt']:.3%} lost,  {r128['pad']:.0%} padded slots",
                (0.62, 0.60), xycoords="axes fraction", fontsize=8.5, color=INK,
                bbox=dict(fc="white", ec=GRID, lw=0.7, pad=4))

    cls = list(tr["per_class_trunc64"])
    vals = [tr["per_class_trunc64"][c] * 100 for c in cls]
    order = np.argsort(vals)
    y = np.arange(len(cls))
    cols = [ORANGE if vals[i] > 10 else BLUE for i in order]
    ax2.barh(y, [vals[i] for i in order], color=cols, height=0.62)
    ax2.set_yticks(y); ax2.set_yticklabels([cls[i] for i in order], fontsize=9)
    ax2.set_xlabel("% of jets truncated at L=64")
    ax2.set_title("truncation is class-biased", loc="left", fontweight="bold")
    for i, v in enumerate([vals[j] for j in order]):
        ax2.annotate(f"{v:.1f}%", (v, i), xytext=(4, 0), textcoords="offset points",
                     va="center", fontsize=8.5)
    ax2.set_xlim(0, 29)
    tidy(ax2, "x")

    fig.text(0.008, -0.05, "Truncation is $p_T$-ordered, so the discarded constituents are the "
             "softest: 1.4% of particles but only 0.115% of jet $p_T$. Hgg and Tbqq lose their "
             "tails most often — the L=128 control shows their AUC is unchanged regardless.",
             fontsize=7.5, color=MUTED)
    fig.tight_layout()
    p_ = os.path.join(OUT, "multiplicity.png")
    fig.savefig(p_, dpi=160, bbox_inches="tight"); plt.close(fig)
    return p_


# ------------------------------------------------------- 8. the hierarchy ----
def hierarchy():
    """Each level is laid out in ITS OWN ball grouping.

    An earlier version positioned every level on the level-0 slot grid, so the
    level-0 ball gaps bled through and level 1's four balls of eight read as
    eight groups of four. Levels are independent partitions and must be drawn
    that way.
    """
    from matplotlib.patches import Rectangle

    W, GAP, H = 64.0, 1.4, 0.62
    levels = [(64, 8, "L0", "64 constituent slots · 8 balls of 8", 0.40),
              (32, 4, "L1", "32 subjets · 4 balls of 8", 0.66),
              (16, 1, "L2", "16 subjets · ONE ball — global", 1.0)]
    YS = [3.0, 1.65, 0.3]

    fig, ax = plt.subplots(figsize=(12.6, 4.2))
    for (n, nb, lab, sub, alpha), y in zip(levels, YS):
        avail = W - (nb - 1) * GAP
        pitch = avail / n
        per = n // nb
        for b in range(nb):
            bx = b * (per * pitch + GAP)
            ax.add_patch(Rectangle((bx - 0.30, y - 0.18), per * pitch + 0.60, H + 0.36,
                                   fc=GRID, ec=MUTED, lw=0.8, zorder=1))
        for k in range(n):
            b, j = divmod(k, per)
            x = b * (per * pitch + GAP) + j * pitch
            ax.add_patch(Rectangle((x, y), pitch * 0.86, H, fc=BLUE, alpha=alpha,
                                   ec="none", zorder=2))
        ax.text(-1.6, y + H / 2 + 0.11, lab, fontsize=13, fontweight="bold",
                ha="right", va="center", color=INK)
        ax.text(-1.6, y + H / 2 - 0.21, sub, fontsize=9, ha="right", va="center", color=MUTED)

    # Pooling annotations sit in the clear band between rows, not over the plates.
    for ytop, ybot in ((YS[0], YS[1]), (YS[1], YS[2])):
        mid = (ytop + ybot + H) / 2
        ax.annotate("", xy=(1.6, ybot + H + 0.22), xytext=(1.6, ytop - 0.22),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.5))
        ax.text(3.2, mid, "BallPooling stride 2 — children concatenated, "
                "their 4-momenta summed into a subjet",
                fontsize=9, color=MUTED, va="center")

    ax.text(0, YS[2] - 0.62, "the bottleneck is a single ball, so every subjet attends to every "
            "other — this is what carries the global information",
            fontsize=9.5, color=INK, fontweight="bold")
    ax.set_xlim(-15.5, W + 0.8)
    ax.set_ylim(-0.95, YS[0] + H + 0.55)
    ax.axis("off")
    ax.set_title("ErwinParTv2 — ball-tree coarsening; attention is local within each ball",
                 loc="left", fontweight="bold", fontsize=11.5)
    fig.tight_layout()
    p_ = os.path.join(OUT, "hierarchy.png")
    fig.savefig(p_, dpi=160, bbox_inches="tight"); plt.close(fig)
    return p_



# ------------------------------------------- 9. L=128 truncation control -----
def l128_control():
    d = json.load(open(os.path.join(RES, "l128_control.json")))
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.4),
                                  gridspec_kw={"width_ratios": [1.15, 1]})

    # -- per-class AUC, ordered by how often L=64 truncates that class --------
    cls = sorted(d["per_class"], key=lambda c: -d["per_class"][c]["trunc"])
    y = np.arange(len(cls))
    a = [d["per_class"][c]["auc64"] for c in cls]
    b = [d["per_class"][c]["auc128"] for c in cls]
    for i, (u, v) in enumerate(zip(a, b)):
        ax.plot([u, v], [i, i], color=GRID, lw=2.5, zorder=1, solid_capstyle="round")
    ax.scatter(a, y, s=46, color=BLUE, zorder=3, label="L=64")
    ax.scatter(b, y, s=46, color=ORANGE, zorder=3, label="L=128 (no truncation)")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{c}   {d['per_class'][c]['trunc']:.0%} cut" for c in cls], fontsize=9)
    for lbl, c in zip(ax.get_yticklabels(), cls):
        if d["per_class"][c]["trunc"] > 0.10:
            lbl.set_color(ORANGE); lbl.set_fontweight("bold")
    ax.set_xlabel("AUC vs QCD"); ax.set_xlim(0.945, 1.004)
    ax.invert_yaxis()
    ax.set_title("per-class AUC — ordered by how often L=64 truncates",
                 loc="left", fontweight="bold", fontsize=10)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left", bbox_to_anchor=(0.0, 0.02))
    tidy(ax, "x")

    # -- accuracy split by whether L=64 actually truncated the jet ------------
    keys = ["untruncated", "truncated", "overall"]
    names = [f"jets L=64 keeps whole\nn ≤ 64  ({d['untruncated']['n']:,} jets)",
             f"jets L=64 truncates\nn > 64  ({d['truncated']['n']} jets)",
             f"all test jets\n({d['overall']['n']:,})"]
    x = np.arange(3); w = 0.36
    va = [d[k]["a"] for k in keys]; vb = [d[k]["b"] for k in keys]
    ax2.bar(x - w/2, va, w, color=BLUE, label="L=64")
    ax2.bar(x + w/2, vb, w, color=ORANGE, label="L=128")
    for i, k in enumerate(keys):
        top = max(va[i], vb[i])
        sig = d[k]["p"] < 0.05
        ax2.annotate(f"{d[k]['d']:+.4f}\np = {d[k]['p']:.3f}", (i, top + 0.012),
                     ha="center", fontsize=8.5, color=ORANGE if sig else MUTED,
                     fontweight="bold" if sig else "normal")
        for xx, vv in ((i - w/2, va[i]), (i + w/2, vb[i])):
            ax2.annotate(f"{vv:.4f}", (xx, vv), xytext=(0, -14),
                         textcoords="offset points", ha="center", fontsize=8, color="white")
    ax2.set_xticks(x); ax2.set_xticklabels(names, fontsize=8.5)
    ax2.set_ylabel("accuracy"); ax2.set_ylim(0.55, 0.85)
    ax2.set_title("the effect is confined to the jets that get truncated",
                  loc="left", fontweight="bold", fontsize=10)
    ax2.legend(frameon=False, fontsize=8.5, loc="upper left")
    tidy(ax2, "y")

    fig.text(0.008, -0.115,
             f"L=128 truncates nothing but costs {d['cost']['min128']/d['cost']['min64']:.1f}× the "
             f"training time and {d['cost']['gflops128']/d['cost']['gflops64']:.1f}× the FLOPs.\n"
             "On jets L=64 keeps whole the two models are IDENTICAL (p = 0.971); on the 6.3% it "
             "truncates, L=128 is 3.7 points better; over the whole test set that washes out to "
             "+0.0023 (p = 0.449).\n"
             "Hgg loses its tail in 24% of jets and its AUC still moves by 0.0002 — truncation is "
             "not why Hgg is hard.", fontsize=7.5, color=MUTED, linespacing=1.6)
    fig.tight_layout()
    p_ = os.path.join(OUT, "l128_control.png")
    fig.savefig(p_, dpi=160, bbox_inches="tight"); plt.close(fig)
    return p_



# ------------------------------------------------- 10. seed reproducibility --
# Numbers from the paired McNemar runs (tests/s11_ablations.sh + s11_seedrepeat.sh).
SEEDS = [("no ParT U bias",            +0.0008, 0.823, +0.0047, 0.132),
         ("no U + no dist + no rot",   -0.0036, 0.282, -0.0048, 0.153),
         ("log-polar tree",            +0.0040, 0.192, -0.0008, 0.819)]


def seed_repeat():
    MDE = 0.006
    fig, ax = plt.subplots(figsize=(9.6, 3.4))
    y = np.arange(len(SEEDS))
    ax.axvspan(-MDE, MDE, color=GRID, alpha=0.75, zorder=0)
    ax.axvline(0, color=MUTED, lw=1.2, zorder=1)
    for i, (lab, d0, p0, d1, p1) in enumerate(SEEDS):
        ax.plot([d0, d1], [i, i], color=GRID, lw=2.5, zorder=2, solid_capstyle="round")
        ax.scatter([d0], [i], s=70, color=BLUE, zorder=3, label="seed 0" if i == 0 else None)
        ax.scatter([d1], [i], s=70, color=ORANGE, zorder=3, label="seed 1" if i == 0 else None)
        flip = d0 * d1 < 0
        ax.annotate(f"p={p0:.2f}  /  p={p1:.2f}" + ("   sign flips" if flip else ""),
                    (0.0092, i), fontsize=8.5, va="center",
                    color=ORANGE if flip else MUTED,
                    fontweight="bold" if flip else "normal")
    ax.set_yticks(y); ax.set_yticklabels([s_[0] for s_ in SEEDS], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(-0.011, 0.0235)
    ax.set_xlabel("change in test accuracy vs baseline (paired McNemar, 10,000 jets)")
    ax.set_title("every conclusion reproduces on a second seed\n"
                 "shaded = below the ±0.006 detection limit",
                 loc="left", fontweight="bold", fontsize=10)
    ax.legend(frameon=False, fontsize=9, loc="upper right", ncol=2)
    tidy(ax, "x")
    fig.text(0.008, -0.10,
             "Seed changes the data split as well as initialisation, so the two seeds use different "
             "test jets — the comparison is baseline-vs-ablation WITHIN each seed.\n"
             "Both architectural nulls hold twice with the same sign. The log-polar tree, the only "
             "arm that looked promising at seed 0, flips sign — it was noise.",
             fontsize=7.5, color=MUTED, linespacing=1.6)
    fig.tight_layout()
    p_ = os.path.join(OUT, "seed_repeat.png")
    fig.savefig(p_, dpi=160, bbox_inches="tight"); plt.close(fig)
    return p_


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for fn in (training_curves, roc_curves, efficiency,
               ballsize_summary, ballsize_roc, ablations, flops,
               multiplicity, hierarchy, l128_control,
               seed_repeat):
        print("wrote", os.path.normpath(fn()))
