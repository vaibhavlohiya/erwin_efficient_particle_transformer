"""Cluster 10% runs vs ParT and the earlier ErwinParT (v1) baseline: figures.

Reads, never trains or evaluates:
  tests/s10_results/cluster_eval.json   Cluster/scripts/eval_runs.py on the 2M-jet
                                        test predictions of each finished run
  tests/s10_results/cluster_flops.json  tests/flops_cluster_models.py
  tests/s10_results/erwin_m*_result.json  the local ball-size sweep (locality)
  tests/s10_results/test_multiplicity.json  constituent counts (jet_nparticles,
                                        a weaver observer) of the same 2M test
                                        jets, read from the prediction files
  REFERENCE below                       v1 and published numbers, copied from
                                        ../Plots/ErwinParT.py (their only record)

Writes plots/new_comparison_plots/*.png and summary.csv there.

    python tests/plot_cluster_comparison.py
"""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "s10_results")
OUT = os.path.join(HERE, "..", "plots", "new_comparison_plots")
WORKSPACE = os.path.abspath(os.path.join(HERE, "..", ".."))

INK, MUTED, GRID = "#1f1f22", "#6b6a74", "#e2e1e6"
# Categorical slots 1-4 of the dataviz reference palette, in fixed order; markers
# and direct labels carry identity too (slot 4 is below 3:1 contrast on white).
MODELS = {   # key -> label, colour, marker, run tags in cluster_eval.json, seq length
    "v2":   ("ErwinParTv2", "#2a78d6", "o", ["v2-10pct-a", "v2-10pct-b"], 64),
    "lund": ("ErwinParTv2 + Lund", "#eb6834", "s", ["lund-10pct-a", "lund-10pct-b"], 64),
    "part": ("ParT", "#1baf7a", "^", ["part-10pct"], 128),
    "v1":   ("ErwinParT v1", "#eda100", "D", [], 128),
}
RUN_STYLE = ["-", "--"]
# ../Plots/ErwinParT.py: ErwinParT v1 (10% JetClass; evaluation details not
# recorded) and ParT as published on full JetClass. Rejections are at 50%.
REFERENCE = {
    "v1":        {"label": "ErwinParT v1 (reported, 10%)", "accuracy": 0.825, "auc": 0.9819,
                  "rej": {"Hbb": 8000, "Hcc": 1681, "Tbqq": 5556, "Wqq": 296}},
    "part_full": {"label": "ParT (published, full JetClass)", "accuracy": 0.861, "auc": 0.9877,
                  "rej": {"Hbb": 10638, "Hcc": 4149, "Tbqq": 32787, "Wqq": 543}},
}
CLASS_TEX = {"Hbb": r"$H\to b\bar b$", "Hcc": r"$H\to c\bar c$", "Hgg": r"$H\to gg$",
             "H4q": r"$H\to 4q$", "Hqql": r"$H\to \ell\nu qq'$", "Zqq": r"$Z\to q\bar q$",
             "Wqq": r"$W\to qq'$", "Tbqq": r"$t\to bqq'$", "Tbl": r"$t\to b\ell\nu$"}
SIGNALS = list(CLASS_TEX)
PREFIX = "20261029-35e96f7-"
EPOCHS = 20

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


# ------------------------------------------------------------------ loading --
def load():
    ev = json.load(open(os.path.join(RES, "cluster_eval.json")))
    runs = {}                 # model key -> list of (tag, record) with full predictions
    for key, (_, _, _, tags, _) in MODELS.items():
        runs[key] = [(t, ev[PREFIX + t]) for t in tags if "accuracy" in ev.get(PREFIX + t, {})]
    flops = json.load(open(os.path.join(RES, "cluster_flops.json")))["models"]
    return ev, runs, flops


def mean_of(recs, f):
    return float(np.mean([f(r) for _, r in recs]))


# ------------------------------------------------------------------- 1. ROC --
def plot_roc(runs):
    fig, axes = plt.subplots(3, 3, figsize=(11.5, 10.5), sharex=True)
    for ax, cls in zip(axes.flat, SIGNALS):
        n_bkg = None
        for key, recs in runs.items():
            label, col = MODELS[key][0], MODELS[key][1]
            for k, (tag, r) in enumerate(recs):
                c = r["vs_qcd"][cls]
                eff = np.array(c["eff"]); rej = np.array([np.nan if v is None else v for v in c["rej"]])
                ax.plot(eff, rej, color=col, ls=RUN_STYLE[k], lw=1.5,
                        label=f"{label} {tag[-1]}  AUC {c['auc']:.4f}")
                n_bkg = c["n_bkg"]
        ax.axhline(n_bkg, color=MUTED, ls=":", lw=0.9)
        ax.text(0.03, n_bkg * 1.25, rf"$N_\mathrm{{QCD}}$ = {n_bkg:,}: no QCD jet passes above",
                fontsize=6.5, color=MUTED)
        ax.set_yscale("log"); ax.set_ylim(1, n_bkg * 4); ax.set_xlim(0, 1)
        ax.set_title(f"{CLASS_TEX[cls]} vs QCD", loc="left", fontsize=10)
        ax.legend(loc="lower left", fontsize=6.3, handlelength=2.2)
        grid(ax)
    for ax in axes[-1]:
        ax.set_xlabel(r"signal efficiency $\varepsilon_S$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"background rejection $1/\varepsilon_B$")
    fig.suptitle("ROC curves on the 2M-jet test set (10% JetClass, 20 epochs); "
                 "AUC is signal vs QCD with ParT's discriminant", x=0.01, ha="left", y=1.0,
                 fontsize=10, color=MUTED)
    fig.tight_layout()
    save(fig, "roc_auc.png")


# -------------------------------------------------------------- 2. accuracy --
def plot_accuracy(runs):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    rows = [(MODELS[k][0], MODELS[k][1], MODELS[k][2], recs) for k, recs in runs.items() if recs]
    rows += [(REFERENCE["v1"]["label"], MODELS["v1"][1], MODELS["v1"][2], "v1"),
             (REFERENCE["part_full"]["label"], MUTED, "^", "part_full")]
    y = np.arange(len(rows))[::-1]
    for ax, metric, xl in zip(axes, ("accuracy", "auc"), ("test accuracy", "macro one-vs-rest AUC")):
        for yi, (label, col, mk, recs) in zip(y, rows):
            if isinstance(recs, str):                       # a reported number, no runs
                v = REFERENCE[recs][metric]
                ax.scatter([v], [yi], s=60, marker=mk, facecolors="white", edgecolors=col, lw=1.6, zorder=3)
                ax.annotate(f"{v:.4f}", (v, yi), xytext=(7, -3), textcoords="offset points", fontsize=8)
                continue
            key = "accuracy" if metric == "accuracy" else "auc_macro"
            vals = [r[key] for _, r in recs]
            ax.plot([min(vals), max(vals)], [yi, yi], color=col, lw=1.5, alpha=0.7)
            ax.scatter(vals, [yi] * len(vals), s=22, facecolors="white", edgecolors=col, lw=1.2, zorder=3)
            ax.scatter([np.mean(vals)], [yi], s=64, marker=mk, color=col, edgecolors="white", lw=1.5, zorder=4)
            ax.annotate(f"{np.mean(vals):.4f}" + (f"  (n={len(vals)})" if len(vals) > 1 else "  (1 run)"),
                        (max(vals), yi), xytext=(7, -3), textcoords="offset points", fontsize=8)
        ax.set_xlabel(xl)
        grid(ax, "x"); ax.tick_params(axis="y", length=0)
        lo, hi = ax.get_xlim(); ax.set_xlim(lo, hi + (hi - lo) * 0.35)
    axes[0].set_yticks(y, [r[0] for r in rows])
    for ax in axes:                   # the two reported rows sit at y = 1, 0
        ax.axhline(1.5, color=MUTED, lw=0.8, ls="--")
    titled(axes[0], "Classification performance",
           "filled: mean over runs; open circles: single runs; open shapes: reported numbers")
    titled(axes[1], "", "below the line: other training regimes, not directly comparable")
    save(fig, "accuracy.png")


# ------------------------------------------------------------- 3. rejection --
def rej_table(runs):
    """{model: {class: (mean rejection, is_lower_bound)}} at ParT's working points."""
    t = {}
    for key, recs in runs.items():
        if not recs:
            continue
        t[key] = {}
        for cls in SIGNALS:
            vals, lb = [], False
            for _, r in recs:
                c = r["vs_qcd"][cls]
                if c["rej_wp"] is None:
                    vals.append(c["n_bkg"]); lb = True      # 0 of N passed: > N
                else:
                    vals.append(c["rej_wp"])
            t[key][cls] = (float(np.exp(np.mean(np.log(vals)))), lb)
    return t


def plot_rejection(runs):
    t = rej_table(runs)
    wp = {cls: next(iter(t.values())) and runs[next(iter(t))][0][1]["vs_qcd"][cls]["wp"] for cls in SIGNALS}
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    y = np.arange(len(SIGNALS))[::-1]
    keys = list(t)
    off = np.linspace(-0.22, 0.22, len(keys) + 2)
    for j, key in enumerate(keys):
        label, col, mk = MODELS[key][:3]
        for yi, cls in zip(y, SIGNALS):
            v, lb = t[key][cls]
            ax.scatter([v], [yi + off[j]], s=46, marker=mk, color=col if not lb else "white",
                       edgecolors=col, lw=1.4, zorder=3, label=label if cls == SIGNALS[0] else None)
            if lb:
                ax.annotate("", xy=(v * 1.8, yi + off[j]), xytext=(v * 1.05, yi + off[j]),
                            arrowprops=dict(arrowstyle="->", color=col, lw=1))
    for j, ref in enumerate(("v1", "part_full")):
        R = REFERENCE[ref]
        col = MODELS["v1"][1] if ref == "v1" else MUTED
        for yi, cls in zip(y, SIGNALS):
            if cls in R["rej"]:
                ax.scatter([R["rej"][cls]], [yi + off[len(keys) + j]], s=42, marker="D" if ref == "v1" else "^",
                           facecolors="white", edgecolors=col, lw=1.4, zorder=3,
                           label=R["label"] if cls == "Hbb" else None)
    ax.set_xscale("log")
    ax.set_yticks(y, [f"{CLASS_TEX[c]}  @{wp[c] * 100:g}%" for c in SIGNALS])
    ax.set_xlabel(r"background rejection $1/\varepsilon_B$ vs QCD at the working point")
    grid(ax, "x"); ax.tick_params(axis="y", length=0)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    titled(ax, "Background rejection per jet type",
           "ParT's working points; hollow + arrow: no QCD jet passed, a lower bound; mean over runs (geometric)")
    save(fig, "rejection_per_class.png")

    # overall: geometric mean over the classes every model resolves (no lower bounds)
    ok = [c for c in SIGNALS if not any(t[k][c][1] for k in keys)]
    fig, ax = plt.subplots(figsize=(6.4, 0.6 * len(keys) + 1.4))
    yy = np.arange(len(keys))[::-1]
    for yi, key in zip(yy, keys):
        label, col, mk = MODELS[key][:3]
        per_run = []
        for _, r in runs[key]:
            per_run.append(np.exp(np.mean([np.log(r["vs_qcd"][c]["rej_wp"]) for c in ok])))
        ax.plot([min(per_run), max(per_run)], [yi, yi], color=col, lw=1.5, alpha=0.7)
        ax.scatter(per_run, [yi] * len(per_run), s=22, facecolors="white", edgecolors=col, lw=1.2, zorder=3)
        g = float(np.exp(np.mean(np.log(per_run))))
        ax.scatter([g], [yi], s=64, marker=mk, color=col, edgecolors="white", lw=1.5, zorder=4)
        ax.annotate(f"{g:,.0f}", (max(per_run), yi), xytext=(7, -3), textcoords="offset points", fontsize=8.5)
    ax.set_yticks(yy, [MODELS[k][0] for k in keys])
    ax.set_xlabel("geometric-mean background rejection")
    grid(ax, "x"); ax.tick_params(axis="y", length=0)
    allv = [np.exp(np.mean([np.log(r["vs_qcd"][c]["rej_wp"]) for c in ok])) for k in keys for _, r in runs[k]]
    span = max(allv) - min(allv) or max(allv) * 0.05
    ax.set_xlim(min(allv) - span * 0.3, max(allv) + span * 0.45)
    ax.set_ylim(yy.min() - 0.6, yy.max() + 0.6)
    titled(ax, "Overall background rejection",
           "geometric mean over " + ", ".join(ok) + " (classes with QCD jets left to count)")
    save(fig, "rejection_overall.png")
    return t, ok


# ------------------------------------------------------------------ 4. FLOPs --
def plot_flops(flops):
    order = [("ParT", "part"), ("ErwinParT v1", "v1"), ("ErwinParTv2", "v2"), ("ErwinParTv2+Lund", "lund")]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.2), gridspec_kw={"width_ratios": [1.15, 1]})
    y = np.arange(len(order))[::-1]
    ax = axes[0]
    vals = [flops[n]["gflops_per_jet"] for n, _ in order]
    ax.barh(y, vals, height=0.6, color=[MODELS[k][1] for _, k in order], edgecolor="white", lw=2)
    for yi, v in zip(y, vals):
        ax.text(v, yi, f"  {v:.3f} GFLOPs  ({v * 1e3:.0f} MFLOPs)  {vals[0] / v:.1f}$\\times$ less" if v != vals[0]
                else f"  {v:.3f} GFLOPs  ({v * 1e3:.0f} MFLOPs)", va="center", fontsize=8)
    ax.set_yticks(y, [MODELS[k][0] for _, k in order]); ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, max(vals) * 1.9); ax.set_xlabel("forward GFLOPs per jet")
    grid(ax, "x")
    titled(ax, "Inference cost", "FlopCounterMode, 512 real jets; Lund clustering adds < 0.003 GFLOPs")
    ax = axes[1]
    tr = [(n, k) for n, k in order if "train_pflops" in flops[n]]
    yt = np.arange(len(tr))[::-1]
    tv = [flops[n]["train_pflops"] for n, _ in tr]
    ax.barh(yt, tv, height=0.6, color=[MODELS[k][1] for _, k in tr], edgecolor="white", lw=2)
    for yi, v in zip(yt, tv):
        ax.text(v, yi, f"  {v:.1f} PFLOPs", va="center", fontsize=8)
    ax.set_yticks(yt, [MODELS[k][0] for _, k in tr]); ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, max(tv) * 1.4); ax.set_xlabel(r"training compute [PFLOPs = $10^{15}$]")
    grid(ax, "x")
    titled(ax, "Training cost", r"3$\times$ forward $\times$ 20.5M jets (10%, 20 epochs); v1's run length unknown")
    fig.tight_layout()
    save(fig, "flops.png")


# --------------------------------------------------------- 5. training time --
def plot_training_time(runs):
    bars = []
    for key, recs in runs.items():
        for k, (tag, r) in enumerate(recs):
            # Epochs 0-19 only: lund-a's log also holds epochs 20-29 of the abandoned
            # 50-epoch schedule, which it rolled back from (last occurrence = the run used).
            ep = {e: v for e, v in r.get("epochs", {}).items() if int(e) < EPOCHS}
            tr = sum(e.get("train", 0) for e in ep.values())
            va = sum(e.get("val", 0) for e in ep.values())
            per = [e["train"] for e in ep.values() if "train" in e]
            bars.append((f"{MODELS[key][0]} {tag[-1]}", MODELS[key][1], tr, va, per, len(ep)))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 0.55 * len(bars) + 1.6), sharey=True,
                             gridspec_kw={"width_ratios": [1.2, 1]})
    y = np.arange(len(bars))[::-1]
    ax = axes[0]
    for yi, (lab, col, tr, va, per, n) in zip(y, bars):
        ax.barh(yi, tr, height=0.6, color=col, edgecolor="white", lw=2)
        ax.barh(yi, va, left=tr, height=0.6, color=col, alpha=0.4, edgecolor="white", lw=2)
        ax.text(tr + va, yi, f"  {tr + va:.0f} min  ({n} epochs)", va="center", fontsize=8)
    ax.set_yticks(y, [b[0] for b in bars]); ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, max(b[2] + b[3] for b in bars) * 1.45)
    ax.set_xlabel("minutes (solid: training, light: validation)")
    grid(ax, "x")
    titled(ax, "Training cost", "summed from train.log; excludes queueing, setup and testing")
    ax = axes[1]
    for yi, (lab, col, tr, va, per, n) in zip(y, bars):
        ax.scatter(per, np.full(len(per), yi) + np.random.default_rng(0).uniform(-0.15, 0.15, len(per)),
                   s=12, color=col, alpha=0.7)
        ax.scatter([np.median(per)], [yi], s=60, marker="|", color=INK, lw=2, zorder=4)
    ax.set_xlabel("minutes per training epoch")
    grid(ax, "x")
    titled(ax, "", "| median; epochs ran on mixed GPUs (4090, L40, 3090...)")
    fig.tight_layout()
    save(fig, "training_time.png")
    return bars


# ------------------------------------------------- 6./7. multiplicity, cuts --
def jet_counts():
    """Per-jet multiplicity and class of the 2M test jets, from per-class histograms."""
    counts = json.load(open(os.path.join(RES, "test_multiplicity.json")))["counts"]
    names = list(counts)
    n = np.concatenate([np.repeat(np.arange(len(c)), c) for c in counts.values()])
    cls = np.concatenate([np.full(sum(c), i) for i, c in enumerate(counts.values())])
    return n, cls, names


def plot_multiplicity(n, cls, names):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    bins = np.arange(0, n.max() + 2) - 0.5
    ax.hist(n, bins=bins, color="#2a78d6", alpha=0.85, edgecolor="white", lw=0.3)
    for L, lab in ((64, "ErwinParTv2 / + Lund keep 64"), (128, "ParT / v1 keep 128")):
        ax.axvline(L, color=INK, ls="--", lw=1)
        ax.text(L + 1, ax.get_ylim()[1] * 0.92, lab, fontsize=8, rotation=90, va="top")
    ax.set_xlabel("constituents per jet"); ax.set_ylabel("jets")
    grid(ax, "y")
    titled(ax, "Jet multiplicity",
           f"2M-jet test set: median {int(np.median(n))}, mean {n.mean():.1f}, "
           f"95th pct {int(np.percentile(n, 95))}, max {n.max()}")
    ax = axes[1]
    y = np.arange(len(names))[::-1]
    parts = [np.percentile(n[cls == i], [25, 50, 75]) for i in range(len(names))]
    for yi, (q1, med, q3) in zip(y, parts):
        ax.plot([q1, q3], [yi, yi], color="#2a78d6", lw=4, alpha=0.5, solid_capstyle="butt")
        ax.scatter([med], [yi], color="#2a78d6", s=30, zorder=3)
    ax.axvline(64, color=INK, ls="--", lw=1)
    ax.set_yticks(y, [CLASS_TEX.get(c, c) for c in names]); ax.tick_params(axis="y", length=0)
    ax.set_xlabel("constituents per jet")
    grid(ax, "x")
    titled(ax, "By jet type", "dot: median; bar: 25th-75th percentile; dashed: L = 64")
    fig.tight_layout()
    save(fig, "jet_multiplicity.png")


def plot_truncation(n, cls, names):
    models = [("v2", 64), ("lund", 64), ("part", 128), ("v1", 128)]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = np.arange(len(names) + 1)
    w = 0.2
    rows = {}
    for j, (key, L) in enumerate(models):
        fr = [100 * (n[cls == i] > L).mean() for i in range(len(names))] + [100 * (n > L).mean()]
        rows[key] = fr
        ax.bar(x + (j - 1.5) * w, fr, width=w, color=MODELS[key][1], edgecolor="white", lw=1,
               label=f"{MODELS[key][0]} (L = {L})")
    for xi, v in zip(x, rows["v2"]):
        ax.text(xi - 0.5 * w, v, f"{v:.1f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x, [CLASS_TEX.get(c, c) for c in names] + ["all jets"], rotation=30, ha="right")
    ax.set_ylabel("jets truncated [%]")
    grid(ax, "y")
    ax.legend(loc="upper right")
    titled(ax, "Sequence truncation",
           "2M-jet test set: share of jets with more constituents than the model keeps "
           f"(lowest-pT dropped); L = 128: {100 * (n > 128).mean():.3f}%")
    save(fig, "truncation.png")
    return rows


# --------------------------------------------------------------- 8. locality --
def plot_locality(runs):
    sweep = []
    for m in (8, 16, 32, 64):
        r = json.load(open(os.path.join(RES, f"erwin_m{m}_result.json")))
        sweep.append((m, r["accuracy"], r["auc_macro"]))
    s1 = json.load(open(os.path.join(RES, "erwin_s1_base_result.json")))
    noise_acc = abs(sweep[0][1] - s1["accuracy"]); noise_auc = abs(sweep[0][2] - s1["auc_macro"])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    ms = [s[0] for s in sweep]
    for ax, idx, yl, noise in ((axes[0], 1, "test accuracy", noise_acc), (axes[1], 2, "macro AUC", noise_auc)):
        v = [s[idx] for s in sweep]
        ax.fill_between(ms, v[0] - noise, v[0] + noise, color=GRID, alpha=0.8, lw=0,
                        label="seed-to-seed spread at m = 8")
        ax.plot(ms, v, "-o", color="#2a78d6", ms=6, label="ErwinParTv2, level-0 ball size m")
        for m, vv in zip(ms, v):
            ax.annotate(f"{vv:.4f}", (m, vv), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8)
        ax.set_xscale("log", base=2)
        ax.set_xticks(ms, [f"{m}" + ("\n(dense level 0)" if m == 64 else "") for m in ms])
        ax.set_xlim(6.5, 80)
        lo, hi = min(v + [v[0] - noise]), max(v + [v[0] + noise])
        ax.set_ylim(lo - (hi - lo) * 0.12, hi + (hi - lo) * 0.18)
        ax.set_xlabel("particles per ball at level 0"); ax.set_ylabel(yl)
        grid(ax)
    axes[0].legend(loc="lower left", fontsize=7.5)
    titled(axes[0], "Is ball locality costing accuracy?",
           "100k-jet local sweep, 12 epochs, one seed per m")
    v2, part = runs.get("v2"), runs.get("part")
    if v2 and part:
        d_acc = mean_of(part, lambda r: r["accuracy"]) - mean_of(v2, lambda r: r["accuracy"])
        d_auc = mean_of(part, lambda r: r["auc_macro"]) - mean_of(v2, lambda r: r["auc_macro"])
        note = f"10% JetClass: ParT - ErwinParTv2 = {d_acc:+.4f} acc, {d_auc:+.4f} AUC"
    else:
        note = "10% JetClass ParT vs ErwinParTv2: ParT run not finished yet"
    titled(axes[1], "", note)
    save(fig, "locality.png")
    return sweep, noise_acc, noise_auc


# ---------------------------------------------------------------- summary --
def write_summary(runs, flops, rej, bars, trunc):
    names = {"v2": "ErwinParTv2", "lund": "ErwinParTv2+Lund", "part": "ParT"}
    rows = []
    for key, recs in runs.items():
        if not recs:
            continue
        fl = flops[names[key]]
        mins = [b[2] + b[3] for b in bars if b[0].startswith(MODELS[key][0] + " ")]
        row = {"model": MODELS[key][0], "runs": len(recs),
               "accuracy": round(mean_of(recs, lambda r: r["accuracy"]), 4),
               "auc_macro": round(mean_of(recs, lambda r: r["auc_macro"]), 4),
               "gflops_per_jet": round(fl["gflops_per_jet"], 4),
               "train_pflops": round(fl.get("train_pflops", float("nan")), 1),
               "train_minutes_mean": round(float(np.mean(mins)), 0) if mins else "",
               "truncated_pct": round(trunc[key][-1], 2)}
        for cls in SIGNALS:
            v, lb = rej[key][cls]
            row[f"rej_{cls}"] = (">" if lb else "") + f"{v:.0f}"
        rows.append(row)
    path = os.path.join(OUT, "summary.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("  wrote", os.path.relpath(path, os.path.join(HERE, "..")))
    for r in rows:
        print("   ", r)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    ev, runs, flops = load()
    print("runs with full predictions:", {k: [t for t, _ in v] for k, v in runs.items()})
    plot_roc(runs)
    plot_accuracy(runs)
    rej, ok = plot_rejection(runs)
    plot_flops(flops)
    bars = plot_training_time(runs)
    n, cls, names = jet_counts()
    plot_multiplicity(n, cls, names)
    trunc = plot_truncation(n, cls, names)
    sweep, na, nu = plot_locality(runs)
    print(f"  locality: m64 - m8 = {sweep[3][1] - sweep[0][1]:+.4f} acc, {sweep[3][2] - sweep[0][2]:+.4f} AUC;"
          f" seed spread {na:.4f} acc, {nu:.4f} AUC")
    write_summary(runs, flops, rej, bars, trunc)
