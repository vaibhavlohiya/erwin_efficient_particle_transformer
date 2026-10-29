"""ErwinParTv2 against its Lund and growing-ball variants: four figures.

Reads what S10/S11 and tests/bench_variants.py already wrote; nothing here
trains, evaluates or times anything.

  tests/s10_results/<tag>_result.json   accuracy, macro AUC, per-class ROC
                                        (one file per model per seed)
  tests/s10_results/variants_bench.json FLOPs/jet and latency per model

or, instead of both, a flat table (--table results.csv|.json) with one row per
model and seed: model, seed, accuracy, auc[, flops_per_jet, latency_us_per_jet].
ROC curves need the s10 JSONs; with a table they are skipped.

  1. variants_efficiency.png   AUC against FLOPs per jet
  2. variants_roc.png          background rejection vs signal efficiency
  3. variants_latency.png      inference time per jet
  4. variants_accuracy.png     accuracy and AUC per seed, plus variants_summary.csv

Adding a model = one entry in MODELS. Axis ranges are arguments of the
plotting functions.

    python tests/plot_variants.py
    python tests/plot_variants.py --classes Hbb,Tbqq --usetex
"""
import argparse
import csv
import json
import os
from dataclasses import dataclass, field

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "s10_results")
OUT = os.path.join(HERE, "..", "plots")

# Baseline in neutral ink; variants take categorical slots 1-3 of the dataviz
# reference palette, the only slots validated for all-pairs use (scatter).
# Markers are the secondary encoding, so identity never rests on colour alone.
INK, MUTED, GRID = "#1f1f22", "#6b6a74", "#e2e1e6"
BASE, BLUE, ORANGE, AQUA = "#52514e", "#2a78d6", "#eb6834", "#1baf7a"


@dataclass
class Model:
    key: str                      # name in variants_bench.json / --table
    label: str                    # legend text (mathtext allowed)
    tags: dict                    # seed -> s10 result tag
    color: str
    marker: str
    runs: list = field(default_factory=list)   # filled by load_*
    bench: dict = field(default_factory=dict)


MODELS = [
    Model("base",  "ErwinParTv2",                        {0: "erwin_m8", 1: "erwin_s1_base"},  BASE,   "o"),
    Model("lund",  "+ Lund tokens",                      {0: "erwin_lund", 1: "erwin_s1_lund"}, BLUE,   "s"),
    Model("grow8", r"Growing balls $8\rightarrow32$",     {0: "erwin_grow8", 1: "erwin_s1_grow8"}, ORANGE, "^"),
    Model("grow2", r"Growing balls $2\rightarrow32$",     {0: "erwin_grow2", 1: "erwin_s1_grow2"}, AQUA,   "D"),
]


def style(usetex=False):
    plt.rcParams.update({
        "text.usetex": usetex,
        "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "axes.formatter.use_mathtext": True,
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 8.5,
        "axes.labelcolor": INK, "text.color": INK, "axes.edgecolor": MUTED,
        "xtick.color": INK, "ytick.color": INK, "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True, "lines.linewidth": 1.8,
        "legend.frameon": False, "figure.facecolor": "white", "savefig.facecolor": "white",
        "savefig.dpi": 200, "savefig.bbox": "tight",
    })


def titled(ax, main, sub=None):
    ax.set_title(main, loc="left", pad=16 if sub else 6)
    if sub:
        ax.text(0.0, 1.015, sub, transform=ax.transAxes, fontsize=8, color=MUTED, va="bottom")


def tidy(ax):
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


# ------------------------------------------------------------------ loading --
def load_s10(models, res_dir=RES):
    """One run per (model, seed) from the S10 result JSONs; missing seeds skipped."""
    for m in models:
        for seed, tag in sorted(m.tags.items()):
            path = os.path.join(res_dir, f"{tag}_result.json")
            if not os.path.exists(path):
                print(f"  missing {path} - {m.key} seed {seed} skipped")
                continue
            r = json.load(open(path))
            m.runs.append({"seed": seed, "accuracy": r["accuracy"], "auc": r["auc_macro"],
                           "roc": r.get("roc"), "rej50": r.get("rej50")})


def load_bench(models, path=os.path.join(RES, "variants_bench.json")):
    if not os.path.exists(path):
        print(f"  no {path} - run tests/bench_variants.py for FLOPs and latency")
        return {}
    b = json.load(open(path))
    for m in models:
        m.bench = b["variants"].get(m.key, {})
    return {k: b[k] for k in ("device", "batch", "reps")}


def load_table(models, path):
    """Flat CSV/JSON: rows of model, seed, accuracy, auc[, flops_per_jet, latency_us_per_jet]."""
    if path.endswith(".json"):
        rows = json.load(open(path))
    else:
        with open(path) as fh:
            rows = list(csv.DictReader(fh))
    by_key = {m.key: m for m in models}
    for row in rows:
        m = by_key.get(row["model"])
        if m is None:
            print(f"  row for unknown model {row['model']!r} - add it to MODELS")
            continue
        m.runs.append({"seed": int(row.get("seed", 0)), "accuracy": float(row["accuracy"]),
                       "auc": float(row["auc"]), "roc": None, "rej50": None})
        for k in ("flops_per_jet", "latency_us_per_jet"):
            if row.get(k) not in (None, ""):
                m.bench[k] = float(row[k])


def mean(m, key):
    return float(np.mean([r[key] for r in m.runs]))


# ---------------------------------------------------------------- 1. FLOPs --
def plot_efficiency(models, path, metric="auc", xlim=None, ylim=None):
    ms = [m for m in models if m.runs and "flops_per_jet" in m.bench]
    if not ms:
        return print("  efficiency: no FLOPs - skipped")
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for m in ms:
        x = m.bench["flops_per_jet"] / 1e6
        ys = [r[metric] for r in m.runs]
        ax.plot([x, x], [min(ys), max(ys)], color=m.color, lw=1.2, alpha=0.7)
        ax.scatter([x] * len(ys), ys, s=22, facecolors="white", edgecolors=m.color, lw=1.2, zorder=3)
        ax.scatter([x], [np.mean(ys)], s=64, marker=m.marker, color=m.color,
                   edgecolors="white", lw=1.5, zorder=4, label=m.label)
        left = any(o is not m and 0 < o.bench["flops_per_jet"] / 1e6 - x < 0.1 * x for o in ms)
        ax.annotate(m.label, (x, np.mean(ys)), xytext=(-8 if left else 8, -3),
                    textcoords="offset points", fontsize=8.5, color=INK,
                    ha="right" if left else "left")
    ax.set_xlabel("inference cost [MFLOPs per jet]")
    ax.set_ylabel({"auc": "macro one-vs-rest AUC", "accuracy": "accuracy"}[metric])
    titled(ax, "Performance versus computational cost",
           "filled: mean over seeds;  open: individual seeds;  FLOPs at inference")
    lo = min(m.bench["flops_per_jet"] for m in ms) / 1e6
    hi = max(m.bench["flops_per_jet"] for m in ms) / 1e6
    ax.set_xlim(xlim or (lo * 0.75, hi * 1.3))
    if ylim:
        ax.set_ylim(ylim)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    tidy(ax)
    fig.savefig(path); plt.close(fig)
    print(f"  wrote {path}")


# ------------------------------------------------------------------ 2. ROC --
def plot_roc(models, path, classes=("Hbb", "Tbqq", "Wqq", "Zqq"), seed=0, ylim=(1, 5e3)):
    ms = [(m, next((r for r in m.runs if r["seed"] == seed and r["roc"]), None)) for m in models]
    ms = [(m, r) for m, r in ms if r]
    if not ms:
        return print("  roc: no per-class ROC in the inputs - skipped")
    n = len(classes)
    fig, axes = plt.subplots(1, n, figsize=(3.3 * n, 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, cls in zip(axes, classes):
        n_bkg = None
        for m, r in ms:
            c = r["roc"][cls]
            eff, eb = np.asarray(c["eff"]), np.asarray(c["eps_bkg"])
            ok = eb > 0
            ax.plot(eff[ok], 1 / eb[ok], color=m.color, label=f"{m.label}  (AUC {c['auc']:.4f})")
            n_bkg = c.get("n_bkg", n_bkg)
        if n_bkg:   # one background jet left = the largest rejection the test split can show
            ax.axhline(n_bkg, color=MUTED, ls=":", lw=1)
            ax.text(0.98, n_bkg * 1.15, rf"$1/N_\mathrm{{QCD}}$ ceiling ({n_bkg})", fontsize=7.5,
                    color=MUTED, ha="right")
        ax.set_yscale("log"); ax.set_ylim(ylim); ax.set_xlim(0, 1)
        ax.set_xlabel(r"signal efficiency $\varepsilon_S$")
        ax.set_title(rf"{cls} vs QCD", loc="left")
        ax.legend(loc="lower left", fontsize=7, handlelength=1.4)
        tidy(ax)
    axes[0].set_ylabel(r"background rejection $1/\varepsilon_B$")
    fig.suptitle(f"ROC curves, seed {seed}, AUC per class in the legend", x=0.01, ha="left",
                 fontsize=10, color=MUTED, y=1.02)
    fig.savefig(path); plt.close(fig)
    print(f"  wrote {path}")


# -------------------------------------------------------------- 3. latency --
def plot_latency(models, path, meta=None, xlim=None):
    ms = [m for m in models if "latency_us_per_jet" in m.bench]
    if not ms:
        return print("  latency: no timings - skipped")
    fig, ax = plt.subplots(figsize=(5.2, 0.55 * len(ms) + 1.2))
    y = np.arange(len(ms))[::-1]
    vals = [m.bench["latency_us_per_jet"] / 1e3 for m in ms]
    ax.barh(y, vals, height=0.6, color=[m.color for m in ms], edgecolor="white", lw=2)
    for yi, v, m in zip(y, vals, ms):
        rel = v / vals[0]
        ax.text(v, yi, f"  {v:.2f} ms" + (f"  ({rel:.2f}$\\times$)" if m is not ms[0] else ""),
                va="center", fontsize=8.5, color=INK)
    ax.set_yticks(y, [m.label for m in ms])
    ax.set_xlabel("inference time per jet [ms]")
    ax.set_xlim(xlim or (0, max(vals) * 1.35))
    titled(ax, "Inference time", meta and
           f"{meta['device'].upper()}, batch {meta['batch']}, median of {meta['reps']} forwards")
    ax.grid(True, axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    fig.savefig(path); plt.close(fig)
    print(f"  wrote {path}")


# ------------------------------------------------------------- 4. accuracy --
def plot_accuracy(models, path, csv_path=None):
    """Dot plot, not bars: the spread is ~0.01, which a zero-based bar hides."""
    ms = [m for m in models if m.runs]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 0.55 * len(ms) + 1.3), sharey=True)
    y = np.arange(len(ms))[::-1]
    for ax, key, xl in zip(axes, ("accuracy", "auc"), ("test accuracy", "macro one-vs-rest AUC")):
        for yi, m in zip(y, ms):
            vals = [r[key] for r in m.runs]
            ax.plot([min(vals), max(vals)], [yi, yi], color=m.color, lw=1.5, alpha=0.7)
            ax.scatter(vals, [yi] * len(vals), s=22, facecolors="white", edgecolors=m.color, lw=1.2, zorder=3)
            ax.scatter([np.mean(vals)], [yi], s=60, marker=m.marker, color=m.color,
                       edgecolors="white", lw=1.5, zorder=4)
        base = mean(ms[0], key)
        ax.axvline(base, color=MUTED, ls="--", lw=0.9)
        ax.set_xlabel(xl)
        ax.grid(True, axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(y, [m.label for m in ms])
    axes[0].set_title("Classification performance", loc="left")
    axes[1].set_title(f"dashed: {ms[0].label} mean; open: seeds", loc="left", fontsize=8, color=MUTED)
    fig.savefig(path); plt.close(fig)
    print(f"  wrote {path}")

    rows = []
    for m in ms:
        rows.append({"model": m.key, "label": m.label, "seeds": len(m.runs),
                     "accuracy": round(mean(m, "accuracy"), 4), "auc": round(mean(m, "auc"), 4),
                     "d_auc_vs_base": round(mean(m, "auc") - mean(ms[0], "auc"), 4),
                     "mflops_per_jet": round(m.bench["flops_per_jet"] / 1e6, 1) if "flops_per_jet" in m.bench else "",
                     "ms_per_jet": round(m.bench["latency_us_per_jet"] / 1e3, 2) if "latency_us_per_jet" in m.bench else ""})
    print()
    print(f"  {'model':<7}{'seeds':>6}{'acc':>8}{'AUC':>8}{'dAUC':>9}{'MFLOP/jet':>11}{'ms/jet':>8}")
    for r in rows:
        print(f"  {r['model']:<7}{r['seeds']:>6}{r['accuracy']:>8.4f}{r['auc']:>8.4f}"
              f"{r['d_auc_vs_base']:>+9.4f}{r['mflops_per_jet']!s:>11}{r['ms_per_jet']!s:>8}")
    if csv_path:
        with open(csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        print(f"  wrote {csv_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", help="flat CSV/JSON instead of the s10 result files")
    ap.add_argument("--bench", default=os.path.join(RES, "variants_bench.json"))
    ap.add_argument("--classes", default="Hbb,Tbqq,Wqq,Zqq")
    ap.add_argument("--seed", type=int, default=0, help="seed whose ROC curves are drawn")
    ap.add_argument("--metric", default="auc", choices=["auc", "accuracy"])
    ap.add_argument("--usetex", action="store_true", help="needs a LaTeX install")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    style(args.usetex)
    os.makedirs(args.out, exist_ok=True)
    meta = None
    if args.table:
        load_table(MODELS, args.table)
    else:
        load_s10(MODELS)
        meta = load_bench(MODELS, args.bench)
    o = lambda f: os.path.join(args.out, f)
    plot_efficiency(MODELS, o("variants_efficiency.png"), metric=args.metric)
    plot_roc(MODELS, o("variants_roc.png"), classes=args.classes.split(","), seed=args.seed)
    plot_latency(MODELS, o("variants_latency.png"), meta)
    plot_accuracy(MODELS, o("variants_accuracy.png"), csv_path=os.path.join(RES, "variants_summary.csv"))
