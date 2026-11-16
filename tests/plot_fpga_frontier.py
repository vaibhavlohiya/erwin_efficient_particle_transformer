"""Accuracy-vs-cost frontier and the ParT scorecard (ideas/visualization_ideas.md #1, #13).

Reads, never measures:
  tests/s10_results/cluster_eval.json     test-set metrics of the cluster 10% runs
  tests/s10_results/cluster_flops.json    deployed FLOPs, incl. PFN/P-CNN/ParticleNet
  tests/s10_results/fpga_efficiency.json  batch-1 memory and CPU latency
  tests/s10_results/test_multiplicity.json  truncation at L = 64

Writes plots/fpga_tradeoff/{frontier.png, scorecard.png, scorecard.md}.

    python tests/plot_fpga_frontier.py
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

INK, MUTED, GRID = "#1f1f22", "#6b6a74", "#e2e1e6"
GOOD, BAD = "#1a7f37", "#b42318"            # verdict text; always with a symbol and a word
GOOD_BG, BAD_BG, SAME_BG = "#e8f5ec", "#fbeceb", "#f2f2f4"
MODELS = {"part": ("ParT", "#1baf7a", "^", ["part-10pct"]),
          "v2":   ("ErwinParTv2", "#2a78d6", "o", ["v2-10pct-a", "v2-10pct-b"]),
          "lund": ("ErwinParTv2 + Lund", "#eb6834", "s", ["lund-10pct-a", "lund-10pct-b"])}
FLOPS_KEY = {"part": "ParT", "v2": "ErwinParTv2", "lund": "ErwinParTv2+Lund"}
V1_REPORTED = 0.825                          # ../Plots/ErwinParT.py; training details unrecorded
# Published full-JetClass accuracies (ParT paper, as in benchmark_comparison.md)
SWEEP = {16: "20261102-04fe81e-m16-10pct", 32: "20261102-04fe81e-m32-10pct", 64: "20261102-04fe81e-m64-10pct"}
SWEEP_FLOPS = {8: "ErwinParTv2", 16: "ErwinParTv2 m16", 32: "ErwinParTv2 m32", 64: "ErwinParTv2 m64"}
PUBLISHED = {"PFN": 0.772, "P-CNN": 0.809, "ParticleNet": 0.844, "ParT": 0.861}

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
    fl = json.load(open(os.path.join(RES, "cluster_flops.json")))
    fp = json.load(open(os.path.join(RES, "fpga_efficiency.json")))["models"]
    runs = {k: [ev[PREFIX + t] for t in m[3] if "accuracy" in ev.get(PREFIX + t, {})]
            for k, m in MODELS.items()}
    # ErwinParTv2's first-level ball size: m = 8 is the default (its 2 runs), then the sweep
    sweep = {8: runs["v2"]}
    sweep.update({m: [ev[t]] for m, t in SWEEP.items() if "accuracy" in ev.get(t, {})})
    return runs, sweep, fl, fp


# ---------------------------------------------------------------- frontier --
def plot_frontier(runs, sweep, fl):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    pts = {}
    # the ball-size curve: ErwinParTv2 at m = 8 (its default), 16, 32, 64 (dense level 0)
    ms = sorted(sweep)
    sx = [fl["models"][SWEEP_FLOPS[m]]["gflops_per_jet"] for m in ms]
    sa = [float(np.mean([r["accuracy"] for r in sweep[m]])) for m in ms]
    ax.plot(sx, sa, "-", color="#2a78d6", lw=1.4, alpha=0.6, zorder=2)
    for m, x, a in zip(ms, sx, sa):
        pts[f"m{m}"] = (x, a)
        if m != 8:
            ax.scatter([x], [a], s=55, marker="o", facecolors="#9cc3ef", edgecolors="#2a78d6", lw=1.2, zorder=4,
                       label="ErwinParTv2, larger first-level balls (1 run each)" if m == 16 else None)
    xp, ap = fl["models"]["ParT"]["gflops_per_jet"], float(np.mean([r["accuracy"] for r in runs["part"]]))
    place = {8: ((0, -14), "center", "top"), 16: ((-9, 0), "right", "center"),
             32: ((-4, 9), "right", "bottom"), 64: ((5, -9), "left", "top")}
    for m, x, a in zip(ms, sx, sa):
        (dx, dy), ha, va = place.get(m, ((0, 9), "center", "bottom"))
        ax.annotate(f"m = {m}" + (" (dense)" if m == 64 else "") + f"\n{xp / x:.1f}$\\times$ fewer FLOPs, "
                    f"{(a - ap) * 100:+.2f} pts", (x, a), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, va=va, fontsize=7.5, color=INK)
    for key, (label, c, mk, _) in MODELS.items():
        if not runs[key]:
            continue
        x = fl["models"][FLOPS_KEY[key]]["gflops_per_jet"]
        acc = [r["accuracy"] for r in runs[key]]
        pts[key] = (x, float(np.mean(acc)))
        if len(acc) > 1:
            ax.scatter([x] * len(acc), acc, s=24, facecolors="white", edgecolors=c, lw=1.2, zorder=3)
        ax.scatter([x], [np.mean(acc)], s=90, marker=mk, color=c, edgecolors="white", lw=1.5, zorder=4,
                   label=f"{label}{' (m = 8)' if key == 'v2' else ''} ({len(acc)} run{'s' if len(acc) > 1 else ''})")
    x1 = fl["models"]["ErwinParT v1"]["gflops_per_jet"]
    ax.scatter([x1], [V1_REPORTED], s=70, marker="D", facecolors="white", edgecolors="#eda100", lw=1.6,
               zorder=3, label="ErwinParT v1 (reported; training unrecorded)")
    # frontier: models nothing else beats on both cost and accuracy
    order = sorted(pts.values())
    front, best = [], -1
    for x, a in order:
        if a > best:
            front.append((x, a)); best = a
    fx, fa = zip(*front)
    ax.plot(fx, fa, color=MUTED, lw=1, ls=":", zorder=1)
    ax.set_xscale("log")
    ax.set_xlim(0.12, 1.1)
    ax.set_xticks([0.15, 0.2, 0.3, 0.5, 0.7, 1.0], ["0.15", "0.2", "0.3", "0.5", "0.7", "1"])
    ax.minorticks_off()
    ax.set_ylim(0.8235, 0.8415)
    ax.set_xlabel("forward GFLOPs per jet, deployed (all 128 slots)")
    ax.set_ylabel("test accuracy (2M jets)")
    ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=7.5)
    titled(ax, "Accuracy vs compute, same training budget",
           "10% of JetClass, 20 epochs; blue line: first-level ball size m; dotted: the efficiency frontier")

    ax = axes[1]
    ref = fl["reference_models"]
    cost = {"PFN": ref["PFN"]["gflops_per_jet"], "P-CNN": ref["P-CNN"]["gflops_per_jet"],
            "ParticleNet": ref["ParticleNet"]["gflops_per_jet"], "ParT": fl["models"]["ParT"]["gflops_per_jet"]}
    for name, a in PUBLISHED.items():
        ax.scatter([cost[name]], [a], s=60, marker="^" if name == "ParT" else "o", color=MUTED,
                   edgecolors="white", lw=1.2, zorder=3)
        ax.annotate(name, (cost[name], a), xytext=(7, -3), textcoords="offset points", fontsize=8.5)
    xv = fl["models"]["ErwinParTv2"]["gflops_per_jet"]
    ax.axvline(xv, color="#2a78d6", ls="--", lw=1.2)
    ax.text(xv * 1.08, 0.776, "ErwinParTv2's cost\n(not yet trained\non full JetClass)", fontsize=8,
            color="#2a78d6", va="bottom")
    ax.set_xscale("log")
    ax.set_xlim(0.005, 2.5)
    ax.set_xticks([0.01, 0.03, 0.1, 0.3, 1.0], ["0.01", "0.03", "0.1", "0.3", "1"])
    ax.set_ylim(0.765, 0.868)
    ax.set_xlabel("forward GFLOPs per jet, deployed")
    ax.set_ylabel("published test accuracy")
    ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    titled(ax, "Context: published models, full JetClass",
           "accuracies from the ParT paper; FLOPs counted here the same way (= 2 x the paper's MACs)")
    fig.tight_layout()
    save(fig, "frontier.png")


# --------------------------------------------------------------- scorecard --
CLASS_NAME = {"Hbb": "H→bb̄", "Hcc": "H→cc̄", "Hgg": "H→gg", "H4q": "H→4q",
              "Hqql": "H→ℓνqq′", "Zqq": "Z→qq̄", "Wqq": "W→qq′",
              "Tbqq": "t→bqq′", "Tbl": "t→bℓν"}
SYMBOL = {1: ("▲", GOOD), 0: ("≈", MUTED), -1: ("▼", BAD)}
CONFIGS = [(8, "ErwinParTv2", "m = 8"), (32, "ErwinParTv2 m32", "m = 32")]   # (m, flops/fpga key, label)


def geo_rej(r):
    return float(np.exp(np.mean([np.log(v["rej_wp"]) for v in r["vs_qcd"].values()])))


def scorecard_rows(runs, sweep, fl, fp):
    """Rows of (measure, ParT text, [(value, comparison, verdict) per config]); a 2-tuple is a
    section header. For accuracy, AUC and rejection, 'similar' means within twice the
    difference between ErwinParTv2's two runs; costs are 'similar' within 10%."""
    part = runs["part"][0]
    a, b = runs["v2"]
    tol_acc = 2 * abs(a["accuracy"] - b["accuracy"])
    tol_auc = 2 * abs(a["auc_macro"] - b["auc_macro"])
    tol_rej = 2 * abs(geo_rej(a) - geo_rej(b)) / ((geo_rej(a) + geo_rej(b)) / 2)
    mem = lambda name: next(r for r in fp[name]["scaling"] if r["n"] == 128)["peak_bytes_fixed"] / 1e6
    lat = lambda name: float(np.median([r["latency_ms_fixed"] for r in fp[name]["scaling"]]))
    spread = lambda name: (lambda v: np.percentile(v, 95) / np.percentile(v, 5))(
        np.array([r["ms"] for r in fp[name]["latency"]]))
    counts = json.load(open(os.path.join(RES, "test_multiplicity.json")))["counts"]
    tot = np.sum([np.array(c) for c in counts.values()], axis=0)
    trunc64, trunc128 = tot[65:].sum() / tot.sum() * 100, tot[129:].sum() / tot.sum() * 100

    def perf(delta, tol):
        return 0 if abs(delta) <= tol else (1 if delta > 0 else -1)

    def cost(ratio):            # ratio = ours / ParT; lower is better
        return 0 if abs(ratio - 1) <= 0.10 else (1 if ratio < 1 else -1)

    fP = fl["models"]["ParT"]
    cells = {k: [] for k in ("acc", "auc", "rej", "worst", "flops", "mem", "params", "wmem", "lat", "spread",
                             "trunc", "train")}
    for m, key, _ in CONFIGS:
        rs, fV = sweep[m], fl["models"][key]
        acc = float(np.mean([r["accuracy"] for r in rs])); auc = float(np.mean([r["auc_macro"] for r in rs]))
        g = float(np.mean([geo_rej(r) for r in rs]))
        ratios = {c: float(np.exp(np.mean([np.log(r["vs_qcd"][c]["rej_wp"]) for r in rs]))) / part["vs_qcd"][c]["rej_wp"]
                  for c in part["vs_qcd"]}
        worst = sorted(ratios.items(), key=lambda kv: kv[1])[:2]
        cells["acc"].append((f"{acc:.4f}", f"{(acc - part['accuracy']) * 100:+.2f} pts",
                             perf(acc - part["accuracy"], tol_acc)))
        cells["auc"].append((f"{auc:.4f}", f"{auc - part['auc_macro']:+.4f}", perf(auc - part["auc_macro"], tol_auc)))
        cells["rej"].append((f"{g:,.0f}", f"{g / geo_rej(part):.2f}$\\times$",
                             perf(g / geo_rej(part) - 1, tol_rej)))
        cells["worst"].append(("", ", ".join(f"{CLASS_NAME[c]} {v:.2f}" for c, v in worst), -1))
        cells["flops"].append((f"{fV['gflops_per_jet']:.3f}", f"{fP['gflops_per_jet'] / fV['gflops_per_jet']:.1f}$\\times$ fewer",
                               cost(fV["gflops_per_jet"] / fP["gflops_per_jet"])))
        cells["mem"].append((f"{mem(key):.2f} MB", f"{mem('ParT') / mem(key):.1f}$\\times$ less", cost(mem(key) / mem("ParT"))))
        cells["params"].append((f"{fV['params'] / 1e6:.2f}M", f"{fV['params'] / fP['params']:.2f}$\\times$",
                                cost(fV["params"] / fP["params"])))
        cells["wmem"].append((f"{fp[key]['weight_bytes'] / 1e6:.1f} MB",
                              f"{fp[key]['weight_bytes'] / fp['ParT']['weight_bytes']:.2f}$\\times$",
                              cost(fp[key]["weight_bytes"] / fp["ParT"]["weight_bytes"])))
        cells["lat"].append((f"{lat(key):.1f} ms", f"{lat('ParT') / lat(key):.1f}$\\times$ faster", cost(lat(key) / lat("ParT"))))
        cells["spread"].append((f"{spread(key):.1f}$\\times$", "flatter", 1 if spread(key) < spread("ParT") else -1))
        cells["trunc"].append((f"64 / {trunc64:.1f}%", "drops lowest-pT", -1))
        cells["train"].append((f"{fV['train_pflops']:.1f} PFLOPs", f"{fP['train_pflops'] / fV['train_pflops']:.1f}$\\times$ less",
                               cost(fV["train_pflops"] / fP["train_pflops"])))
    mP = mem("ParT")
    return [
        ("Performance", "same training budget: 10% of JetClass, 20 epochs; 2M test jets"),
        ("Test accuracy", f"{part['accuracy']:.4f}", cells["acc"]),
        ("Macro AUC", f"{part['auc_macro']:.4f}", cells["auc"]),
        ("Rejection, geometric mean of 9 classes", f"{geo_rej(part):,.0f}", cells["rej"]),
        ("Weakest classes (rejection / ParT's)", "", cells["worst"]),
        ("Cost when deployed", "every jet uses all 128 input slots, as weaver builds inference/ONNX models"),
        ("Forward GFLOPs per jet", f"{fP['gflops_per_jet']:.3f}", cells["flops"]),
        ("Peak activation memory (fp32)", f"{mP:.2f} MB", cells["mem"]),
        ("Parameters", f"{fP['params'] / 1e6:.2f}M", cells["params"]),
        ("Weight memory (fp32)", f"{fp['ParT']['weight_bytes'] / 1e6:.1f} MB", cells["wmem"]),
        ("Latency per jet (CPU, batch 1)", f"{lat('ParT'):.1f} ms", cells["lat"]),
        ("Latency spread, real jets (p95/p5)", f"{spread('ParT'):.1f}$\\times$", cells["spread"]),
        ("Particles kept / jets truncated", f"128 / {trunc128:.3f}%", cells["trunc"]),
        ("Training", "same budget"),
        ("Training compute", f"{fP['train_pflops']:.1f} PFLOPs", cells["train"]),
    ]


def plot_scorecard(rows):
    n = len(rows)
    fig = plt.figure(figsize=(13, 0.42 * n + 1.5))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(n + 2.4, 0)
    ax.text(0.01, 0.5, "ErwinParTv2 vs ParT: scorecard", fontsize=13, va="center")
    xs = [(0.53, 0.545), (0.775, 0.79)]                       # (value right edge, comparison left edge)
    ax.text(0.40, 1.4, "ParT", fontsize=9.5, color=MUTED, ha="right", va="center", fontweight="bold")
    for (xv, xc), (m, _, lab) in zip(xs, CONFIGS):
        ax.text(xv, 1.4, f"ErwinParTv2, {lab}", fontsize=9.5, color=MUTED, ha="right", va="center", fontweight="bold")
        ax.text(xc, 1.4, "vs ParT", fontsize=9.5, color=MUTED, ha="left", va="center", fontweight="bold")
    ax.text(0.01, 1.4, "Measure", fontsize=9.5, color=MUTED, va="center", fontweight="bold")
    ax.axhline(1.85, color=MUTED, lw=0.8)
    y = 2.3
    for row in rows:
        if len(row) == 2:
            ax.text(0.01, y, row[0], fontsize=10.5, fontweight="bold", va="center")
            ax.text(0.18, y, row[1], fontsize=8, color=MUTED, va="center")
            ax.axhline(y + 0.45, color=GRID, lw=0.6)
            y += 1
            continue
        name, p, cs = row
        ax.text(0.02, y, name, fontsize=9.5, va="center")
        ax.text(0.40, y, p, fontsize=9.5, va="center", ha="right")
        for (xv, xc), (val, cmp, verdict) in zip(xs, cs):
            sym, col = SYMBOL[verdict]
            ax.text(xv, y, val, fontsize=9.5, va="center", ha="right", fontweight="bold")
            ax.text(xc, y, f"{sym} {cmp}", fontsize=9, va="center", color=col)
        y += 1
    ax.text(0.01, y + 0.3, "▲ better / ▼ worse than ParT; ≈ similar: accuracy, AUC and rejection within twice the "
            "difference between ErwinParTv2's two m = 8 runs, costs within 10%. m = 8: 2 runs; m = 32 and ParT: 1 run.",
            fontsize=8, color=MUTED, va="center")
    save(fig, "scorecard.png")


def write_scorecard_md(rows):
    clean = lambda s: s.replace("$\\times$", "×")
    lines = ["# ErwinParTv2 vs ParT: scorecard", "",
             "Generated by `tests/plot_fpga_frontier.py` from `tests/s10_results/*.json`. ErwinParTv2 at its",
             "default first-level ball size (m = 8, 2 runs) and at m = 32 (1 run), the best point of the",
             "ball-size sweep.", "",
             "| Measure | ParT | ErwinParTv2, m = 8 | vs ParT | ErwinParTv2, m = 32 | vs ParT |",
             "| --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        if len(row) == 2:
            lines.append(f"| **{row[0]}** ({row[1]}) | | | | | |")
            continue
        name, p, cs = row
        cells = []
        for val, cmp, verdict in cs:
            cells += [f"**{clean(val)}**" if val else "", f"{SYMBOL[verdict][0]} {clean(cmp)}"]
        lines.append(f"| {name} | {clean(p)} | " + " | ".join(cells) + " |")
    lines += ["", "▲ better / ▼ worse than ParT; ≈ similar: accuracy, AUC and rejection within twice the difference",
              "between ErwinParTv2's two m = 8 runs, costs within 10%.", "",
              "Latency is batch-1 CPU time (1 thread, fp32): use it for ratios and trends, not as an FPGA estimate.",
              "Memory counts what PyTorch allocates, an upper bound for a dataflow design."]
    path = os.path.join(OUT, "scorecard.md")
    open(path, "w").write("\n".join(lines) + "\n")
    print("  wrote", os.path.relpath(path, os.path.join(HERE, "..")))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    runs, sweep, fl, fp = load()
    plot_frontier(runs, sweep, fl)
    rows = scorecard_rows(runs, sweep, fl, fp)
    plot_scorecard(rows)
    write_scorecard_md(rows)
