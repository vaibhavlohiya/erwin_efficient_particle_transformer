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
    return runs, fl, fp


# ---------------------------------------------------------------- frontier --
def plot_frontier(runs, fl):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    pts = {}
    for key, (label, c, mk, _) in MODELS.items():
        if not runs[key]:
            continue
        x = fl["models"][FLOPS_KEY[key]]["gflops_per_jet"]
        acc = [r["accuracy"] for r in runs[key]]
        pts[key] = (x, float(np.mean(acc)))
        if len(acc) > 1:
            ax.scatter([x] * len(acc), acc, s=24, facecolors="white", edgecolors=c, lw=1.2, zorder=3)
        ax.scatter([x], [np.mean(acc)], s=90, marker=mk, color=c, edgecolors="white", lw=1.5, zorder=4,
                   label=f"{label} ({len(acc)} run{'s' if len(acc) > 1 else ''})")
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
    ax.step(fx, fa, where="post", color=MUTED, lw=1, ls=":", zorder=1)
    (xv, av), (xp, ap) = pts["v2"], pts["part"]
    ax.annotate("", xy=(xv * 1.06, av), xytext=(xp * 0.94, ap),
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.1))
    ax.text(np.sqrt(xv * xp), (av + ap) / 2 + 0.0012,
            f"{xp / xv:.1f}$\\times$ fewer FLOPs\n{(av - ap) * 100:+.2f} points accuracy\n"
            f"({av / ap * 100:.1f}% of ParT)", ha="center", va="bottom", fontsize=8.5)
    ax.set_xscale("log")
    ax.set_xlim(0.12, 1.1)
    ax.set_xticks([0.15, 0.2, 0.3, 0.5, 0.7, 1.0], ["0.15", "0.2", "0.3", "0.5", "0.7", "1"])
    ax.minorticks_off()
    ax.set_ylim(0.822, 0.844)
    ax.set_xlabel("forward GFLOPs per jet, deployed (all 128 slots)")
    ax.set_ylabel("test accuracy (2M jets)")
    ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=7.5)
    titled(ax, "Accuracy vs compute, same training budget",
           "10% of JetClass, 20 epochs; dotted: the efficiency frontier; open circles: single runs")

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
def geo_rej(r):
    return float(np.exp(np.mean([np.log(v["rej_wp"]) for v in r["vs_qcd"].values()])))


def scorecard_rows(runs, fl, fp):
    part, v2 = runs["part"][0], runs["v2"]
    m = lambda f: float(np.mean([f(r) for r in v2]))
    ratios = {c: float(np.exp(np.mean([np.log(r["vs_qcd"][c]["rej_wp"]) for r in v2]))) / part["vs_qcd"][c]["rej_wp"]
              for c in part["vs_qcd"]}
    worst = sorted(ratios.items(), key=lambda kv: kv[1])[:2]
    fP, fV = fl["models"]["ParT"], fl["models"]["ErwinParTv2"]
    mem = lambda name: next(r for r in fp[name]["scaling"] if r["n"] == 128)["peak_bytes_fixed"] / 1e6
    lat = lambda name: float(np.median([r["latency_ms_fixed"] for r in fp[name]["scaling"]]))
    spread = lambda name: (lambda v: np.percentile(v, 95) / np.percentile(v, 5))(
        np.array([r["ms"] for r in fp[name]["latency"]]))
    counts = json.load(open(os.path.join(RES, "test_multiplicity.json")))["counts"]
    tot = np.sum([np.array(c) for c in counts.values()], axis=0)
    trunc64 = tot[65:].sum() / tot.sum() * 100
    trunc128 = tot[129:].sum() / tot.sum() * 100
    acc_p, acc_v = part["accuracy"], m(lambda r: r["accuracy"])
    auc_p, auc_v = part["auc_macro"], m(lambda r: r["auc_macro"])
    g_p, g_v = geo_rej(part), m(geo_rej)
    # (section, measure, ParT, ErwinParTv2, comparison, verdict: +1 better / 0 similar / -1 worse)
    return [
        ("Performance", "same training budget: 10% of JetClass, 20 epochs; 2M test jets"),
        ("Test accuracy", f"{acc_p:.4f}", f"{acc_v:.4f}",
         f"{(acc_v - acc_p) * 100:+.2f} points ({acc_v / acc_p * 100:.1f}%)", -1),
        ("Macro AUC", f"{auc_p:.4f}", f"{auc_v:.4f}", f"{auc_v - auc_p:+.4f} ({auc_v / auc_p * 100:.1f}%)", -1),
        ("Background rejection, geometric mean of 9 classes", f"{g_p:,.0f}", f"{g_v:,.0f}",
         f"{g_v / g_p:.2f}$\\times$", -1),
        ("Weakest classes (rejection vs ParT)", "", "",
         ", ".join(f"{CLASS_NAME[c]} {v:.2f}$\\times$" for c, v in worst), -1),
        ("Cost when deployed", "every jet uses all 128 input slots, as weaver builds inference/ONNX models"),
        ("Forward GFLOPs per jet", f"{fP['gflops_per_jet']:.3f}", f"{fV['gflops_per_jet']:.3f}",
         f"{fP['gflops_per_jet'] / fV['gflops_per_jet']:.1f}$\\times$ fewer", 1),
        ("Peak activation memory per jet (fp32)", f"{mem('ParT'):.2f} MB", f"{mem('ErwinParTv2'):.2f} MB",
         f"{mem('ParT') / mem('ErwinParTv2'):.1f}$\\times$ less", 1),
        ("Parameters", f"{fP['params'] / 1e6:.2f}M", f"{fV['params'] / 1e6:.2f}M",
         f"{fV['params'] / fP['params']:.2f}$\\times$", 0),
        ("Weight memory (fp32)", f"{fp['ParT']['weight_bytes'] / 1e6:.1f} MB",
         f"{fp['ErwinParTv2']['weight_bytes'] / 1e6:.1f} MB",
         f"{fp['ErwinParTv2']['weight_bytes'] / fp['ParT']['weight_bytes']:.2f}$\\times$", 0),
        ("Latency per jet, batch 1 (CPU, 1 thread)", f"{lat('ParT'):.1f} ms", f"{lat('ErwinParTv2'):.1f} ms",
         f"{lat('ParT') / lat('ErwinParTv2'):.1f}$\\times$ faster", 1),
        ("Latency spread over real jets, trimmed (p95/p5)", f"{spread('ParT'):.1f}$\\times$",
         f"{spread('ErwinParTv2'):.1f}$\\times$", "flatter, more predictable", 1),
        ("Particles kept / test jets truncated", f"128 / {trunc128:.3f}%", f"64 / {trunc64:.1f}%",
         "drops lowest-pT constituents", -1),
        ("Training", "same budget"),
        ("Training compute", f"{fP['train_pflops']:.1f} PFLOPs", f"{fV['train_pflops']:.1f} PFLOPs",
         f"{fP['train_pflops'] / fV['train_pflops']:.1f}$\\times$ less", 1),
    ]


CLASS_NAME = {"Hbb": "H→bb̄", "Hcc": "H→cc̄", "Hgg": "H→gg", "H4q": "H→4q",
              "Hqql": "H→ℓνqq′", "Zqq": "Z→qq̄", "Wqq": "W→qq′",
              "Tbqq": "t→bqq′", "Tbl": "t→bℓν"}

VERDICT = {1: ("▲ better", GOOD, GOOD_BG), 0: ("≈ similar", MUTED, SAME_BG),
           -1: ("▼ worse", BAD, BAD_BG)}


def plot_scorecard(rows):
    n = len(rows)
    fig = plt.figure(figsize=(11.5, 0.42 * n + 1.1))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(n + 1.6, 0)
    cols = [(0.01, "Measure", "left"), (0.43, "ParT", "right"), (0.56, "ErwinParTv2", "right"),
            (0.59, "ErwinParTv2 vs ParT", "left"), (0.86, "", "left")]
    ax.text(0.01, 0.5, "ErwinParTv2 vs ParT: scorecard", fontsize=13, va="center")
    for x, h, ha in cols:
        ax.text(x, 1.4, h, fontsize=9.5, color=MUTED, ha=ha, va="center", fontweight="bold")
    ax.axhline(1.85, color=MUTED, lw=0.8)
    y = 2.3
    for row in rows:
        if len(row) == 2:                                   # section header
            ax.text(0.01, y, row[0], fontsize=10.5, fontweight="bold", va="center")
            ax.text(0.20, y, row[1], fontsize=8, color=MUTED, va="center")
            ax.axhline(y + 0.45, color=GRID, lw=0.6)
            y += 1
            continue
        name, p, v, cmp, verdict = row
        word, col, bg = VERDICT[verdict]
        ax.add_patch(plt.Rectangle((0.855, y - 0.38), 0.135, 0.76, color=bg, lw=0))
        ax.text(0.02, y, name, fontsize=9.5, va="center")
        ax.text(0.43, y, p, fontsize=9.5, va="center", ha="right")
        ax.text(0.56, y, v, fontsize=9.5, va="center", ha="right", fontweight="bold")
        ax.text(0.59, y, cmp, fontsize=9.5, va="center")
        ax.text(0.865, y, word, fontsize=9.5, va="center", color=col)
        y += 1
    save(fig, "scorecard.png")


def write_scorecard_md(rows):
    lines = ["# ErwinParTv2 vs ParT: scorecard", "",
             "Generated by `tests/plot_fpga_frontier.py` from `tests/s10_results/*.json`.", "",
             "| Measure | ParT | ErwinParTv2 | ErwinParTv2 vs ParT | |", "| --- | --- | --- | --- | --- |"]
    for row in rows:
        if len(row) == 2:
            lines.append(f"| **{row[0]}** ({row[1]}) | | | | |")
            continue
        name, p, v, cmp, verdict = row
        clean = lambda s: s.replace("$\\times$", "×")
        bold = f"**{clean(v)}**" if v else ""
        lines.append(f"| {name} | {clean(p)} | {bold} | {clean(cmp)} | {VERDICT[verdict][0]} |")
    lines += ["", "Latency is batch-1 CPU time (1 thread, fp32): use it for ratios and trends, not as an "
              "FPGA estimate. Memory counts what PyTorch allocates, an upper bound for a dataflow design."]
    path = os.path.join(OUT, "scorecard.md")
    open(path, "w").write("\n".join(lines) + "\n")
    print("  wrote", os.path.relpath(path, os.path.join(HERE, "..")))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    runs, fl, fp = load()
    plot_frontier(runs, fl)
    rows = scorecard_rows(runs, fl, fp)
    plot_scorecard(rows)
    write_scorecard_md(rows)
