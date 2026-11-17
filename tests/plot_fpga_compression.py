"""Progress plots for the local FPGA-compression study (ErwinParTLite).

Reads tests/s10_results/lite_*_result.json (s16_lite_train.py) and
lite_baseline.json (s15_lite_baseline.py). Plot 4 also needs each width-sweep
checkpoint's train and validation cross-entropy; those are computed once
(CPU, eval mode) and cached in lite_losses.json.

Writes plots/fpga_compression/
  frontier.png   accuracy vs MACs per jet, every post-fix Lite run and ErwinParTv2
  ladder.png     the D = 64 ablation ladder, both seeds
  per_class.png  per-class rejection at 90% efficiency and AUC vs ErwinParTv2
  losses.png     train vs validation cross-entropy across widths

Everything is on the 100k-jet local sample, 12 epochs, the same recipe and
split for every model. Run from the weaver conda env:
    python tests/plot_fpga_compression.py
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "s10_results")
OUT = os.path.join(HERE, "..", "plots", "fpga_compression")

INK, MUTED, GRID = "#1f1f22", "#6b6a74", "#e2e1e6"
BASE = "#3d3c44"                                       # ErwinParTv2, the reference
D_COLOR = {16: "#9cc3ef", 32: "#5a9be2", 64: "#2a6fc4", 128: "#163f78"}   # sequential: width
TRAIN, VAL = "#2a6fc4", "#d9661f"                      # categorical pair for plot 4
SIGNAL = ["Hbb", "Hcc", "Hgg", "H4q", "Hqql", "Zqq", "Wqq", "Tbqq", "Tbl"]
LABEL = {"Hbb": r"H$\to b\bar b$", "Hcc": r"H$\to c\bar c$", "Hgg": r"H$\to gg$", "H4q": r"H$\to 4q$",
         "Hqql": r"H$\to \ell\nu qq'$", "Zqq": r"Z$\to q\bar q$", "Wqq": r"W$\to qq'$",
         "Tbqq": r"t$\to bqq'$", "Tbl": r"t$\to b\ell\nu$"}

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


def res(tag):
    return json.load(open(os.path.join(RES, f"lite_{tag}_result.json")))


BL = json.load(open(os.path.join(RES, "lite_baseline.json")))
V2 = BL["checkpoints"]["erwin_m8"]                     # ErwinParTv2 default (m = 8), seed-0 split
V2_S1 = BL["checkpoints"]["erwin_s1_base"]             # the same model, seed-1 split
V2_COST = BL["cost_baseline_ball8"]

# Post-fix runs (after 60fa762) of the carried-forward variant: physics summaries,
# lookup-table features, BatchNorm. (tag(s), D, kind); several tags = seeds.
RUNS = [
    (["d64_phys_lut_bn_s0", "d64_phys_lut_bn_s1"], 64, "full"),
    (["w128_f4"], 128, "full"), (["w64_f4"], 64, "full"),
    (["w32_f4"], 32, "full"), (["w32_f2"], 32, "full"),
    (["w16_f4"], 16, "full"), (["w16_f2"], 16, "full"),
    (["dp64_111"], 64, "one block"), (["dp32f4_111"], 32, "one block"),
    (["dp64_cls1"], 64, "one cls"), (["dp32f4_cls1"], 32, "one cls"),
    (["kd64_f2"], 64, "distilled"), (["kd32_f4"], 32, "distilled"), (["kd32_f2"], 32, "distilled"),
    (["kd16_f4"], 16, "distilled"),
]
MARK = {"full": ("o", "2 blocks per level"), "one block": ("s", "1 block per level"),
        "one cls": ("^", "1 class-attention layer"), "distilled": ("D", "distilled from ErwinParTv2")}


def available(tags):
    return [t for t in tags if os.path.exists(os.path.join(RES, f"lite_{t}_result.json"))]


# ------------------------------------------------------------ 1. frontier --
def plot_frontier():
    fig, ax = plt.subplots(figsize=(7.6, 4.9))
    pts = []
    seen = set()
    for tags, D, kind in RUNS:
        tags = available(tags)
        if not tags:
            continue
        rs = [res(t) for t in tags]
        acc = [r["accuracy"] for r in rs]
        macs = rs[0]["cost"]["macs"] / 1e6
        mk, lab = MARK[kind]
        filled = kind != "distilled"
        ax.scatter(macs, np.mean(acc), s=46, marker=mk, zorder=4,
                   facecolor=D_COLOR[D] if filled else "white", edgecolor=D_COLOR[D] if filled else D_COLOR[D],
                   linewidth=1.6 if not filled else 0.8)
        if len(acc) > 1:
            ax.plot([macs, macs], [min(acc), max(acc)], color=D_COLOR[D], lw=1.4, zorder=3)
        if kind not in seen:
            ax.scatter([], [], marker=mk, s=40, facecolor="white" if not filled else MUTED,
                       edgecolor=MUTED, label=lab)
            seen.add(kind)
        pts.append((macs, np.mean(acc), tags[0], kind, rs[0]["cost"]["params"]))
    # ErwinParTv2: seed 0 and seed 1 splits, same recipe
    v2 = [V2["accuracy"], V2_S1["accuracy"]]
    vm = V2_COST["macs"] / 1e6
    ax.scatter(vm, np.mean(v2), marker="X", s=80, color=BASE, zorder=4, label="ErwinParTv2 (m = 8)")
    ax.plot([vm, vm], [min(v2), max(v2)], color=BASE, lw=1.4)
    ax.annotate("ErwinParTv2\n2.21M params", (vm, np.mean(v2)), xytext=(-8, -26), textcoords="offset points",
                ha="right", fontsize=8, color=BASE)
    # Pareto frontier over the non-distilled Lite runs (fewer MACs and higher accuracy)
    front = sorted((p for p in pts if p[3] != "distilled"), key=lambda p: p[0])
    best, line = -1, []
    for p in front:
        if p[1] > best:
            line.append(p); best = p[1]
    ax.step([p[0] for p in line], [p[1] for p in line], where="post", color=MUTED, lw=0.9, ls="--", zorder=2)
    # call-outs: the two candidates
    for tag, txt, off in (("d64_phys_lut_bn_s0", "D = 64: 376k params\n6.2$\\times$ fewer MACs", (-150, 14)),
                          ("dp32f4_111", "D = 32, 1 block/level: 93k\n23$\\times$ fewer MACs", (10, -26))):
        p = next(q for q in pts if q[2] == tag)
        ax.annotate(txt, (p[0], p[1]), xytext=off, textcoords="offset points", fontsize=8, color=INK,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    for D, c in D_COLOR.items():
        ax.scatter([], [], color=c, s=40, label=f"D = {D}")
    ax.set_xscale("log")
    ax.set_xlabel("MACs per jet (multiply-accumulates, batch 1, 64 particles)")
    ax.set_ylabel("test accuracy (10 classes)")
    ax.grid(True, color=GRID, lw=0.6, which="both")
    ax.legend(loc="lower right", ncol=2)
    titled(ax, "Accuracy vs compute: ErwinParTLite against ErwinParTv2",
           "local 100k-jet sample, 12 epochs, same recipe for all; vertical bars = two seeds; dashed = frontier")
    save(fig, "frontier.png")


# -------------------------------------------------------------- 2. ladder --
LADDER = [("Morton tiles", "d64_morton"),
          ("+ physics summaries", "d64_phys"),
          ("+ summaries, lookup-table features", "d64_phys_lut"),
          ("+ summaries, BatchNorm", "d64_phys_bn"),
          ("+ summaries, lookup table, BatchNorm  (chosen)", "d64_phys_lut_bn")]


def plot_ladder():
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    lo, hi = sorted([V2["accuracy"], V2_S1["accuracy"]])
    ax.axvspan(lo, hi, color=GRID, zorder=0)
    ax.text(hi + 0.0004, len(LADDER) - 0.45, f"ErwinParTv2, two seeds\n({lo:.4f}, {hi:.4f}), 95.7M MACs",
            fontsize=7.5, color=MUTED, va="top")
    for i, (lab, stem) in enumerate(LADDER):
        y = len(LADDER) - 1 - i
        accs = [res(f"{stem}_s{s}")["accuracy"] for s in (0, 1)]
        macs = res(f"{stem}_s0")["cost"]["macs"] / 1e6
        ax.plot(accs, [y, y], color=MUTED, lw=1.0, zorder=2)
        ax.scatter(accs[0], y, marker="o", s=34, color=D_COLOR[64], zorder=3, label="seed 0" if i == 0 else None)
        ax.scatter(accs[1], y, marker="s", s=34, facecolor="white", edgecolor=D_COLOR[64], lw=1.3, zorder=3,
                   label="seed 1" if i == 0 else None)
        ax.plot([np.mean(accs)] * 2, [y - 0.22, y + 0.22], color=INK, lw=2, zorder=4,
                label="mean" if i == 0 else None)
        ax.text(1.01, y, f"mean {np.mean(accs):.4f}\n{macs:.1f}M MACs", fontsize=7.5, color=MUTED,
                va="center", ha="left", transform=ax.get_yaxis_transform())
    ax.set_yticks(range(len(LADDER)))
    ax.set_yticklabels([l for l, _ in LADDER][::-1])
    ax.set_xlim(0.7390, 0.7830)
    ax.set_ylim(-0.6, len(LADDER) - 0.3)
    ax.set_xlabel("test accuracy")
    ax.grid(True, axis="x", color=GRID, lw=0.6)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    titled(ax, "What each FPGA-friendly choice costs or buys (D = 64, 376k parameters)",
           "two seeds each (different train/test splits); seed-to-seed spread ~0.005")
    save(fig, "ladder.png")


# ----------------------------------------------------------- 3. per class --
CANDIDATES = [("d64_phys_lut_bn_s0", 64, "D = 64 (376k, 15.5M MACs)"),
              ("dp32f4_111", 32, "D = 32, 1 block/level (93k, 4.2M)"),
              ("w16_f4", 16, "D = 16 (38k, 1.7M)")]


def plot_per_class():
    ref = V2
    resolved = [c for c in SIGNAL if not ref["rej90"][c]["capped"] and ref["rej90"][c]["qcd_pass"] >= 10]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0), gridspec_kw={"width_ratios": [len(resolved), len(SIGNAL)]})
    ax = axes[0]
    w = 0.26
    x = np.arange(len(resolved))
    for k, (tag, D, lab) in enumerate(CANDIDATES):
        r = res(tag)
        ratio, err = [], []
        for c in resolved:
            a, b = r["rej90"][c], ref["rej90"][c]
            q = a["rej"] / b["rej"]
            ratio.append(q)
            err.append(q * np.sqrt(1 / max(a["qcd_pass"], 1) + 1 / b["qcd_pass"]))   # Poisson, both counts
        ax.bar(x + (k - 1) * w, ratio, w * 0.9, color=D_COLOR[D], label=lab, zorder=3)
        ax.errorbar(x + (k - 1) * w, ratio, yerr=err, fmt="none", ecolor=MUTED, lw=0.8, capsize=2, zorder=4)
    ax.axhline(1, color=INK, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[c] for c in resolved])
    ax.set_ylabel("QCD rejection at 90% signal eff., relative to ErwinParTv2")
    ax.grid(True, axis="y", color=GRID, lw=0.6)
    ax.legend(loc="upper left")
    titled(ax, "Background rejection by class",
           "classes where ErwinParTv2 lets ≥ 10 of 1,017 QCD test jets through; bars = Poisson error")
    ax = axes[1]
    x = np.arange(len(SIGNAL))
    for k, (tag, D, lab) in enumerate(CANDIDATES):
        r = res(tag)
        d = [1e3 * (r["auc"][c] - ref["auc"][c]) for c in SIGNAL]
        ax.bar(x + (k - 1) * w, d, w * 0.9, color=D_COLOR[D], zorder=3)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[c] for c in SIGNAL], rotation=30, ha="right")
    ax.set_ylabel(r"AUC difference vs ErwinParTv2 ($\times 10^{-3}$)")
    ax.grid(True, axis="y", color=GRID, lw=0.6)
    titled(ax, "One-vs-rest AUC by class", "all signal classes, including the ones rejection cannot resolve locally")
    fig.tight_layout()
    save(fig, "per_class.png")


# -------------------------------------------------------------- 4. losses --
WIDTH = [("w16_f2", 16, 2), ("w16_f4", 16, 4), ("w32_f2", 32, 2), ("w32_f4", 32, 4),
         ("d64_phys_lut_bn_s0", 64, 2), ("w64_f4", 64, 4), ("w128_f4", 128, 4)]


def losses():
    """Train (10k training jets) and validation cross-entropy of each checkpoint, eval mode, CPU."""
    path = os.path.join(RES, "lite_losses.json")
    have = json.load(open(path)) if os.path.exists(path) else {}
    todo = [t for t, _, _ in WIDTH if t not in have]
    if todo:
        import torch
        sys.path[:0] = [HERE, os.path.join(HERE, "..", "networks")]
        from erwin_part.fpga import ErwinParTLite, LiteConfig
        from jetclass_data import find_root, load_all
        from s16_lite_train import lite_call, predict
        data = load_all(find_root(os.path.join(HERE, "..", "..")), seed=0)
        n = len(data[-1])
        sets = {"train": torch.arange(10000), "val": torch.arange(int(0.8 * n), int(0.9 * n))}
        for t in todo:
            r = res(t)
            cfg = {k: tuple(v) if isinstance(v, list) else v for k, v in r["config"].items()}
            m = ErwinParTLite(LiteConfig(**cfg))
            m.load_state_dict(torch.load(os.path.join(RES, f"lite_{t}_best.pt")))
            have[t] = {"params": r["cost"]["params"]}
            for k, idx in sets.items():
                p, y = predict(m, lite_call, data, idx, torch.device("cpu"))
                have[t][k] = float(-torch.log(p.gather(1, y.view(-1, 1)).clamp(min=1e-12)).mean())
            print(f"  {t}: train {have[t]['train']:.4f}  val {have[t]['val']:.4f}")
        json.dump(have, open(path, "w"), indent=1)
    return have


def plot_losses():
    L = losses()
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    for ffn, ls, mk in ((4, "-", "o"), (2, "--", "s")):
        rows = sorted((L[t]["params"], L[t]["train"], L[t]["val"], D) for t, D, f in WIDTH if f == ffn)
        p = [r[0] for r in rows]
        ax.plot(p, [r[1] for r in rows], ls=ls, marker=mk, color=TRAIN, lw=1.6, ms=5,
                label=f"train, FFN ratio {ffn}")
        ax.plot(p, [r[2] for r in rows], ls=ls, marker=mk, color=VAL, lw=1.6, ms=5,
                label=f"validation, FFN ratio {ffn}")
        if ffn == 4:
            ax.fill_between(p, [r[1] for r in rows], [r[2] for r in rows], color=VAL, alpha=0.08, lw=0)
            for r in rows:
                ax.annotate(f"D = {r[3]}\ngap {r[2] - r[1]:.3f}", (r[0], r[1]), xytext=(0, -12),
                            textcoords="offset points", ha="center", va="top", fontsize=7.5, color=MUTED)
    ax.set_xscale("log")
    ax.set_xlabel("parameters")
    ax.set_ylabel("cross-entropy at the best checkpoint")
    ax.set_ylim(0.50, 0.775)
    ax.grid(True, color=GRID, lw=0.6, which="both")
    ax.legend(loc="upper right", ncol=2, bbox_to_anchor=(1.0, 0.86))
    ax.text(0.98, 0.95, "small gap: capacity-limited\nlarge gap: overfitting 80k jets",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color=MUTED)
    titled(ax, "Where the local sample stops being the limit",
           "train = 10k training jets, eval mode (same footing as validation); seed 0")
    save(fig, "losses.png")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    plot_frontier()
    plot_ladder()
    plot_per_class()
    plot_losses()
