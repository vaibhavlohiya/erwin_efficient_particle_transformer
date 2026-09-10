"""Render the results deck to a shareable 16:9 PDF.

No HTML-to-PDF engine is available on this machine (no Chrome, wkhtmltopdf,
weasyprint or prince), so the deck is composed directly with matplotlib's
PdfPages. Pages mirror the published deck slide for slide; the figures are the
same PNGs `make_plots.py` writes, placed at their native aspect ratio.

    python tests/make_pdf.py            -> results/ErwinParTv2_results.pdf
    python tests/make_pdf.py --notes    -> also emit the presenter-notes version
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(__file__)
PLOTS = os.path.join(HERE, "..", "plots")
OUT = os.path.join(HERE, "..", "results")

W, H = 13.333, 7.5                       # 16:9
GROUND, CARD = "#f4f3f8", "#ffffff"
INK, INK2, MUTED = "#171526", "#443f58", "#6a6680"
ACCENT, ACCENT_SOFT = "#4a3aa7", "#ece9f8"
OK, OK_SOFT = "#0e6f78", "#e2f1f2"
BAD, BAD_SOFT = "#9d4318", "#f8ebe3"
WARN, WARN_SOFT = "#8a6d1f", "#f6f0dd"
RULE = "#ddd9e8"
SANS = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": SANS})


def page(pdf, bg=GROUND):
    fig = plt.figure(figsize=(W, H), facecolor=bg)
    return fig


def head(fig, num, eyebrow, title):
    fig.text(0.045, 0.935, num, color=ACCENT, fontsize=10.5, fontweight="bold",
             family="monospace")
    fig.text(0.075, 0.935, eyebrow.upper(), color=MUTED, fontsize=10.5,
             family="monospace")
    fig.text(0.045, 0.868, title, color=INK, fontsize=25, fontweight="bold",
             va="top")


def chips(fig, items, y=0.045):
    x = 0.045
    for kind, txt in items:
        fc, ec, tc = {"": ("#efedf5", RULE, INK2), "ok": (OK_SOFT, OK, OK),
                      "bad": (BAD_SOFT, BAD, BAD), "warn": (WARN_SOFT, WARN, WARN)}[kind]
        w = 0.0088 * len(txt) + 0.022
        fig.patches.append(FancyBboxPatch((x, y), w, 0.042, transform=fig.transFigure,
                                          boxstyle="round,pad=0.004,rounding_size=0.006",
                                          fc=fc, ec=ec, lw=0.9, zorder=2))
        fig.text(x + w / 2, y + 0.021, txt, ha="center", va="center", color=tc,
                 fontsize=10.5, family="monospace", zorder=3)
        x += w + 0.014


def figure_page(pdf, num, eyebrow, title, png, callouts=()):
    fig = page(pdf)
    head(fig, num, eyebrow, title)
    img = mpimg.imread(os.path.join(PLOTS, png))
    ih, iw = img.shape[0], img.shape[1]
    # content box in figure coords, then fit the image inside preserving aspect
    bx, by, bw, bh = 0.045, 0.115 if callouts else 0.065, 0.91, 0.66
    box_ar = (bw * W) / (bh * H)
    img_ar = iw / ih
    if img_ar > box_ar:
        w_, h_ = bw, bw * W / img_ar / H
    else:
        h_, w_ = bh, bh * H * img_ar / W
    ax = fig.add_axes([bx + (bw - w_) / 2, by + (bh - h_) / 2, w_, h_])
    ax.imshow(img); ax.axis("off")
    if callouts:
        chips(fig, callouts)
    pdf.savefig(fig); plt.close(fig)


def title_page(pdf, off=0):
    fig = page(pdf, CARD)
    fig.text(0.06, 0.86, "FASTML · ARCHITECTURE REVIEW", color=MUTED, fontsize=11,
             family="monospace")
    fig.text(0.06, 0.72, "Ball-tree attention", color=INK, fontsize=54, fontweight="bold",
             va="top")
    fig.text(0.06, 0.585, "for jet tagging", color=INK, fontsize=54, fontweight="bold",
             va="top")
    fig.text(0.06, 0.43, "Does a jet need every constituent\nto see every other one?",
             color=INK2, fontsize=19, va="top", linespacing=1.5)
    chips(fig, [("", "ErwinParTv2"), ("", "JetClass · 100k jets"),
                ("", "117 verification checks"), ("", "2 seeds")], y=0.16)
    fig.text(0.06, 0.085, "No ParT accuracy baseline: ParT could not be trained on this\n"
             "hardware. Every accuracy number here is the model against itself.",
             color=BAD, fontsize=11.5, family="monospace", va="top", linespacing=1.7)
    pdf.savefig(fig); plt.close(fig)


def overview_page(pdf, off=0):
    fig = page(pdf)
    head(fig, f"{2 + off:02d}", "how the pieces fit", "The analysis in one line")
    steps = [("Build", "ball tree + hierarchy\nin pure PyTorch", False),
             ("Verify", "117 checks, S0–S9\ntree parity 512/512", False),
             ("Train", "80k / 10k / 10k\n20 epochs", False),
             ("Ablate", "ball size · U bias\nrotation · readout", True),
             ("Cost", "FLOPs, memory\nthroughput", True)]
    x, w, gap = 0.045, 0.172, 0.018
    for name, sub, hi in steps:
        fig.patches.append(FancyBboxPatch((x, 0.45), w, 0.22, transform=fig.transFigure,
                                          boxstyle="round,pad=0.004,rounding_size=0.006",
                                          fc=ACCENT_SOFT if hi else "#efedf5",
                                          ec=ACCENT if hi else RULE, lw=1.1))
        fig.text(x + 0.014, 0.625, name, fontsize=17, fontweight="bold", color=INK)
        fig.text(x + 0.014, 0.545, sub, fontsize=10.5, color=MUTED, family="monospace",
                 va="top", linespacing=1.6)
        if x + w + gap < 0.96:
            fig.text(x + w + gap / 2, 0.56, "\u203a", ha="center", color=MUTED, fontsize=21)
        x += w + gap
    for i, t in enumerate([f"slides {3+off}–{5+off} — what it is, and is the input right",
                           f"slides {6+off}–{9+off} — does locality cost anything",
                           f"slide {10+off} — what it buys"]):
        fig.text(0.048, 0.33 - i * 0.055, t, color=MUTED, fontsize=11.5, family="monospace")
        fig.add_artist(plt.Line2D([0.045, 0.045], [0.318 - i * 0.055, 0.348 - i * 0.055],
                                  color=ACCENT, lw=2.2, transform=fig.transFigure))
    pdf.savefig(fig); plt.close(fig)


CLAIMS_NUM, SYN_NUM = "11", "12"


def claims_page(pdf):
    fig = page(pdf)
    head(fig, CLAIMS_NUM, "epistemics", "What the data do and do not establish")
    cols = [(0.045, OK, OK_SOFT, "SUPPORTED", [
        "Dense attention is not significantly better than\n8-particle balls  (p = 0.295)",
        "The hierarchy is load-bearing; the readout must\nsee coarse levels",
        "U bias, distance bias and rotation are all\nremovable — confirmed on two seeds",
        "1.5–1.9× fewer FLOPs; trains where ParT\ncannot on 16 GB",
        "L=64 costs 0.115% of jet pT; effect confined\nto 6.3% of jets"]),
        (0.515, BAD, BAD_SOFT, "NOT SUPPORTED", [
        "That it tags better than ParT — no baseline\nexists locally",
        "That coarse-level pair features help —\npredicted, measured, did not",
        "Effects smaller than ±0.006 accuracy —\nbelow our detection limit",
        "Anything at 100M-jet scale; 80k jets is\n0.08% of JetClass",
        "That the stripped model should ship — the\nnulls are bounded, not zero"])]
    for x, c, soft, hdr, items in cols:
        fig.patches.append(FancyBboxPatch((x, 0.075), 0.44, 0.70, transform=fig.transFigure,
                                          boxstyle="round,pad=0.006,rounding_size=0.006",
                                          fc=soft, ec=c, lw=1.2))
        fig.text(x + 0.02, 0.735, hdr, color=c, fontsize=11.5, fontweight="bold",
                 family="monospace")
        y = 0.665
        for it in items:
            fig.text(x + 0.028, y, "•", color=c, fontsize=13, va="top")
            fig.text(x + 0.045, y, it, color=INK2, fontsize=12.5, va="top", linespacing=1.45)
            y -= 0.125
    pdf.savefig(fig); plt.close(fig)


def synthesis_page(pdf):
    fig = page(pdf)
    head(fig, SYN_NUM, "synthesis", "Where this leaves us")
    items = [("1", "Locality is not costing us accuracy.", "Dense attention inside our own "
              "hierarchy is no better than 8-particle balls, p = 0.295.", False),
             ("2", "The hierarchy is what matters.", "The readout must see coarse levels; "
              "everything else we inherited is removable, on two seeds.", False),
             ("3", "The efficiency is real but modest at jet scale.", "1.5–1.9× FLOPs — and the "
              "difference between training and not training on 16 GB.", False),
             ("4", "We cannot yet say it tags better.", "No local ParT baseline exists. "
              "That is the open question, not a footnote.", False),
             ("›", "Next: the cluster run.", "kube/erwinpartv2-job.yml, full JetClass, "
              "cheap configuration first.", True)]
    y = 0.70
    for n, bold, rest, hi in items:
        fig.patches.append(FancyBboxPatch((0.045, y - 0.095), 0.91, 0.10,
                                          transform=fig.transFigure,
                                          boxstyle="round,pad=0.004,rounding_size=0.006",
                                          fc=ACCENT_SOFT if hi else "#efedf5",
                                          ec=ACCENT if hi else RULE, lw=1.1))
        fig.text(0.065, y - 0.048, n, color=ACCENT, fontsize=19, fontweight="bold",
                 ha="center", va="center")
        fig.text(0.093, y - 0.030, bold, color=INK, fontsize=14.5, fontweight="bold", va="center")
        fig.text(0.093, y - 0.068, rest, color=INK2, fontsize=12.5, va="center")
        y -= 0.125
    pdf.savefig(fig); plt.close(fig)


FIGS = [("03", "what we built", "Attention is local; the hierarchy carries global information",
         "hierarchy.png", [("", "64 > 32 > 16 nodes"), ("ok", "bottleneck = one ball")]),
        ("04", "is the input right", "Why 64 slots per jet", "multiplicity.png",
         [("", "median 38"), ("", "93.7% fit"), ("warn", "Hgg loses its tail in 24% of jets")]),
        ("05", "the control", "Does truncation cost us anything?", "l128_control.png",
         [("ok", "untruncated: identical, p = 0.971"), ("warn", "truncated: +3.7 pts, p = 0.013"),
          ("", "overall p = 0.449")]),
        ("06", "the core result", "Is ball locality costing us accuracy?", "ballsize_summary.png",
         [("ok", "m=64 IS dense attention"), ("ok", "vs m=8: p = 0.295, not significant"),
          ("warn", "2.1x the training time")]),
        ("07", "per-class check", "Does that hide a class-specific effect?", "ballsize_roc.png",
         [("ok", "curves overlap in all nine classes")]),
        ("08", "what earns its keep", "Which parts of the architecture actually matter?",
         "ablations.png", [("bad", "ParT's U bias: p = 0.823"),
                           ("ok", "only the readout is significant"),
                           ("", "detection limit +/- 0.006")]),
        ("09", "does it reproduce", "The same conclusions on a second seed", "seed_repeat.png",
         [("ok", "both nulls hold twice"), ("bad", "log-polar flips sign - it was noise")]),
        ("10", "what it buys", "Efficiency, scoped to jets that actually exist", "flops.png",
         [("ok", "1.5-1.9x over the physical range"), ("", "L^1.03 vs L^1.19"),
          ("warn", "L > 128 is synthetic")])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-num", type=int, default=1,
                    help="first slide number printed on the pages (for merged decks)")
    ap.add_argument("--out", default="ErwinParTv2_results.pdf")
    ap.add_argument("--size", default="960x540", help="page size in points")
    a = ap.parse_args()
    global W, H
    pw, ph = (float(x) for x in a.size.split("x"))
    W, H = pw / 72, ph / 72
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, a.out)
    off = a.start_num - 1
    with PdfPages(path) as pdf:
        global CLAIMS_NUM, SYN_NUM
        CLAIMS_NUM, SYN_NUM = f"{11 + off:02d}", f"{12 + off:02d}"
        title_page(pdf, off)
        overview_page(pdf, off)
        for num, eb, ti, png, cs in FIGS:
            figure_page(pdf, f"{int(num) + off:02d}", eb, ti, png, cs)
        claims_page(pdf)
        synthesis_page(pdf)
        d = pdf.infodict()
        d["Title"] = "Ball-tree attention for jet tagging — ErwinParTv2 results"
        d["Subject"] = "FastML architecture review · JetClass 100k"
    print(f"wrote {os.path.normpath(path)}  "
          f"({os.path.getsize(path)/1e6:.2f} MB, {2 + len(FIGS) + 2} pages)")


if __name__ == "__main__":
    main()
