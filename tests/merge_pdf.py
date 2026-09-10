"""Merge the full ErwinParT story into one deck.

    1–7    ErwinParT.pdf        the original ball-tree talk (May)
    8–11   four v1 figures      ball partitioning, centroids, complexity, topologies
    12–23  ErwinParTv2 results  this cycle's measurements

All three sources are 16:9. The original deck and the figure pages are
1920x1080 pt; the v2 deck is regenerated at the same size so the merged file has
one uniform page geometry rather than pages that jump size in a viewer.

    python tests/merge_pdf.py    -> results/ErwinParT_full_deck.pdf
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pypdf import PdfReader, PdfWriter

import make_pdf as M

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "..", "results")
W_PT, H_PT = 1920, 1080

# The four figures carried over from the v1 work. They use the ORIGINAL framing:
# a single attention level, fixed ball size m in {16, 32, 64}, jets padded to
# next_pow2(n) = 128. v2 uses m = 8 with a three-level hierarchy at L = 64 — the
# notes flag the change so the differing m values do not read as inconsistency.
FIGS = [
    ("08", "ball partitioning", "How the tree groups constituents in (Δη, Δφ)",
     "fig1_constituent_grouping.png"),
    ("09", "each ball is a subjet", "$p_T$-weighted ball centroids",
     "fig2_ball_centroids.png"),
    ("10", "the complexity argument", "O(n·m) against O(n²)",
     "fig3_complexity.png"),
    ("11", "across topologies", "The same partitioning on top, W and QCD jets",
     "fig4_multijet_grouping.png"),
]


def figure_pages(path):
    """Render the four carried-over figures as titled 16:9 pages."""
    M.W, M.H = W_PT / 72, H_PT / 72
    with PdfPages(path) as pdf:
        for num, eyebrow, title, png in FIGS:
            fig = M.page(pdf)
            M.head(fig, num, eyebrow, title)
            img = mpimg.imread(os.path.join(RES, png))
            ih, iw = img.shape[0], img.shape[1]
            bx, by, bw, bh = 0.045, 0.05, 0.91, 0.70
            box_ar = (bw * M.W) / (bh * M.H)
            img_ar = iw / ih
            if img_ar > box_ar:
                w_, h_ = bw, bw * M.W / img_ar / M.H
            else:
                h_, w_ = bh, bh * M.H * img_ar / M.W
            ax = fig.add_axes([bx + (bw - w_) / 2, by + (bh - h_) / 2, w_, h_])
            ax.imshow(img); ax.axis("off")
            fig.text(0.045, 0.80, "carried over from the v1 ball-tree study · "
                     "single level, ball size m ∈ {16, 32, 64}, jets padded to 128",
                     color=M.MUTED, fontsize=13, family="monospace")
            pdf.savefig(fig); plt.close(fig)


def main():
    os.makedirs(RES, exist_ok=True)
    figs_pdf = os.path.join(RES, "_figs_for_merge.pdf")
    figure_pages(figs_pdf)

    out = PdfWriter()
    parts = [("ErwinParT.pdf", "original deck"),
             ("_figs_for_merge.pdf", "v1 figures"),
             ("_v2_for_merge.pdf", "v2 results")]
    for fname, label in parts:
        r = PdfReader(os.path.join(RES, fname))
        for pg in r.pages:
            w, h = float(pg.mediabox.width), float(pg.mediabox.height)
            if abs(w - W_PT) > 1 or abs(h - H_PT) > 1:
                raise SystemExit(f"{fname}: page is {w:.0f}x{h:.0f}, expected "
                                 f"{W_PT}x{H_PT} — refusing to merge mismatched geometry")
            out.add_page(pg)
        print(f"  {label:<16} {len(r.pages):>2} pages  ({fname})")

    out.add_metadata({"/Title": "ErwinParT — full deck (v1 study + v2 results)",
                      "/Subject": "FastML · JetClass jet tagging"})
    dst = os.path.join(RES, "ErwinParT_full_deck.pdf")
    with open(dst, "wb") as fh:
        out.write(fh)
    for tmp in ("_figs_for_merge.pdf", "_v2_for_merge.pdf"):
        os.remove(os.path.join(RES, tmp))
    print(f"\nwrote {os.path.normpath(dst)}  "
          f"({os.path.getsize(dst)/1e6:.2f} MB, {len(out.pages)} pages)")


if __name__ == "__main__":
    main()
