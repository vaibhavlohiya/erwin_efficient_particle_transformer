#!/usr/bin/env python
"""
Task 2 — Ball-assignment visualisation for the Erwin / BallMSA attention block.

Reproduces the *exact* grouping logic used inside `ParticleErwinBlock.forward`
for one jet and draws the particles in the (eta, phi) plane, colour-coded by the
ball they land in:

    1. positions come from `_compute_positions` (the real function the model uses),
    2. real particles are sorted by ascending Delta-R from the jet axis using the
       block's own `ParticleErwinBlock._delta_r_sort_idx` static method,
    3. the sorted sequence is cut into consecutive chunks of `ball_size`
       (rearrange '(n m) d -> n m d' groups adjacent members) -> ball id,
    4. the left panel is the unshifted layout; the right panel is the
       shift=True layout (sequence cyclically rolled by ball_size // 2), which
       is how alternating Erwin layers straddle ball boundaries.

Saved to `erwin_balls.png`.

Run from the repo root with the weaver env:
    /opt/homebrew/Caskroom/miniconda/base/envs/weaver/bin/python visualize_erwin_balls.py
"""
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from networks.EfficientParticleTransformer import ParticleErwinBlock, _compute_positions


def synth_jet(num_real, seed=1):
    """Build a physically-consistent dummy jet.

    We choose (eta, phi, pt) per particle, then construct the 4-momentum so that
    `_compute_positions` recovers exactly those (eta, phi) -- the plot therefore
    uses the model's real position function, not a shortcut.
    """
    rng = np.random.default_rng(seed)
    eta0, phi0 = 0.3, 0.7                                   # jet axis
    # a dense core plus a sparser halo, typical of a jet
    eta = np.concatenate([rng.normal(eta0, 0.06, num_real - num_real // 4),
                          rng.normal(eta0, 0.25, num_real // 4)])
    phi = np.concatenate([rng.normal(phi0, 0.06, num_real - num_real // 4),
                          rng.normal(phi0, 0.25, num_real // 4)])
    pt = rng.uniform(0.5, 20.0, num_real)
    px, py = pt * np.cos(phi), pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    e = pt * np.cosh(eta)
    v = torch.tensor(np.stack([px, py, pz, e]), dtype=torch.float32).unsqueeze(0)  # (1,4,P)
    return v


def ball_ids(real_pos, ball_size, shift):
    """Ball index per real particle, replicating ParticleErwinBlock.forward.

    Returns an int array of shape (num_real,) aligned to the ORIGINAL particle
    order, plus the sorted order and jet axis for annotation.
    """
    num_real = real_pos.shape[0]
    sort_idx = ParticleErwinBlock._delta_r_sort_idx(real_pos)   # sorted -> original
    # pad the sorted sequence up to a multiple of ball_size (repeat last particle)
    pad = (ball_size - num_real % ball_size) % ball_size
    padded_len = num_real + pad
    half = ball_size // 2

    # ball id along the (possibly rolled) sorted+padded sequence
    seq_ball = np.arange(padded_len) // ball_size
    if shift:
        # forward() rolls features/positions by -half before grouping; a slot's
        # ball is therefore the ball at its rolled position.
        seq_ball = np.roll(seq_ball, half)   # inverse of roll(-half) => position i's ball

    ball_sorted = seq_ball[:num_real]                          # per sorted real particle
    ball_orig = np.empty(num_real, dtype=int)
    ball_orig[sort_idx.numpy()] = ball_sorted                  # scatter back to original
    jet_axis = real_pos.mean(dim=0).numpy()
    n_balls = int(np.ceil(padded_len / ball_size))
    return ball_orig, jet_axis, n_balls


def panel(ax, eta, phi, ball_orig, jet_axis, ball_size, n_balls, title):
    cmap = plt.get_cmap("tab10" if n_balls <= 10 else "tab20")
    for b in range(n_balls):
        sel = ball_orig == b
        ax.scatter(eta[sel], phi[sel], s=45, color=cmap(b % cmap.N),
                   edgecolor="k", linewidth=0.3, label=f"ball {b}")
    ax.scatter(*jet_axis, marker="*", s=320, color="black", zorder=5,
               label="jet axis")
    ax.set_xlabel(r"$\eta$"); ax.set_ylabel(r"$\phi$")
    ax.set_title(title)
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(fontsize=7, loc="best", ncol=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-real", type=int, default=70,
                    help="real particles in the jet (70 -> pads to 80 = 5 balls of 16)")
    ap.add_argument("--ball-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="erwin_balls.png")
    args = ap.parse_args()

    v = synth_jet(args.num_real, args.seed)
    pos = _compute_positions(v)[0]              # (P,2) -- real function, single jet
    eta, phi = pos[:, 0].numpy(), pos[:, 1].numpy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharex=True, sharey=True)
    for ax, shift, tag in ((axes[0], False, "unshifted (even layers)"),
                           (axes[1], True, "shift=True (odd layers, roll ½·B)")):
        ball_orig, jet_axis, n_balls = ball_ids(pos, args.ball_size, shift)
        panel(ax, eta, phi, ball_orig, jet_axis, args.ball_size, n_balls,
              f"{tag}\n{args.num_real} particles → {n_balls} balls of {args.ball_size}")

    fig.suptitle("Erwin BallMSA — particle → ball assignment in the (η, φ) plane "
                 "(sorted by ΔR from jet axis)", fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"saved -> {args.out}")
    # quick text summary of ball sizes (unshifted)
    ball_orig, _, n_balls = ball_ids(pos, args.ball_size, False)
    counts = np.bincount(ball_orig, minlength=n_balls)
    print(f"{args.num_real} real particles, ball_size={args.ball_size} -> "
          f"{n_balls} balls; real-particle counts per ball: {counts.tolist()}")


if __name__ == "__main__":
    main()
