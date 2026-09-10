"""S9 - can the model actually fit? Overfit a small class-balanced sample.

Everything before this checks that ErwinParTv2 is *correct*. This checks it has
the capacity and the gradient path to learn at all: given 512 jets and enough
epochs it should drive training accuracy to ~100%. A model that cannot overfit a
small sample will never produce a meaningful S10 number, and the failure would
otherwise hide behind a plausible-looking loss curve on the full dataset.

Dropout is disabled - overfitting is the goal here, not generalisation. The
held-out split is reported only to confirm the model is fitting *this* sample
rather than having learnt nothing; it is not the S10 measurement.

Run from the weaver conda env:
    python tests/s9_overfit.py [--jets 512] [--epochs 60] [--device auto]
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "networks"))
from erwin_part.model import ErwinParticleTransformerV2  # noqa: E402
from jetclass_data import CLASS_NAMES, NUM_FEATURES, find_root, load_jets  # noqa: E402

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
failures = []


def check(name, ok, detail=""):
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}{' - ' + detail if detail else ''}")
    if not ok:
        failures.append(name)


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jets", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--holdout", type=int, default=128)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = pick_device(args.device)
    torch.manual_seed(args.seed)
    torch.set_num_threads(6)

    total = args.jets + args.holdout
    pts, fts, vec, msk, y = load_jets(find_root(WORKSPACE), total, seed=args.seed)
    n = len(y)
    tr, ho = slice(0, n - args.holdout), slice(n - args.holdout, n)
    pts, fts, vec, msk, y = (t.to(dev) for t in (pts, fts, vec, msk, y))
    ntr = n - args.holdout
    counts = np.bincount(y[tr].cpu().numpy(), minlength=10)
    assert counts.min() > 0, (
        f"a class is absent from the training split ({counts.tolist()}) - the "
        f"sample is not class-mixed")
    print(f"device {dev} | {ntr} train + {args.holdout} held-out jets | "
          f"{counts.min()}-{counts.max()} per class")

    model = ErwinParticleTransformerV2(
        input_dim=NUM_FEATURES, num_classes=10, seq_len=64,
        block_params={"dropout": 0.0, "attn_dropout": 0.0, "activation_dropout": 0.0},
    ).to(dev)
    model.trimmer.enabled = False           # no random truncation while overfitting
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs * max(1, ntr // args.batch), eta_min=1e-6)

    def evaluate(sl):
        model.eval()
        with torch.no_grad():
            logits = torch.cat([
                model(fts[sl][i:i + 256], v=vec[sl][i:i + 256], mask=msk[sl][i:i + 256],
                      points=pts[sl][i:i + 256])
                for i in range(0, len(y[sl]), 256)])
        model.train()
        return (float((logits.argmax(-1) == y[sl]).float().mean()),
                float(F.cross_entropy(logits, y[sl])), logits)

    print(f"\n{'epoch':>6} {'loss':>8} {'train acc':>10} {'held-out':>9} {'s/epoch':>8}")
    model.train()
    hist, t_start = [], time.perf_counter()
    for ep in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        perm = torch.randperm(ntr, device=dev)
        run = 0.0
        for i in range(0, ntr, args.batch):
            b = perm[i:i + args.batch]
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(
                model(fts[b], v=vec[b], mask=msk[b], points=pts[b]), y[b])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run += loss.item() * len(b)
        if ep % max(1, args.epochs // 12) == 0 or ep == args.epochs:
            acc, _, _ = evaluate(tr)
            hacc, _, _ = evaluate(ho)
            hist.append((ep, run / ntr, acc, hacc))
            print(f"{ep:>6} {run/ntr:>8.4f} {acc:>10.3f} {hacc:>9.3f} "
                  f"{time.perf_counter()-t0:>8.2f}")

    elapsed = time.perf_counter() - t_start
    acc, tl, logits = evaluate(tr)
    hacc, _, _ = evaluate(ho)

    print(f"\ntrained {args.epochs} epochs in {elapsed:.0f}s "
          f"({elapsed/args.epochs:.2f}s/epoch, {elapsed*1e3/(args.epochs*ntr):.2f} ms/jet)")

    print()
    check("training accuracy reaches ~100%", acc > 0.95,
          f"{acc:.3f} after {args.epochs} epochs (chance = 0.100)")
    check("training loss is driven near zero", tl < 0.15, f"loss = {tl:.4f}")
    check("the loss decreased monotonically in trend",
          hist[-1][1] < hist[0][1], f"{hist[0][1]:.3f} -> {hist[-1][1]:.3f}")
    check("held-out accuracy is above chance",
          hacc > 0.20, f"{hacc:.3f} on {args.holdout} unseen jets "
                       f"(chance 0.100; this is NOT the S10 number)")

    per = [float((logits.argmax(-1)[y[tr] == c] == c).float().mean()) for c in range(10)]
    check("every class is fitted, not just the easy ones", min(per) > 0.85,
          "min " + f"{min(per):.2f} ({CLASS_NAMES[int(np.argmin(per))]}), "
          f"max {max(per):.2f}")
    print("       per-class train accuracy: " +
          "  ".join(f"{CLASS_NAMES[c]} {per[c]:.2f}" for c in range(10)))

    jets_per_s = ntr * args.epochs / elapsed
    print(f"\n       throughput {jets_per_s:.0f} jets/s on {dev} -> a 100k-jet epoch "
          f"would take ~{100_000/jets_per_s/60:.0f} min")

    print()
    if failures:
        print(f"S9 FAILED: {len(failures)} check(s) - {', '.join(failures)}")
        sys.exit(1)
    print("S9 PASSED")


if __name__ == "__main__":
    main()
