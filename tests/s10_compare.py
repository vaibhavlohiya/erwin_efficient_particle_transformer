"""S10 - ErwinParTv2 against ParT on the full local JetClass sample.

Identical data, identical splits, identical seed, identical optimiser and
schedule; the only thing that differs is the architecture. Both models are
trained from scratch here - no pre-trained weights - so the comparison is of the
architectures, not of their pre-training budgets.

Metrics are the ones the jet-tagging literature reports, so the numbers drop
straight into Plots/ErwinParT.py:

  accuracy      over all 10 classes
  AUC           macro one-vs-rest
  rej50         background rejection 1/eps_QCD at 50% signal efficiency, per
                signal class, scored on the binary discriminant p_s/(p_s + p_QCD)

Scope, stated plainly: this is 100k jets, which is the local sample. Real
JetClass training is 100M. Treat these as a controlled small-scale comparison,
not as published numbers.

    python tests/s10_compare.py --epochs 30 --batch 256
    python tests/s10_compare.py --models erwin          # one model only
"""
import argparse
import json
import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "networks"))
from jetclass_data import CLASS_NAMES, NUM_FEATURES, find_root, load_all  # noqa: E402

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTDIR = os.path.join(os.path.dirname(__file__), "s10_results")
QCD = 0


def log(path, msg):
    line = f"[{datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(path, "a") as fh:
        fh.write(line + "\n")


def build(name, device, ball=8, rotate=45.0, seq_len=64, **opts):
    if name.startswith("erwin"):
        from erwin_part.model import ErwinParticleTransformerV2
        # `ball` is the level-0 (constituent) ball size. Level 1 stays at 8; the
        # level-2 ball is seq_len//4 so the bottleneck is always exactly ONE ball
        # whatever the sequence length - 64->16, 96->24, 128->32. ball=seq_len
        # makes level 0 a single ball too, i.e. full dense attention.
        if seq_len % 16 or (ball and seq_len % ball):
            raise ValueError(f"seq_len {seq_len} must be divisible by 16 and by ball {ball}")
        return ErwinParticleTransformerV2(
            input_dim=NUM_FEATURES, num_classes=10, seq_len=seq_len,
            ball_sizes=[ball, 8, seq_len // 4], strides=[2, 2], depths=[2, 2, 2],
            pair_embed_dims=[64, 64], rotate=rotate, **opts,
        ).to(device)
    from weaver.nn.model.ParticleTransformer import ParticleTransformer
    return ParticleTransformer(
        input_dim=NUM_FEATURES, num_classes=10, pair_input_dim=4,
        pair_embed_dims=[64, 64, 64], num_layers=8, num_cls_layers=2,
        block_params={"dropout": 0.1, "attn_dropout": 0.1, "activation_dropout": 0.1},
        cls_block_params={"dropout": 0, "attn_dropout": 0, "activation_dropout": 0},
        fc_params=[]).to(device)


def call(model, name, pts, fts, vec, msk):
    if name.startswith("erwin"):
        return model(fts, v=vec, mask=msk, points=pts)
    return model(fts, v=vec, mask=msk)


def auc_ovr(scores, y, cls):
    """One-vs-rest AUC by rank statistic - no sklearn dependency."""
    pos = (y == cls)
    npos, nneg = int(pos.sum()), int((~pos).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks over ties
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return (ranks[pos].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)


def rejection(probs, y, sig, eff=0.5):
    """1/eps_bkg at fixed signal efficiency, signal vs QCD only."""
    keep = (y == sig) | (y == QCD)
    p = probs[keep]
    d = p[:, sig] / np.clip(p[:, sig] + p[:, QCD], 1e-12, None)
    s, b = d[y[keep] == sig], d[y[keep] == QCD]
    if len(s) == 0 or len(b) == 0:
        return float("nan")
    thr = np.quantile(s, 1.0 - eff)
    n_pass = int((b >= thr).sum())
    # With a finite test set the measurable rejection is capped: if no background
    # jet survives the cut, all we can honestly say is "> len(b)". Reporting inf
    # would overstate it, so return the ceiling and let the caller flag it.
    return float(len(b)) if n_pass == 0 else len(b) / n_pass


@torch.no_grad()
def _probs(model, name, data, idx, device, batch):
    pts, fts, vec, msk, y = data
    model.eval()
    out = []
    for i in range(0, len(idx), batch):
        b = idx[i:i + batch]
        out.append(torch.softmax(
            call(model, name, pts[b].to(device), fts[b].to(device),
                 vec[b].to(device), msk[b].to(device)).float(), dim=-1).cpu())
    return torch.cat(out).numpy(), y[idx].numpy()


def roc_curves(model, name, data, idx, device, batch):
    """Full background-rejection curve per signal class, for the figures."""
    probs, yy = _probs(model, name, data, idx, device, batch)
    out = {}
    for c in range(1, 10):
        keep = (yy == c) | (yy == QCD)
        p, lab = probs[keep], yy[keep]
        d = p[:, c] / np.clip(p[:, c] + p[:, QCD], 1e-12, None)
        s_, b_ = d[lab == c], d[lab == QCD]
        effs = np.linspace(0.02, 1.0, 150)
        eps = np.array([(b_ >= t).mean() for t in np.quantile(s_, 1 - effs)])
        out[CLASS_NAMES[c]] = {"eff": effs.tolist(), "eps_bkg": eps.tolist(),
                               "auc": auc_ovr(d, lab, c), "n_bkg": int(len(b_))}
    return out


@torch.no_grad()
def evaluate(model, name, data, idx, device, batch):
    pts, fts, vec, msk, y = data
    model.eval()
    out = []
    for i in range(0, len(idx), batch):
        b = idx[i:i + batch]
        out.append(torch.softmax(
            call(model, name, pts[b].to(device), fts[b].to(device),
                 vec[b].to(device), msk[b].to(device)).float(), dim=-1).cpu())
    probs = torch.cat(out).numpy()
    yy = y[idx].numpy()
    acc = float((probs.argmax(1) == yy).mean())
    aucs = [auc_ovr(probs[:, c], yy, c) for c in range(10)]
    rej = {CLASS_NAMES[c]: rejection(probs, yy, c) for c in range(1, 10)}
    return {"accuracy": acc, "auc_macro": float(np.nanmean(aucs)),
            "auc_per_class": {CLASS_NAMES[c]: float(aucs[c]) for c in range(10)},
            "rej50": rej}


def train_one(name, data, splits, args, device):
    os.makedirs(OUTDIR, exist_ok=True)
    name = f"{name}{args.tag}"
    logp = os.path.join(OUTDIR, f"{name}_log.txt")
    tr, va, te = splits
    pts, fts, vec, msk, y = data

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    model = build(name, device, ball=args.ball_size, rotate=args.rotate,
                  seq_len=args.seq_len,
                  use_pair_bias=not args.no_pair_bias,
                  use_dist_bias=not args.no_dist_bias,
                  readout=args.readout, tree_space=args.tree_space)
    nparam = sum(p.numel() for p in model.parameters())
    log(logp, f"=== {name} | {nparam/1e6:.2f}M params | L {args.seq_len} | "
              f"ball {args.ball_size} | rotate {args.rotate} | device {device} | "
              f"train {len(tr)} val {len(va)} test {len(te)} ===")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * ((len(tr) + args.batch - 1) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)

    best, best_state, hist = -1.0, None, []
    t_start = time.perf_counter()
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = tr[torch.randperm(len(tr))]
        run, seen, t0 = 0.0, 0, time.perf_counter()
        micro = args.micro_batch or args.batch
        for i in range(0, len(perm), args.batch):
            b = perm[i:i + args.batch]
            opt.zero_grad(set_to_none=True)
            # Accumulate over micro-batches. Each chunk's loss is weighted by its
            # share of the full batch, so the accumulated gradient equals the one
            # a single forward over `b` would have produced - the optimiser sees
            # exactly the same update, only peak activation memory differs.
            for j in range(0, len(b), micro):
                mb = b[j:j + micro]
                loss = F.cross_entropy(
                    call(model, name, pts[mb].to(device), fts[mb].to(device),
                         vec[mb].to(device), msk[mb].to(device)), y[mb].to(device))
                (loss * (len(mb) / len(b))).backward()
                run += loss.item() * len(mb)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            seen += len(b)
        vm = evaluate(model, name, data, va, device, args.eval_batch)
        hist.append({"epoch": ep, "loss": run / seen, "val_acc": vm["accuracy"],
                     "val_auc": vm["auc_macro"], "secs": time.perf_counter() - t0})
        star = ""
        if vm["accuracy"] > best:
            best, star = vm["accuracy"], "  *"
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        log(logp, f"epoch {ep:>3}/{args.epochs}  loss {run/seen:.4f}  "
                  f"val acc {vm['accuracy']:.4f}  val AUC {vm['auc_macro']:.4f}  "
                  f"{time.perf_counter()-t0:5.1f}s{star}")
        with open(os.path.join(OUTDIR, f"{name}_history.json"), "w") as fh:
            json.dump(hist, fh, indent=1)

    if best_state:
        model.load_state_dict(best_state)
    # Final metrics on CPU: MPS evaluation shifts with batch size (0.7568 at 256
    # vs 0.7684 at 512 for the same checkpoint), while CPU is exactly stable.
    # Training stays on the accelerator; only this last pass moves.
    model = model.to("cpu")
    tm = evaluate(model, name, data, te, torch.device("cpu"), args.eval_batch)
    tm["eval_device"] = "cpu"
    tm["ball_size"] = args.ball_size
    tm["rotate"] = args.rotate
    tm["seq_len"] = args.seq_len
    tm["config"] = {"pair_bias": not args.no_pair_bias,
                    "dist_bias": not args.no_dist_bias,
                    "readout": args.readout, "tree_space": args.tree_space}
    tm["roc"] = roc_curves(model, name, data, te, torch.device("cpu"), args.eval_batch)
    model = model.to(device)
    tm["params"] = nparam
    tm["train_minutes"] = (time.perf_counter() - t_start) / 60
    tm["epochs"] = args.epochs
    tm["history"] = hist
    log(logp, f"TEST  acc {tm['accuracy']:.4f}  AUC {tm['auc_macro']:.4f}  "
              f"({tm['train_minutes']:.1f} min)")
    log(logp, "  rej50: " + "  ".join(f"{k} {v:.0f}" for k, v in tm["rej50"].items()))
    with open(os.path.join(OUTDIR, f"{name}_result.json"), "w") as fh:
        json.dump(tm, fh, indent=1)
    torch.save(best_state, os.path.join(OUTDIR, f"{name}_best.pt"))
    return tm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256,
                    help="effective batch (the optimisation batch)")
    ap.add_argument("--micro-batch", type=int, default=0,
                    help="forward/backward chunk; 0 = same as --batch. Use a "
                         "smaller value when a model's activations do not fit: "
                         "gradients accumulate to the full --batch, so the "
                         "optimisation is identical, only peak memory drops.")
    ap.add_argument("--eval-batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--models", default="erwin,part")
    ap.add_argument("--ball-size", type=int, default=8,
                    help="level-0 ball size; 64 = one ball = dense attention")
    ap.add_argument("--rotate", type=float, default=45.0,
                    help="tree rotation angle in degrees; 0 disables cross-ball mixing")
    ap.add_argument("--tag", default="", help="suffix for output filenames")
    ap.add_argument("--seq-len", type=int, default=64,
                    help="slots per jet; jets are pt-sorted and truncated to this")
    ap.add_argument("--no-pair-bias", action="store_true", help="drop ParT's U")
    ap.add_argument("--no-dist-bias", action="store_true", help="drop Erwin Eq. 10")
    ap.add_argument("--readout", default="multi", choices=["multi", "fine", "coarse"])
    ap.add_argument("--tree-space", default="etaphi", choices=["etaphi", "logpolar"])
    ap.add_argument("--jets", type=int, default=0, help="0 = whole file")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    dev = (torch.device("cuda") if torch.cuda.is_available()
           else torch.device("mps") if torch.backends.mps.is_available()
           else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)
    torch.set_num_threads(6)
    os.makedirs(OUTDIR, exist_ok=True)
    main_log = os.path.join(OUTDIR, "run_log.txt")

    t = time.perf_counter()
    # Always read the whole file, then subsample AFTER the shuffle. Passing
    # `entry_stop` here instead would read a class-sorted prefix - the first
    # 3000 entries of JetClass are 3000 QCD jets, and the run would silently
    # report 100% accuracy on a one-class problem.
    data = load_all(find_root(WORKSPACE), seed=args.seed)
    if args.jets:
        data = tuple(t[:args.jets] for t in data)
    n = len(data[-1])
    log(main_log, f"loaded {n} jets in {time.perf_counter()-t:.1f}s | device {dev}")

    ntr, nva = int(0.8 * n), int(0.1 * n)
    idx = torch.arange(n)
    splits = (idx[:ntr], idx[ntr:ntr + nva], idx[ntr + nva:])
    for label, sp in zip(("train", "val", "test"), splits):
        present = len(torch.unique(data[-1][sp]))
        assert present == 10, (
            f"{label} split has {present}/10 classes - the sample is not "
            f"class-mixed, so accuracy and AUC would be meaningless")
    log(main_log, f"split {len(splits[0])}/{len(splits[1])}/{len(splits[2])} "
                  f"| epochs {args.epochs} batch {args.batch} lr {args.lr}")

    results = {}
    for name in args.models.split(","):
        results[name] = train_one(name.strip(), data, splits, args, dev)

    if len(results) > 1:
        log(main_log, "\n" + "=" * 66)
        log(main_log, f"{'model':<14}{'params':>9}{'acc':>9}{'AUC':>9}{'min':>8}")
        for k, r in results.items():
            log(main_log, f"{k:<14}{r['params']/1e6:>8.2f}M{r['accuracy']:>9.4f}"
                          f"{r['auc_macro']:>9.4f}{r['train_minutes']:>8.1f}")
        n_qcd = int((data[-1][splits[2]] == QCD).sum())
        log(main_log, "\nbackground rejection at 50% signal efficiency (vs QCD)")
        log(main_log, f"  ceiling {n_qcd} - only {n_qcd} QCD jets in the test split, "
                      f"so a value at the ceiling means 'at least this'")
        names = list(results)
        log(main_log, f"{'class':<8}" + "".join(f"{k:>12}" for k in names))
        for c in CLASS_NAMES[1:]:
            log(main_log, f"{c:<8}" + "".join(
                f"{results[k]['rej50'][c]:>12.0f}" for k in names))
        with open(os.path.join(OUTDIR, "summary.json"), "w") as fh:
            json.dump({k: {kk: vv for kk, vv in r.items() if kk != "history"}
                       for k, r in results.items()}, fh, indent=1)


if __name__ == "__main__":
    main()
