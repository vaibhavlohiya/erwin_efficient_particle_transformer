"""Train ErwinParTLite locally for the FPGA-compression study (steps 3-5).

Same recipe as tests/s10_compare.py, so a variant differs from the baseline
only by its model: the 100k-jet sample shuffled by --seed and split 80/10/10,
AdamW (lr 1e-3, wd 0.01), OneCycle (10% warm-up), grad-norm clip 1.0, the best
epoch by validation accuracy, test metrics on CPU. Scored with
lite_metrics.local_metrics; deployed cost from lite_metrics.deployed_cost.

Distillation (step 4): --teacher <s10 tag> --teacher-ball <m> uses that
ErwinParTv2 checkpoint (trained on the SAME seed's split) as the teacher; its
logits on the training jets are computed once and cached, and the loss is
    alpha * CE(student, y) + (1 - alpha) * T^2 * KL(softmax(t/T) || softmax(s/T)).

Writes tests/s10_results/lite_<tag>_{result.json,log.txt} and lite_<tag>_best.pt.

    python tests/s16_lite_train.py --tag d64 --D 64 --ffn-ratio 2 --pair-hidden 16,16
    python tests/s16_lite_train.py --tag d32_kd --D 32 --ffn-ratio 2 --pair-hidden 16,16 \
        --teacher erwin_m64 --teacher-ball 64 --alpha 0.5 --temp 4
"""
import argparse
import json
import os
import random
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")
HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "networks"))
from erwin_part.fpga import ErwinParTLite, LiteConfig  # noqa: E402
from jetclass_data import NUM_FEATURES, find_root, load_all  # noqa: E402
from lite_metrics import deployed_cost, local_metrics  # noqa: E402
from s10_compare import OUTDIR, WORKSPACE, build, log  # noqa: E402


def lite_call(m, pts, fts, vec, msk):
    return m(fts, v=vec, mask=msk, points=pts)


@torch.no_grad()
def predict(model, call, data, idx, device, batch=512, logits=False):
    pts, fts, vec, msk, y = data
    model.eval()
    out = []
    for i in range(0, len(idx), batch):
        b = idx[i:i + batch]
        z = call(model, pts[b].to(device), fts[b].to(device), vec[b].to(device), msk[b].to(device)).float()
        out.append((z if logits else torch.softmax(z, -1)).cpu())
    return torch.cat(out), y[idx]


def teacher_logits(tag, ball, seed, data, idx, device):
    """The teacher's logits on `idx`, cached per (teacher, seed)."""
    path = os.path.join(OUTDIR, f"teacher_{tag}_seed{seed}_logits.pt")
    if os.path.exists(path):
        cached = torch.load(path)
        if torch.equal(cached["idx"], idx):
            return cached["logits"]
    t = build("erwin", device, ball=ball)
    t.load_state_dict(torch.load(os.path.join(OUTDIR, f"{tag}_best.pt"), map_location=device))
    z, _ = predict(t, lambda m, p, f, v, k: m(f, v=v, mask=k, points=p), data, idx, device, logits=True)
    torch.save({"idx": idx, "logits": z}, path)
    return z


def ints(s):
    return tuple(int(v) for v in s.split(",")) if s else ()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    # model (LiteConfig)
    ap.add_argument("--D", type=int, default=128)
    ap.add_argument("--heads", type=int, default=0)
    ap.add_argument("--ffn-ratio", type=int, default=4)
    ap.add_argument("--embed-ratio", type=int, default=4)
    ap.add_argument("--depths", default="2,2,2")
    ap.add_argument("--strides", default="2,2")
    ap.add_argument("--flat", action="store_true", help="hierarchy=False (one level)")
    ap.add_argument("--cls-layers", type=int, default=2)
    ap.add_argument("--partition", default="morton", choices=["morton", "pt", "tree"])
    ap.add_argument("--no-shift", action="store_true")
    ap.add_argument("--ball-summary", default="off", choices=["off", "plain", "physics"])
    ap.add_argument("--pair-features", default="exact", choices=["exact", "lut", "none"])
    ap.add_argument("--pair-hidden", default="64,64")
    ap.add_argument("--norm", default="layernorm", choices=["layernorm", "batchnorm", "affine", "none"])
    # distillation
    ap.add_argument("--teacher", default="", help="s10 checkpoint tag, e.g. erwin_m64")
    ap.add_argument("--teacher-ball", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--temp", type=float, default=4.0)
    args = ap.parse_args()

    dev = (torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")) \
        if args.device == "auto" else torch.device(args.device)
    torch.set_num_threads(6)
    name = f"lite_{args.tag}"
    logp = os.path.join(OUTDIR, f"{name}_log.txt")

    data = load_all(find_root(WORKSPACE), seed=args.seed)
    n = len(data[-1])
    ntr, nva = int(0.8 * n), int(0.1 * n)
    idx = torch.arange(n)
    tr, va, te = idx[:ntr], idx[ntr:ntr + nva], idx[ntr + nva:]
    pts, fts, vec, msk, y = data

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    cfg = dict(input_dim=NUM_FEATURES, D=args.D, num_heads=args.heads, ffn_ratio=args.ffn_ratio,
               embed_ratio=args.embed_ratio, depths=ints(args.depths),
               strides=() if args.flat else ints(args.strides), hierarchy=not args.flat,
               cls_layers=args.cls_layers, partition=args.partition, shift=not args.no_shift,
               ball_summary=args.ball_summary, pair_features=args.pair_features,
               pair_hidden=ints(args.pair_hidden), norm=args.norm)
    model = ErwinParTLite(LiteConfig(**cfg)).to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    log(logp, f"=== {name} | {nparam:,} params | {cfg} | device {dev} | seed {args.seed} ===")

    t_logits = None
    if args.teacher:
        t_logits = teacher_logits(args.teacher, args.teacher_ball, args.seed, data, tr, dev)
        log(logp, f"distillation: teacher {args.teacher} (ball {args.teacher_ball}), "
                  f"alpha {args.alpha}, T {args.temp}")
    pos_in_tr = torch.full((n,), -1, dtype=torch.long)
    pos_in_tr[tr] = torch.arange(len(tr))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * ((len(tr) + args.batch - 1) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
    best, best_state, hist = -1.0, None, []
    t_start = time.perf_counter()
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = tr[torch.randperm(len(tr))]
        run, seen, t0 = 0.0, 0, time.perf_counter()
        for i in range(0, len(perm), args.batch):
            b = perm[i:i + args.batch]
            opt.zero_grad(set_to_none=True)
            z = lite_call(model, pts[b].to(dev), fts[b].to(dev), vec[b].to(dev), msk[b].to(dev))
            loss = F.cross_entropy(z, y[b].to(dev))
            if t_logits is not None:
                tz = t_logits[pos_in_tr[b]].to(dev)
                T = args.temp
                kd = F.kl_div(F.log_softmax(z / T, -1), F.softmax(tz / T, -1), reduction="batchmean") * T * T
                loss = args.alpha * loss + (1 - args.alpha) * kd
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            run += loss.item() * len(b); seen += len(b)
        probs, yy = predict(model, lite_call, data, va, dev)
        vm = local_metrics(probs.numpy(), yy.numpy())
        vloss = float(-torch.log(probs.gather(1, yy.view(-1, 1)).clamp(min=1e-12)).mean())   # plain CE
        hist.append({"epoch": ep, "loss": run / seen, "val_loss": vloss, "val_acc": vm["accuracy"],
                     "val_auc": vm["auc_macro"], "secs": time.perf_counter() - t0})
        star = ""
        if vm["accuracy"] > best:
            best, star = vm["accuracy"], "  *"
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        log(logp, f"epoch {ep:>3}/{args.epochs}  loss {run / seen:.4f}  val loss {vloss:.4f}  val acc {vm['accuracy']:.4f}  "
                  f"val AUC {vm['auc_macro']:.4f}  {time.perf_counter() - t0:5.1f}s{star}")

    model.load_state_dict(best_state)
    model = model.to("cpu")
    probs, yy = predict(model, lite_call, data, te, torch.device("cpu"))
    tm = local_metrics(probs.numpy(), yy.numpy())
    tm.update(eval_device="cpu", config=cfg, seed=args.seed, epochs=args.epochs, history=hist,
              train_minutes=(time.perf_counter() - t_start) / 60,
              distill={"teacher": args.teacher, "teacher_ball": args.teacher_ball, "alpha": args.alpha,
                       "temp": args.temp} if args.teacher else None,
              cost=deployed_cost(model, lite_call, NUM_FEATURES))
    log(logp, f"TEST acc {tm['accuracy']:.4f}  AUC {tm['auc_macro']:.4f}  | {tm['cost']['params']:,} params, "
              f"{tm['cost']['macs'] / 1e6:.1f}M MACs, {tm['cost']['peak_act_mb_fp32']:.3f} MB  "
              f"({tm['train_minutes']:.1f} min)")
    log(logp, "  rej90: " + "  ".join(f"{k} {v['rej']:.0f}{'+' if v['capped'] else ''}"
                                      for k, v in tm["rej90"].items()))
    with open(os.path.join(OUTDIR, f"{name}_result.json"), "w") as fh:
        json.dump(tm, fh, indent=1)
    torch.save(best_state, os.path.join(OUTDIR, f"{name}_best.pt"))


if __name__ == "__main__":
    main()
