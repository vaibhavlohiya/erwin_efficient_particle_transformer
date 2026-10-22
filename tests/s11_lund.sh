#!/bin/bash
# S11 Lund arms. Each adds PLuM-style Lund splitting tokens to the m=8 baseline
# (tests/s10_results/erwin_m8_result.json) and varies one thing: same data,
# split, seed, schedule, 12 epochs, level-0 ball size 8.
#
#   lund          kT tokens in the bottleneck ball (PLuM's recipe; the default
#                 ErwinParTv2Lund config)
#   lund_readout  the same tokens, class attention only - they never attend to
#                 the subjets, which PLuM credits for the gain
#   lund_ca       C/A instead of kT (PLuM saw no gain from C/A)
#   lund_kt0      drop splittings with ln(kT/GeV) <= 0, the hadronisation-
#                 robustness cut
#
# PLuM's gains are +7-12% in background rejection; at 100k jets that is inside
# seed noise, so read these as a plumbing check and run SEED=1 before believing
# a difference. The real test is the cluster run.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${WEAVER_PY:-/opt/homebrew/Caskroom/miniconda/base/envs/weaver/bin/python}
E=${EPOCHS:-12}
SEED=${SEED:-0}
S=$([ "$SEED" = 0 ] && echo "" || echo "_s$SEED")
run () { echo "=== $1 ($(date +%H:%M)) ==="; local tag=$1; shift
         $PY tests/s10_compare.py --models erwin --epochs $E --batch 256 --lr 1e-3 \
           --seed $SEED --tag "_${tag}$S" "$@" || echo "FAILED"; }
run "lund"         --lund bottleneck
run "lund_readout" --lund readout
run "lund_ca"      --lund bottleneck --lund-alg ca
run "lund_kt0"     --lund bottleneck --lund-ktcut 0
echo "=== Lund arms complete $(date +%H:%M) ==="
