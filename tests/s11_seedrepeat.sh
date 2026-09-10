#!/bin/bash
# S11 seed repeat - re-run the load-bearing arms at a second seed.
#
# --seed changes BOTH the model init and the data split, so the seed-1 test set
# is different jets. That means no paired test against seed 0; the comparison is
# baseline-vs-ablation *within* seed 1, which is the design that matters - it
# asks whether the same conclusion reappears, not whether the same number does.
#
# Four arms, chosen because each carries a claim:
#   base      the reference for this seed
#   noU       the §1.1 correction: does dropping ParT's U really cost nothing?
#   minimal   the headline simplification: U + distance bias + rotation all out
#   logpolar  the largest positive delta at seed 0 (+0.0040, p=0.192) - real?
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${WEAVER_PY:-/opt/homebrew/Caskroom/miniconda/base/envs/weaver/bin/python}
S=${SEED:-1}
E=${EPOCHS:-12}
run () { echo "=== $1 seed=$S ($(date +%H:%M)) ==="; local tag=$1; shift
         $PY tests/s10_compare.py --models erwin --epochs $E --batch 256 --lr 1e-3 \
             --seed $S --tag "_s${S}_${tag}" "$@" || echo "FAILED"; }
run base
run noU      --no-pair-bias
run minimal  --no-pair-bias --no-dist-bias --rotate 0
run logpolar --tree-space logpolar
echo "=== seed repeat complete $(date +%H:%M) ==="
