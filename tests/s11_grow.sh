#!/bin/bash
# S11 growing-ball arms. Instead of the hierarchy (pool to subjets, then a
# single-ball bottleneck), keep all 64 constituents and widen the ball layer by
# layer until the last layer is full attention. Same 6 attention blocks as the
# m=8 baseline (tests/s10_results/erwin_m8_result.json, and erwin_s1_base at
# SEED=1); same data, split, schedule, 12 epochs. Read out from the last level.
#
#   grow8   ball 8 -> 16 -> 32 -> 64, depths 2,2,1,1
#   grow2   ball 2 -> 4 -> 8 -> 16 -> 32 -> 64, one block each
#
# No pooling means no subjets: this measures "growing receptive field" against
# "hierarchy of subjets". The last ball is full attention, so expect ~m=64 cost.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${WEAVER_PY:-/opt/homebrew/Caskroom/miniconda/base/envs/weaver/bin/python}
E=${EPOCHS:-12}
SEED=${SEED:-0}
S=$([ "$SEED" = 0 ] && echo "" || echo "_s$SEED")
run () { echo "=== $1 seed=$SEED ($(date +%H:%M)) ==="; local tag=$1; shift
         $PY tests/s10_compare.py --models erwin --epochs $E --batch 256 --lr 1e-3 \
           --seed $SEED --tag "_${tag}$S" --readout coarse "$@" || echo "FAILED"; }
run grow8 --ball-sizes 8,16,32,64       --strides 1,1,1     --depths 2,2,1,1
run grow2 --ball-sizes 2,4,8,16,32,64   --strides 1,1,1,1,1 --depths 1,1,1,1,1,1
echo "=== grow arms complete $(date +%H:%M) ==="
