#!/bin/bash
# S11 growing-ball arms. Instead of the hierarchy (pool to subjets, then a
# single-ball bottleneck), keep all 64 constituents and widen the ball layer by
# layer. Same 6 attention blocks as the m=8 baseline
# (tests/s10_results/erwin_m8_result.json, and erwin_s1_base at SEED=1); same
# data, split, schedule, 12 epochs. Read out from the last level.
#
#   grow8   ball 8 -> 16 -> 32, depths 2,2,2
#   grow2   ball 2 -> 4 -> 8 -> 16 -> 32, depths 1,1,1,1,2
#
# The top level stops at two balls of 32 rather than one of 64: a 64-wide level
# on top of the 16 and 32 ones needed 9.4 GB at batch 256 and swapped a 16 GB
# Mac to a standstill, against 5.9 GB here (and 3.5 GB for the hierarchy). The
# halves still meet through the rotated tree (every level has 2+ blocks at the
# top, so one rotated block) and through the class-attention readout, which
# sees all 64 tokens. No pooling means no subjets: this measures "growing local
# receptive field" against "hierarchy of subjets".
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${WEAVER_PY:-/opt/homebrew/Caskroom/miniconda/base/envs/weaver/bin/python}
E=${EPOCHS:-12}
SEED=${SEED:-0}
S=$([ "$SEED" = 0 ] && echo "" || echo "_s${SEED}_")
run () { echo "=== $1 seed=$SEED ($(date +%H:%M)) ==="; local tag=$1; shift
         $PY tests/s10_compare.py --models erwin --epochs $E --batch 256 --lr 1e-3 \
           --seed $SEED --tag "${S:-_}${tag}" --readout coarse "$@" || echo "FAILED"; }
run grow8 --ball-sizes 8,16,32      --strides 1,1   --depths 2,2,2
run grow2 --ball-sizes 2,4,8,16,32  --strides 1,1,1,1 --depths 1,1,1,1,2
echo "=== grow arms complete $(date +%H:%M) ==="
