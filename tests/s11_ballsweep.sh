#!/bin/bash
# S11 ablation 1 - level-0 ball size.
#
# Sweeps the constituent-level ball size m while holding everything else fixed:
# levels 1 and 2 stay at 8 and 16, strides [2,2], depths [2,2,2], same data,
# same split, same seed, same schedule. Parameter count is identical at every m
# (2.21M), so this isolates the geometry and nothing else.
#
#   m=8   production config, 8 balls of 8
#   m=16  4 balls of 16
#   m=32  2 balls of 32
#   m=64  ONE ball - full dense attention over all constituents, i.e. the
#         no-locality control: what the hierarchy gives up by being local.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${WEAVER_PY:-/opt/homebrew/Caskroom/miniconda/base/envs/weaver/bin/python}
EPOCHS=${EPOCHS:-12}
for M in 8 16 32 64; do
  echo "=== ball size $M ($(date +%H:%M)) ==="
  $PY tests/s10_compare.py --models erwin --ball-size $M --tag "_m$M" \
      --epochs $EPOCHS --batch 256 --lr 1e-3 || echo "m=$M FAILED"
done
echo "=== sweep complete $(date +%H:%M) ==="
