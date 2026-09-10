#!/bin/bash
# S11 ablations 2-6. Each varies exactly one thing against the m=8 baseline
# (tests/s10_results/erwin_m8_result.json): same data, split, seed, schedule,
# 12 epochs, level-0 ball size 8.
#
#   rot0      rotate=0      no cross-ball mixing at all - balls never exchange
#                           information except through pooling
#   noU       no pair bias  drop ParT's learned interaction bias entirely
#   noD       no dist bias  drop Erwin Eq. 10's distance decay
#   fine      readout=fine  class attention sees constituents only, not subjets
#   coarse    readout=coarse class attention sees the bottleneck only
#   logpolar  warped tree   partition in a radially-warped space that expands
#                           the collinear core
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${WEAVER_PY:-/opt/homebrew/Caskroom/miniconda/base/envs/weaver/bin/python}
E=${EPOCHS:-12}
run () { echo "=== $1 ($(date +%H:%M)) ==="; shift
         $PY tests/s10_compare.py --models erwin --epochs $E --batch 256 --lr 1e-3 "$@" \
           || echo "FAILED"; }
run "rot0"     --tag _rot0     --rotate 0
run "noU"      --tag _noU      --no-pair-bias
run "noD"      --tag _noD      --no-dist-bias
run "fine"     --tag _fine     --readout fine
run "coarse"   --tag _coarse   --readout coarse
run "logpolar" --tag _logpolar --tree-space logpolar
echo "=== ablations complete $(date +%H:%M) ==="
