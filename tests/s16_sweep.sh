#!/bin/bash
# FPGA-compression study, step 3: local ErwinParTLite sweeps (100k-jet sample only).
#
#   bash tests/s16_sweep.sh variants   # stage 0: pick the variant to carry forward (D = 64)
#   SEED=1 bash tests/s16_sweep.sh repeat   # the deciding stage-0 arms at another seed
#   bash tests/s16_sweep.sh decide          # the same, after the self-pair fix, seeds 0 and 1
#
# variants/repeat (v64_*) predate the self-pair fix in PairNet; decide (d64_*) is after it.
#
# Each run writes tests/s10_results/lite_<tag>_{result.json,log.txt}. Runs are
# sequential; wrap in `caffeinate -i -s` on mains power for long stages.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${WEAVER_PY:-/opt/homebrew/Caskroom/miniconda/base/envs/weaver/bin/python}
SEED=${SEED:-0}
run () { local tag=$1; shift
         echo "=== $tag seed=$SEED ($(date +%H:%M)) ==="
         $PY tests/s16_lite_train.py --tag "$tag" --seed "$SEED" "$@" 2>&1 | grep -E "TEST|rej90|===" \
           || echo "FAILED $tag"; }
D64="--D 64 --ffn-ratio 2 --pair-hidden 16,16"

case "${1:?usage: s16_sweep.sh variants}" in
  variants)
    run v64_morton        $D64
    run v64_morton_phys   $D64 --ball-summary physics
    run v64_tree          $D64 --partition tree
    run v64_ptflat        $D64 --partition pt --flat --depths 6 --ball-summary plain
    run v64_phys_lut      $D64 --ball-summary physics --pair-features lut
    run v64_phys_lut_bn   $D64 --ball-summary physics --pair-features lut --norm batchnorm
    run v64_phys_lut_aff  $D64 --ball-summary physics --pair-features lut --norm affine
    ;;
  repeat)   # stage 0 at a second seed for the arms that decide the carried-forward variant
    run v64_morton_s$SEED        $D64
    run v64_morton_phys_s$SEED   $D64 --ball-summary physics
    run v64_phys_lut_s$SEED      $D64 --ball-summary physics --pair-features lut
    run v64_phys_lut_bn_s$SEED   $D64 --ball-summary physics --pair-features lut --norm batchnorm
    ;;
  decide)   # after excluding self-pairs from the pair norm: the deciding arms, seeds 0 and 1
    for SEED in 0 1; do
      run d64_morton_s$SEED        $D64
      run d64_phys_s$SEED          $D64 --ball-summary physics
      run d64_phys_bn_s$SEED       $D64 --ball-summary physics --norm batchnorm
      run d64_phys_lut_s$SEED      $D64 --ball-summary physics --pair-features lut
      run d64_phys_lut_bn_s$SEED   $D64 --ball-summary physics --pair-features lut --norm batchnorm
    done
    ;;
  *) echo "unknown stage $1" >&2; exit 2 ;;
esac
echo "=== stage $1 complete $(date +%H:%M) ==="
