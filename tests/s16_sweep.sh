#!/bin/bash
# FPGA-compression study, step 3: local ErwinParTLite sweeps (100k-jet sample only).
#
#   bash tests/s16_sweep.sh variants   # stage 0: pick the variant to carry forward (D = 64)
#   SEED=1 bash tests/s16_sweep.sh repeat   # the deciding stage-0 arms at another seed
#   bash tests/s16_sweep.sh decide          # the same, after the self-pair fix, seeds 0 and 1
#   bash tests/s16_sweep.sh width           # stage 1: D in {128, 64, 32, 16} x FFN {2, 4}
#   bash tests/s16_sweep.sh depth           # stage 2: depth cuts
#   bash tests/s16_sweep.sh kd              # step 4: distillation from ErwinParTv2
#   bash tests/s16_sweep.sh qat             # step 5: 8 / 6 / 4 bits, attention floor off or 8
#   bash tests/s16_sweep.sh particles       # 32 / 16 particles per jet
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
         $PY tests/s16_lite_train.py --tag "$tag" --seed "$SEED" "$@" 2>&1 | grep -E "TEST|rej90|===|QAT|post-training" \
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
  width)    # stage 1: width x FFN ratio for the carried-forward variant (physics summaries,
            # lookup-table features, BatchNorm); pair net scaled with D. D = 64 / FFN 2 is decide's.
    V="--ball-summary physics --pair-features lut --norm batchnorm"
    run w128_f4   --D 128 --ffn-ratio 4 --pair-hidden 64,64 $V
    run w64_f4    --D 64  --ffn-ratio 4 --pair-hidden 16,16 $V
    run w32_f2    --D 32  --ffn-ratio 2 --pair-hidden 16,16 $V
    run w32_f4    --D 32  --ffn-ratio 4 --pair-hidden 16,16 $V
    run w16_f2    --D 16  --ffn-ratio 2 --pair-hidden 8,8   $V
    run w16_f4    --D 16  --ffn-ratio 4 --pair-hidden 8,8   $V
    ;;
  depth)    # stage 2: fewer blocks per level / class-attention layers at D = 64 and 32
    V="--ball-summary physics --pair-features lut --norm batchnorm"
    run dp64_111       $D64 --depths 1,1,1 $V
    run dp64_cls1      $D64 --cls-layers 1 $V
    run dp32f4_111     --D 32 --ffn-ratio 4 --pair-hidden 16,16 --depths 1,1,1 $V
    run dp32f4_cls1    --D 32 --ffn-ratio 4 --pair-hidden 16,16 --cls-layers 1 $V
    ;;
  kd)       # step 4: distillation from ErwinParTv2 (erwin_m64, seed-0 split) at alpha 0.5, T 4
    V="--ball-summary physics --pair-features lut --norm batchnorm --teacher erwin_m64 --teacher-ball 64"
    run kd64_f2        $D64 $V
    run kd32_f4        --D 32 --ffn-ratio 4 --pair-hidden 16,16 $V
    run kd32_f2        --D 32 --ffn-ratio 2 --pair-hidden 16,16 $V
    run kd16_f4        --D 16 --ffn-ratio 4 --pair-hidden 8,8 $V
    run kd32_f4_a9t2   --D 32 --ffn-ratio 4 --pair-hidden 16,16 $V --alpha 0.9 --temp 2
    ;;
  qat)      # step 5: quantization-aware training from the two candidates' float checkpoints;
            # 6 epochs at lr 3e-4, distance capped at 16 (changes neither float model by > 0.0001)
    Q="--epochs 6 --lr 3e-4 --dist-cap 16"
    for INIT in d64_phys_lut_bn_s0 dp32f4_111; do
      S=${INIT%%_*}                       # d64 / dp32f4
      run q_${S}_b8        --init $INIT $Q --quant-bits 8
      run q_${S}_b6        --init $INIT $Q --quant-bits 6
      run q_${S}_b6_f8     --init $INIT $Q --quant-bits 6 --attn-floor 8
      run q_${S}_b4        --init $INIT $Q --quant-bits 4
      run q_${S}_b4_f8     --init $INIT $Q --quant-bits 4 --attn-floor 8
    done
    ;;
  particles) # trigger realism: only the hardest 32 / 16 particles per jet (64 = the decide/depth runs)
    V="--ball-summary physics --pair-features lut --norm batchnorm --dist-cap 16"
    for L in 32 16; do
      run n${L}_d64    $D64 --seq-len $L $V
      run n${L}_d32    --D 32 --ffn-ratio 4 --pair-hidden 16,16 --depths 1,1,1 --seq-len $L $V
    done
    ;;
  *) echo "unknown stage $1" >&2; exit 2 ;;
esac
echo "=== stage $1 complete $(date +%H:%M) ==="
