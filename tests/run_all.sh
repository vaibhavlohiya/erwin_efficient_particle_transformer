#!/bin/bash
# Full ErwinParTv2 verification suite (S0-S9).
#
# Two interpreters on purpose: S2 checks our pure-torch ball tree against Erwin's
# compiled `balltree`, which is installed alongside torch in the system python,
# while everything else needs weaver-core from the conda env. Override either
# with WEAVER_PY / ORACLE_PY.
set -uo pipefail
cd "$(dirname "$0")/.."
WEAVER_PY=${WEAVER_PY:-/opt/homebrew/Caskroom/miniconda/base/envs/weaver/bin/python}
ORACLE_PY=${ORACLE_PY:-python3}
QUICK=${QUICK:-0}

tot=0; fail=0
run() {  # run <label> <interpreter> <script...>
  local label=$1 py=$2; shift 2
  local out code n
  out=$("$py" "$@" 2>&1); code=$?
  n=$(grep -c '\[OK ' <<<"$out")
  tot=$((tot+n)); [ $code -ne 0 ] && { fail=$((fail+1)); printf '%s\n' "$out" | tail -25; }
  printf "  %-20s %-5s %3d checks\n" "$label" "$([ $code -eq 0 ] && echo PASS || echo FAIL)" "$n"
}

echo "=================== ErwinParTv2 verification ==================="
run s0_env           "$WEAVER_PY" tests/s0_env.py
run s1_geometry      "$WEAVER_PY" tests/s1_geometry.py
run s2_tree_parity   "$ORACLE_PY" tests/s2_tree_parity.py
run s3_s4_invariance "$WEAVER_PY" tests/s3_s4_invariance.py
run s5_pair_parity   "$WEAVER_PY" tests/s5_pair_parity.py
run s6_leakage       "$WEAVER_PY" tests/s6_leakage.py
run s7_gradients     "$WEAVER_PY" tests/s7_gradients.py
run s8_complexity    "$WEAVER_PY" tests/s8_complexity.py --max-l $([ "$QUICK" = 1 ] && echo 128 || echo 512)
run s9_overfit       "$WEAVER_PY" tests/s9_overfit.py $([ "$QUICK" = 1 ] && echo "--jets 256 --epochs 40")
echo "  ------------------------------------------------------------"
printf "  %-20s %-5s %3d checks\n" "TOTAL" "$([ $fail -eq 0 ] && echo PASS || echo FAIL)" "$tot"
[ $fail -eq 0 ] || echo "  $fail script(s) failed"
exit $fail
