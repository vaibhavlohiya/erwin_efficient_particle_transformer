#!/bin/bash
# Prediction entrypoint. Needs RUN_DIR - the directory a training run wrote,
# as recorded in its run_manifest.json and in Cluster/docs/RUNS.md.
#
# RUN_DIR is what makes this work at all: test_JetClass.sh's historical path
# re-expanded {auto} to a fresh timestamp and could never match a trained run.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

MODEL=${MODEL:?MODEL must be set}
FEATURE=${FEATURE:-full}
RUN_DIR=${RUN_DIR:?RUN_DIR must point at the training output directory}
export RUN_DIR

# Same 10% default as training: the test slice is deterministic (weaver disables
# the shuffle for for_training=False), so every model scores the identical jets.
DATA_FRACTION=${DATA_FRACTION:-0.1}

if ! ls "$RUN_DIR"/net*.pt >/dev/null 2>&1; then
    echo "FATAL: no net*.pt checkpoints under $RUN_DIR" >&2
    ls -la "$RUN_DIR" >&2 || true
    exit 1
fi
echo ">>> predicting from $RUN_DIR"
ls -1 "$RUN_DIR"/net*best*.pt 2>/dev/null || echo "(no _best_ checkpoint; weaver will use the prefix as given)"

set -x
exec ./test_JetClass.sh "$MODEL" "$FEATURE" \
    --data-fraction "$DATA_FRACTION" \
    ${EXTRA_ARGS:-}
