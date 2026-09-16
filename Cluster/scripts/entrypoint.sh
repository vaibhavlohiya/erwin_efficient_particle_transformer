#!/bin/bash
# Container entrypoint for every Cluster/jobs/*.yml.
#
# The job manifests clone this repo and then exec this file, so the YAML stays
# small and this logic is reviewable, lintable and testable as an ordinary shell
# script instead of a string embedded in a block scalar.
#
# Required env: ROLE (train|predict|smoke|tests), plus whatever that role needs.
set -euo pipefail

echo "=== ErwinParTv2 cluster entrypoint ==============================="
echo "role       : ${ROLE:?ROLE must be train|predict|smoke|tests}"
echo "commit     : $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "branch     : $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
echo "run id     : ${RUN_ID:-<unset>}"
echo "gpu        : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo none)"
echo "=================================================================="

# Floors only; torch comes from the image (see requirements.txt).
if [ -f requirements.txt ]; then
    pip install --quiet -r requirements.txt
fi

# Checkpoints and logs land on the output PVC. train_JetClass.sh writes to the
# relative path ./training, so this symlink is what redirects it.
if [ -e ./training ] && [ ! -L ./training ]; then
    echo "WARNING: ./training exists and is not a symlink; leaving it untouched."
else
    ln -sfn "${OUTPUT_PATH:-/output}" ./training
fi

# Fail on the plugin contract and the numerics in seconds, not after the
# dataset loads. Skipped only for the roles that are themselves the check.
if [ "${SKIP_PREFLIGHT:-0}" != "1" ]; then
    python Cluster/scripts/preflight.py
fi

case "$ROLE" in
    train)   exec bash Cluster/scripts/run_train.sh   ;;
    predict) exec bash Cluster/scripts/run_predict.sh ;;
    smoke)   exec bash Cluster/scripts/run_train.sh   ;;
    tests)   exec bash Cluster/scripts/run_tests.sh   ;;
    *) echo "unknown ROLE=$ROLE" >&2; exit 2 ;;
esac
