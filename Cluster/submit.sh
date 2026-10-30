#!/bin/bash
# Render Cluster/jobs/job.yml and submit it. Same knobs as the Makefile, but
# plain bash - macOS gates /usr/bin/make behind the Xcode licence, so this is
# the interface that works everywhere.
#
#   ./Cluster/submit.sh smoke   MODEL=ErwinParTv2
#   ./Cluster/submit.sh tests
#   ./Cluster/submit.sh train   MODEL=ErwinParTv2 TAG=v2-10pct
#   ./Cluster/submit.sh train   MODEL=ParT        TAG=part-10pct
#   ./Cluster/submit.sh predict MODEL=ErwinParTv2 RUN_ID=<trained run id>
#   ./Cluster/submit.sh dry-run train MODEL=ParT TAG=x     # render + validate only
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
HERE="Cluster"

DRY=0
if [ "${1:-}" = "dry-run" ]; then DRY=1; shift; fi
ROLE=${1:?usage: submit.sh [dry-run] <train|predict|smoke|tests> [KEY=VALUE ...]}
shift
for kv in "$@"; do export "${kv?}"; done

NS=${NS:-cms-ml}
BRANCH=${BRANCH:-ErwinParTCluster}
IMAGE=${IMAGE:-gitlab-registry.nrp-nautilus.io/jmduarte/hbb_interaction_network:latest}
MODEL=${MODEL:-ErwinParTv2}
FEATURE=${FEATURE:-full}

# ---- SCOPE: 10% of JetClass unless deliberately overridden ------------------
DATA_FRACTION=${DATA_FRACTION:-0.1}
SAMPLES_PER_EPOCH=${SAMPLES_PER_EPOCH:-1024000}
SAMPLES_PER_EPOCH_VAL=${SAMPLES_PER_EPOCH_VAL:-128000}
EPOCHS=${EPOCHS:-50}
DEADLINE=${DEADLINE:-172800}
MEM=${MEM:-80Gi}
CPU=${CPU:-8}
GPUS=${GPUS:-NVIDIA-GeForce-RTX-4090, NVIDIA-A40, NVIDIA-L40, NVIDIA-GeForce-RTX-3090}
EXTRA_ARGS=${EXTRA_ARGS:-}
RUN_DIR=${RUN_DIR:-}
TEST_DATA=${TEST_DATA:-/output/testdata/JetClass_example_100k.root}
SKIP_PREFLIGHT=${SKIP_PREFLIGHT:-0}

SHA=$(git rev-parse --short HEAD)
TAG=${TAG:-dev}

case "$ROLE" in
  smoke)   TAG=smoke; DATA_FRACTION=0.001; EPOCHS=2
           SAMPLES_PER_EPOCH=51200; SAMPLES_PER_EPOCH_VAL=12800; DEADLINE=86400
           # Two epochs so run_train.sh can kill the run after epoch 0 and
           # check that it resumes. Memory, CPUs and loader workers match
           # training: smaller ones hid the /dev/shm crash 6 workers caused.
           # Smoke only proves the code runs, so it may also take an A10
           # (24 GB, slower; its timings do not transfer to training cards).
           GPUS="$GPUS, NVIDIA-A10" ;;
  tests)   TAG=tests; SKIP_PREFLIGHT=1; DEADLINE=7200 ;;
  predict) : "${RUN_ID:?predict needs RUN_ID=<the training run id>}"
           RUN_DIR=${RUN_DIR:-/output/runs/${RUN_ID}}
           TRAINED=$RUN_ID; RUN_ID="${RUN_ID}-pred"; DEADLINE=43200 ;;
  train)   : ;;
  *) echo "unknown role: $ROLE" >&2; exit 2 ;;
esac
RUN_ID=${RUN_ID:-$(echo "$(date +%Y%m%d)-${SHA}-${TAG}" | tr '[:upper:]' '[:lower:]')}

# ---- the guard that matters ------------------------------------------------
# A job clones $BRANCH from GitHub. Local edits that are not committed AND
# pushed are simply not in the run - the single easiest way to spend a day
# testing code you did not change.
if [ "$DRY" = 0 ]; then
    if [ -n "$(git status --porcelain)" ]; then
        echo "REFUSING: working tree is dirty; the cluster clones $BRANCH from GitHub." >&2
        git status --short >&2; exit 1
    fi
    cur=$(git rev-parse --abbrev-ref HEAD)
    [ "$cur" = "$BRANCH" ] || { echo "REFUSING: on '$cur' but the job clones '$BRANCH'." >&2; exit 1; }
    git fetch -q origin "$BRANCH" 2>/dev/null || true
    if [ "$(git rev-parse HEAD)" != "$(git rev-parse "origin/$BRANCH" 2>/dev/null || echo none)" ]; then
        echo "REFUSING: HEAD != origin/$BRANCH. Push first, or the cluster runs older code." >&2; exit 1
    fi
    echo "OK: $BRANCH @ $SHA is clean and pushed."
fi

render() {
  sed -e "s|@@ROLE@@|${ROLE}|g" -e "s|@@RUN_ID@@|${RUN_ID}|g" -e "s|@@MODEL@@|${MODEL}|g" \
      -e "s|@@FEATURE@@|${FEATURE}|g" -e "s|@@BRANCH@@|${BRANCH}|g" -e "s|@@IMAGE@@|${IMAGE}|g" \
      -e "s|@@DATA_FRACTION@@|${DATA_FRACTION}|g" -e "s|@@SAMPLES_PER_EPOCH@@|${SAMPLES_PER_EPOCH}|g" \
      -e "s|@@SAMPLES_PER_EPOCH_VAL@@|${SAMPLES_PER_EPOCH_VAL}|g" -e "s|@@EPOCHS@@|${EPOCHS}|g" \
      -e "s|@@DEADLINE@@|${DEADLINE}|g" -e "s|@@EXTRA_ARGS@@|${EXTRA_ARGS}|g" \
      -e "s|@@RUN_DIR@@|${RUN_DIR}|g" -e "s|@@TEST_DATA@@|${TEST_DATA}|g" \
      -e "s|@@SKIP_PREFLIGHT@@|${SKIP_PREFLIGHT}|g" \
      -e "s|@@MEM@@|${MEM}|g" -e "s|@@CPU@@|${CPU}|g" -e "s|@@GPUS@@|${GPUS}|g" "$HERE/jobs/job.yml"
}

echo ">>> role=$ROLE model=$MODEL run-id=$RUN_ID scope: data-fraction=$DATA_FRACTION"
echo "    samples/epoch=$SAMPLES_PER_EPOCH epochs=$EPOCHS mem=$MEM cpu=$CPU
    gpus: $GPUS${TRAINED:+  reading /output/runs/$TRAINED}"
if [ "$DRY" = 1 ]; then
    render | kubectl -n "$NS" apply --dry-run=client -f - && echo "OK: manifest valid (nothing submitted)"
else
    render | kubectl -n "$NS" apply -f -
    echo ">>> kubectl -n $NS logs -f job/vleept-$RUN_ID"
fi
