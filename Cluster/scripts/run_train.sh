#!/bin/bash
# Training entrypoint. Wraps ./train_JetClass.sh with a STABLE output directory,
# the 10% default scope, and automatic resume.
#
# Why not just call train_JetClass.sh: its --model-prefix contains {auto}, which
# weaver expands to <timestamp>_<config>_<optim>_lr<lr>_batch<batch>. That is a
# new directory on every start, so (a) a predict run can never find what
# training wrote and (b) --load-epoch has nothing stable to resume from. Passing
# --model-prefix again here overrides it: train_JetClass.sh forwards "${@:3}"
# AFTER its own flags, and argparse keeps the last occurrence.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

MODEL=${MODEL:?MODEL must be set (ParT, ErwinParTv2, ...)}
FEATURE=${FEATURE:-full}
RUN_ID=${RUN_ID:?RUN_ID must be set}
OUTPUT_PATH=${OUTPUT_PATH:-/output}
RUN_DIR="${OUTPUT_PATH}/runs/${RUN_ID}"

# ---- scope -----------------------------------------------------------------
# DEFAULT IS 10% OF JETCLASS. --data-fraction caps what each epoch reads from
# every file (class balance preserved); samples-per-epoch caps the compute so an
# epoch is roughly one pass over that 10% rather than ten passes. Raising either
# is an explicit decision - see Cluster/docs/RUNBOOK.md.
# Exported, not just assigned: the provenance step below runs in a subprocess
# and would otherwise record nulls for the very fields that define the run.
export DATA_FRACTION=${DATA_FRACTION:-0.1}
export SAMPLES_PER_EPOCH=${SAMPLES_PER_EPOCH:-1024000}
export SAMPLES_PER_EPOCH_VAL=${SAMPLES_PER_EPOCH_VAL:-128000}
export EPOCHS=${EPOCHS:-50}
export MODEL FEATURE RUN_ID

mkdir -p "$RUN_DIR"
export LOG_DIR="$RUN_DIR"          # train_JetClass.sh puts --log + tensorboard here

# ---- provenance ------------------------------------------------------------
python - "$RUN_DIR" <<'PY'
import json, os, subprocess, sys, datetime
run_dir = sys.argv[1]
def sh(*c):
    try: return subprocess.check_output(c, text=True).strip()
    except Exception: return None
rec = {
    "run_id": os.environ.get("RUN_ID"),
    "started": datetime.datetime.now().isoformat(timespec="seconds"),
    "model": os.environ.get("MODEL"), "feature": os.environ.get("FEATURE", "full"),
    "git_sha": sh("git", "rev-parse", "HEAD"),
    "git_branch": sh("git", "rev-parse", "--abbrev-ref", "HEAD"),
    "image": os.environ.get("IMAGE"),
    "scope": {k: os.environ.get(k) for k in
              ("DATA_FRACTION", "SAMPLES_PER_EPOCH", "SAMPLES_PER_EPOCH_VAL", "EPOCHS")},
    "extra_args": os.environ.get("EXTRA_ARGS", ""),
}
try:
    import torch; rec["torch"] = torch.__version__; rec["cuda"] = torch.version.cuda
except Exception: pass
try:
    import weaver; rec["weaver"] = getattr(weaver, "__version__", "?")
except Exception: pass
path = os.path.join(run_dir, "run_manifest.json")
with open(path, "w") as fh: json.dump(rec, fh, indent=2)
print("provenance ->", path)
PY

# ---- resume ----------------------------------------------------------------
# weaver writes ${model_prefix}_epoch-N_state.pt each epoch. If the job was
# preempted, pick up from the highest N rather than starting over.
resume=()
last=$(ls "$RUN_DIR"/net_epoch-*_state.pt 2>/dev/null \
       | sed -n 's/.*net_epoch-\([0-9][0-9]*\)_state\.pt/\1/p' | sort -n | tail -1 || true)
if [[ -n ${last:-} ]]; then
    echo ">>> resuming from epoch ${last} in ${RUN_DIR}"
    resume=(--load-epoch "$last")
else
    echo ">>> fresh run in ${RUN_DIR}"
fi

set -x
exec ./train_JetClass.sh "$MODEL" "$FEATURE" \
    --model-prefix "${RUN_DIR}/net" \
    --data-fraction "$DATA_FRACTION" \
    --samples-per-epoch "$SAMPLES_PER_EPOCH" \
    --samples-per-epoch-val "$SAMPLES_PER_EPOCH_VAL" \
    --num-epochs "$EPOCHS" \
    ${resume[@]+"${resume[@]}"} \
    ${EXTRA_ARGS:-}
