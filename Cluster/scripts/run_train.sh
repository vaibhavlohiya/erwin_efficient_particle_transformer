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

# ---- provenance + resume guard ---------------------------------------------
# The manifest is written PER ATTEMPT and never overwrites an earlier one, so a
# resumed run keeps its original start time instead of losing it.
#
# The guard refuses a run directory created by a DIFFERENT model: the resume
# path below would otherwise pass --load-epoch and load another architecture's
# weights, which fails confusingly at best and trains silently wrong at worst.
python - "$RUN_DIR" <<'PYPROV'
import datetime, glob, json, os, re, subprocess, sys

run_dir = sys.argv[1]
model = os.environ.get("MODEL")

def sh(*c):
    try:
        return subprocess.check_output(c, text=True).strip()
    except Exception:
        return None

existing = sorted(glob.glob(os.path.join(run_dir, "run_manifest*.json")))
if existing:
    try:
        prior = json.load(open(existing[0])).get("model")
    except Exception:
        prior = None
    if prior and prior != model:
        sys.exit(
            "FATAL: %s was created by model %r, but MODEL=%r.\n"
            "       Resuming would load %s's weights via --load-epoch.\n"
            "       Use a different TAG (the run id embeds it), or a different RUN_ID."
            % (run_dir, prior, model, prior))

attempt = len(existing) + 1
rec = {
    "run_id": os.environ.get("RUN_ID"),
    "attempt": attempt,
    "started": datetime.datetime.now().isoformat(timespec="seconds"),
    "model": model,
    "feature": os.environ.get("FEATURE", "full"),
    "git_sha": sh("git", "rev-parse", "HEAD"),
    "git_branch": sh("git", "rev-parse", "--abbrev-ref", "HEAD"),
    "image": os.environ.get("IMAGE"),
    "scope": {k: os.environ.get(k) for k in
              ("DATA_FRACTION", "SAMPLES_PER_EPOCH", "SAMPLES_PER_EPOCH_VAL", "EPOCHS")},
    "extra_args": os.environ.get("EXTRA_ARGS", ""),
}
ckpts = glob.glob(os.path.join(run_dir, "net_epoch-*_state.pt"))
if ckpts:
    rec["resumed_from_epoch"] = max(
        int(re.search(r"net_epoch-(\d+)_state", c).group(1)) for c in ckpts)
for mod, keys in (("torch", ("__version__",)), ("weaver", ("__version__",))):
    try:
        m = __import__(mod)
        rec[mod] = getattr(m, "__version__", "?")
        if mod == "torch":
            rec["cuda"] = m.version.cuda
    except Exception:
        pass

name = "run_manifest.json" if attempt == 1 else "run_manifest.attempt%d.json" % attempt
path = os.path.join(run_dir, name)
with open(path, "w") as fh:
    json.dump(rec, fh, indent=2)
print("provenance -> %s (attempt %d)" % (path, attempt))
PYPROV

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
