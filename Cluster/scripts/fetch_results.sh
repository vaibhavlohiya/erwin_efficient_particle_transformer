#!/bin/bash
# Pull a finished run off the output PVC into the workspace outputs/ directory.
# Copies only; nothing on the PVC is removed.
#
#   bash Cluster/scripts/fetch_results.sh <pod-name> <run-id> [dest]
set -euo pipefail
POD=${1:?usage: fetch_results.sh <pod-name> <run-id> [dest]}
RUN_ID=${2:?usage: fetch_results.sh <pod-name> <run-id> [dest]}
NS=${NS:-cms-ml}
DEST=${3:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/outputs/cluster}

mkdir -p "$DEST"
echo ">>> $POD:/output/runs/$RUN_ID -> $DEST/$RUN_ID"
kubectl -n "$NS" cp "$POD:/output/runs/$RUN_ID" "$DEST/$RUN_ID"
echo ">>> fetched:"; ls -la "$DEST/$RUN_ID" | head -20
