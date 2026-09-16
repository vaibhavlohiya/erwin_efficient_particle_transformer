#!/bin/bash
# Copy the 100k JetClass sample onto the output PVC, once, so Cluster/jobs/tests.yml
# can find it. The file is 136 MB and .gitignore excludes *.root, so it cannot
# travel with the clone.
#
# Needs a running pod that mounts the output PVC - Cluster/pods/probe.yml or
# jupyter.yml will do.
#
#   bash Cluster/scripts/stage_testdata.sh <pod-name>
set -euo pipefail
POD=${1:?usage: stage_testdata.sh <pod-name>}
NS=${NS:-cms-ml}
SRC=${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/JetClass_example_100k.root}

[ -f "$SRC" ] || { echo "FATAL: $SRC not found" >&2; exit 1; }
echo ">>> copying $(du -h "$SRC" | cut -f1) to $POD:/output/testdata/"
kubectl -n "$NS" exec "$POD" -- mkdir -p /output/testdata
kubectl -n "$NS" cp "$SRC" "$POD:/output/testdata/JetClass_example_100k.root"
kubectl -n "$NS" exec "$POD" -- ls -la /output/testdata/
