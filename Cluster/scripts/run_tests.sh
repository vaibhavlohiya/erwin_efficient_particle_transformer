#!/bin/bash
# S0-S9 verification suite, on cluster hardware.
#
# Two things differ from a laptop run:
#
#  1. Data. The suite reads JetClass_example_100k.root, which .gitignore
#     excludes (*.root), so a fresh clone does not have it. Stage it once on the
#     output PVC and point TEST_DATA at it; this links it where the tests look.
#  2. Interpreters. run_all.sh defaults to the laptop's conda path for the
#     weaver env and `python3` for the ball-tree oracle. In the container both
#     are just `python` - but the oracle needs the compiled `balltree` package,
#     which the image does not ship. If it cannot be installed, S2 fails and
#     that failure is EXPECTED; S2 is the tree-parity check against Erwin's
#     reference implementation and remains a local gate.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

TEST_DATA=${TEST_DATA:-/output/testdata/JetClass_example_100k.root}
if [ -f "$TEST_DATA" ]; then
    ln -sfn "$TEST_DATA" ./JetClass_example_100k.root
    echo ">>> test data linked from $TEST_DATA"
else
    echo "WARNING: $TEST_DATA not found - data-dependent checks will be skipped or fail."
    echo "         Stage it with Cluster/scripts/stage_testdata.sh, or set TEST_DATA."
fi

if python -c "import balltree" 2>/dev/null; then
    echo ">>> balltree present; S2 tree parity will run"
else
    echo ">>> balltree absent; attempting install"
    pip install --quiet balltree 2>/dev/null \
        && echo ">>> balltree installed" \
        || echo ">>> balltree unavailable; EXPECT S2 to fail (see header)"
fi

export WEAVER_PY=${WEAVER_PY:-python}
export ORACLE_PY=${ORACLE_PY:-python}
exec ./tests/run_all.sh
