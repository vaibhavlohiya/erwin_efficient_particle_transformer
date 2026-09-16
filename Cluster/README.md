# Cluster/ — running ErwinParTv2 on NRP Nautilus

Everything needed to train, evaluate and verify the ball-tree Particle
Transformer on the `cms-ml` namespace, in one place.

**Default scope is 10% of JetClass.** Every job here sets `--data-fraction 0.1`
with a matching `--samples-per-epoch`. Raising it is an explicit decision — see
[docs/RUNBOOK.md](docs/RUNBOOK.md#scope).

## The one rule

> A job runs the branch **as it exists on GitHub**. Local edits that are not
> committed *and pushed* are not in the run.

`submit.sh` refuses to submit when the tree is dirty, when you are on a
different branch than the job clones, or when `HEAD != origin/<branch>`.

## Quick start

```bash
# 0. verify the cluster matches our assumptions (versions, /data layout, PVCs)
kubectl -n cms-ml apply -f Cluster/pods/probe.yml
kubectl -n cms-ml logs -f vleept-probe

# 1. stage the test sample once (136 MB; .gitignore excludes *.root)
bash Cluster/scripts/stage_testdata.sh vleept-probe

# 2. smoke: preflight + forward/backward + one 0.1% epoch  (~15 min)
./Cluster/submit.sh smoke MODEL=ErwinParTv2

# 3. the S0-S9 suite on cluster hardware
./Cluster/submit.sh tests

# 4. the deliverable: matched comparison at 10%
./Cluster/submit.sh train MODEL=ParT        TAG=part-10pct
./Cluster/submit.sh train MODEL=ErwinParTv2 TAG=v2-10pct

# 5. predict from a finished run (id is in docs/RUNS.md)
./Cluster/submit.sh predict MODEL=ErwinParTv2 RUN_ID=<run-id>

# anything can be rehearsed first
./Cluster/submit.sh dry-run train MODEL=ParT TAG=x
```

`Makefile` offers the same targets (`make -C Cluster train …`), but macOS gates
`/usr/bin/make` behind the Xcode licence, so `submit.sh` is the portable one.

## Layout

| Path | What it is |
|---|---|
| `submit.sh` | renders `jobs/job.yml` and submits it; the push guard lives here |
| `Makefile` | same targets via `make`, where `make` is available |
| `jobs/job.yml` | **the single job template.** Every role — train, predict, smoke, tests — is this file with different env. The affinity/toleration/PVC boilerplate exists once |
| `scripts/entrypoint.sh` | container entrypoint: pip, `/output` symlink, preflight, dispatch |
| `scripts/preflight.py` | versions + plugin contract + a real fwd/bwd on sparse jets |
| `scripts/run_train.sh` | stable `RUN_ID` dir, 10% scope, auto-resume, provenance |
| `scripts/run_predict.sh` | predict from an explicit `RUN_DIR` |
| `scripts/run_tests.sh` | S0–S9 with cluster interpreters and staged data |
| `scripts/stage_testdata.sh` | one-time copy of the 100k sample to the PVC |
| `scripts/fetch_results.sh` | pull a finished run back to `outputs/cluster/` (copy only) |
| `pods/probe.yml` | the assumption-checker; also serves `kubectl cp` |
| `pods/jupyter.yml` | Jupyter on the output PVC (copy of `../../pods/…`) |
| `pvc/vl-eept.yml` | reference copy of the output PVC (already bound) |
| `docs/RUNBOOK.md` | submit, monitor, diagnose, resume |
| `docs/RUNS.md` | append-only ledger: run id, SHA, image, scope, outcome |
| `docs/INVENTORY.md` | what every stray file in this repo is |

## Storage

| PVC | Mount | Size | Class | Rule |
|---|---|---|---|---|
| `j-jepa-vol` | `/data` | 40000Gi | `rook-cephfs-central` | **read-only, always.** Owned elsewhere; holds JetClass |
| `vl-eept-vol` | `/output` | 500Gi | `rook-cephfs` | ours: `runs/<run-id>/`, `testdata/` |

Both verified `Bound`. They are on *different* storage classes, so "can one node
mount both" is a real question — `pods/probe.yml` settles it.

## Relationship to `kube/`

`kube/*.yml` is untouched and still works; those jobs clone `ErwinParT` and
reproduce the older behaviour exactly. `Cluster/` clones `ErwinParTCluster`.
Do not mix them in one experiment. See `kube/README.md`.
