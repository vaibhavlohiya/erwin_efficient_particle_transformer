# Runbook

## Scope

**Default: 10% of JetClass.** Set by `DATA_FRACTION=0.1` plus a matched
`SAMPLES_PER_EPOCH`. These are two different things and both matter:

| Knob | What it actually does |
|---|---|
| `--data-fraction 0.1` | loads a contiguous 10% window of events **from every file**, so class balance and file diversity survive. For train/val the offset is **redrawn each epoch**; for test (`for_training=False`) it is **deterministic** — the first 10% of each file |
| `--samples-per-epoch` | independent of the above. Shrinking the data pool alone does **not** shrink compute |
| `--file-fraction` | avoid. Applied after `np.random.shuffle(filelist)`, so it is a random 10% of *files*, redrawn per epoch; weaver itself warns against it |

Consequence worth knowing: `--data-fraction` caps *per-epoch* exposure, not
cumulative. Over 50 epochs the redrawn window sweeps most of each file. If you
need a hard ceiling on data ever touched, either cap at ~10 epochs or
materialise a frozen subset on the PVC — neither is the current default.

Because the test slice is deterministic, **every model scores the identical
jets**, which is what makes the comparison fair.

At 10% and `SAMPLES_PER_EPOCH=1024000`: 2,000 steps/epoch at batch 512,
estimated 6–17 min/epoch, ~5–14 h for 50 epochs. Measure it on the smoke run
rather than trusting that estimate.

## Submitting

```bash
./Cluster/submit.sh dry-run train MODEL=ParT TAG=x   # rehearse
./Cluster/submit.sh train MODEL=ErwinParTv2 TAG=v2-10pct
```

`submit.sh` refuses when the tree is dirty, the branch is wrong, or `HEAD` is
ahead of `origin`. That guard exists because the job clones from GitHub: without
it, the easiest possible mistake is spending a day testing code you did not
change.

## Monitoring

```bash
kubectl -n cms-ml get jobs -l jobgroup=erwinpart-cluster
kubectl -n cms-ml logs -f job/vleept-<run-id>
kubectl -n cms-ml describe job vleept-<run-id> | tail -30
```

The training log and tensorboard events are also on the PVC at
`/output/runs/<run-id>/` (`train.log`, `tensorboard/`) — they survive the pod,
which they did not before.

## Resuming

`run_train.sh` finds the highest `net_epoch-N_state.pt` in the run directory and
passes `--load-epoch N`. To resume a preempted run, **submit with the same
`RUN_ID`**; it picks up where it stopped. This only works because the run
directory is stable — `{auto}` would have produced a new one.

## Failure modes

| Symptom | Cause | Fix |
|---|---|---|
| `Invalid model <name>!` | the model has no branch in `train_JetClass.sh`/`test_JetClass.sh` | add one; note `Single`/`SinglePairs` point at files that do not exist |
| `ModuleNotFoundError: particle_transformer` | pre-Stage-A checkout | the plugins now fall back to a path-anchored import; check you are on `ErwinParTCluster` |
| preflight: `weaver-core not importable` | the image does not ship weaver-core | `requirements.txt` installs it; check the pip step succeeded |
| preflight: `non-finite logits/gradients` | numerics, most likely the fp16 masked-key path | do not run the long job; reproduce locally with `tests/s6_leakage.py`, `tests/s7_gradients.py` |
| Pod `Pending` forever | no node satisfies GPU affinity + both PVC mounts | widen `nvidia.com/gpu.product`, or check the probe pod scheduled at all |
| `predict` finds no checkpoint | wrong `RUN_ID` | `ls /output/runs/` from the jupyter or probe pod |
| Job vanished mid-run | preemption | resubmit with the same `RUN_ID`; it resumes |
| Two runs overwrote each other | same `RUN_ID` | `RUN_ID` embeds date+SHA+tag; change `TAG` |

## After a run

```bash
bash Cluster/scripts/fetch_results.sh vleept-probe <run-id>   # copies only
```

Then append a row to `docs/RUNS.md`. `run_manifest.json` in the run directory
already records SHA, branch, image, scope and library versions.
