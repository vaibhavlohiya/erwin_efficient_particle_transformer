# Repository inventory

Documentation, not a cleanup list. **Nothing here is scheduled for deletion.**
Reorganisation in this repo is additive by policy: files stay where they are and
this page records what each one is, so "is this current?" has an answer that is
not archaeology.

## Two generations of the model — the easy mistake

| | `ErwinParT` (v1) | `ErwinParTv2` |
|---|---|---|
| Code | `networks/EfficientParticleTransformer.py`, `attn_type: 'erwin'` | `networks/erwin_part/` + `networks/example_ErwinParTv2.py` |
| Partition | 1-D ΔR ordering cut into fixed chunks | 2-D recursive median-split ball tree |
| ParT pair bias | dropped | kept, within-ball pairs only |
| Hierarchy | none | constituents → subjets → subjets |
| Batch size | 16 (per-jet Python loop) | 512 (vectorised) |

v2 supersedes v1. v1 stays as the baseline it is.

## Current / load-bearing

| Path | Status |
|---|---|
| `networks/erwin_part/` | **current.** The v2 implementation |
| `networks/example_ErwinParTv2.py` | **current.** weaver plugin for v2 |
| `tests/s0..s9_*.py`, `tests/run_all.sh` | **current.** 117 checks, full suite passes |
| `tests/s10_compare.py`, `tests/s11_*.sh` | **current.** Local small-scale comparisons |
| `tests/s10_results/*.json` | **current.** What `make_plots.py` reads |
| `Cluster/` | **current.** Cluster entry point |
| `train_JetClass.sh`, `test_JetClass.sh` | **current.** Launchers |

## Superseded but kept

| Path | What it is |
|---|---|
| `kube/*.yml` | the pre-`Cluster/` job manifests. Still valid; they clone `ErwinParT` and reproduce the older behaviour. Superseded by `Cluster/jobs/job.yml` |
| `kube/part-vol.yml` | declares `part-vol-3` (3000Gi), belonging to the older `../weaver/`-fork workflow, not this repo |
| `test_erwin_block.py` | v1 smoke test. Imports `particle_transformer.*` and assumes the checkout is named that, so it does not run from this directory |
| `test_cls_integration.py` | v1 smoke test, self-contained, still runs |
| `benchmark_erwin_complexity.py`, `visualize_erwin_balls.py` | v1 analysis scripts → `erwin_complexity.png`, `erwin_balls.png` |
| `smoke_test_erwin.ipynb` | v1 notebook against the local sample |

## Where figures live — four homes, on purpose

| Directory | Produced by | Subject |
|---|---|---|
| `figures/` | upstream ParT paper | `arch.png`, `dataset.png`, `jet-tagging.png` |
| `plots/` | `tests/make_plots.py` | v2 results; regenerable |
| `results/` | `tests/make_pdf.py`, `merge_pdf.py` | deck assets + `SPEAKER_NOTES.md` |
| repo root | v1 scripts above | `erwin_balls.png`, `erwin_complexity.png`, `erwin_part_v2_complexity.png`, `loss_accuracy_Erwin.png` |

Only `plots/` is regenerated from committed JSON. The rest are artefacts of the
run that made them.

## Loose ends (recorded, not actioned)

- `results/EXO-23-012-paper-v10 copy.pdf` — an unrelated CMS paper, committed.
- `New_test.ipynb` — zero bytes.
- `notebooks/JetClass101.ipynb` — third copy (also at the workspace root and in `Jet_Class/`).
- `tests/*.out` — captured stdout, committed. `.gitignore` excludes `tests/s10_results/*.out` but not these.
- `train_JetClass.sh` / `test_JetClass.sh` `Single` and `SinglePairs` branches point at `networks/example_Single*.py`, which do not exist.
- `networks/erwin_part/pair_features.py` cites `tests/calibrate_pair_norm.py`, which is not present. The `PAIR_FEATURE_NORM` constants are frozen and geometry-specific: **changing `seq_len`, `ball_sizes` or `strides` silently invalidates them** and there is currently no tool to re-measure.
- `from_part_pair_embed` contains `offset = 0 if (…) else 0` — a no-op, unused.

## Known gap in the science

There is **no ParT accuracy baseline** anywhere: `tests/s10_results/` holds 18
`erwin_*_result.json` and zero `part_*`. `results/SPEAKER_NOTES.md` says so
outright — ParT could not be trained on the available hardware. Every accuracy
number to date is ErwinParTv2 against itself or its own dense-attention control.
Closing that gap is what `Cluster/` exists for.
