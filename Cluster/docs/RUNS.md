# Run ledger

Append-only. One row per cluster run. `run_manifest.json` in each run directory
holds the machine-readable version (SHA, branch, image, scope, torch/weaver
versions); this file is the human index.

Scope is 10% of JetClass unless a row says otherwise.

| Run ID | Date | Role | Model | SHA | Scope | Outcome | Notes |
|---|---|---|---|---|---|---|---|
| 20261029-35e96f7-v2-10pct-a | 2026-10-29 | train | ErwinParTv2 | 35e96f7 | 0.1 / 1,024,000 / 20 | completed | RTX 4090. Test acc 0.8328, AUC 0.9833 |
| 20261029-35e96f7-v2-10pct-b | 2026-10-29 | train | ErwinParTv2 | 35e96f7 | 0.1 / 1,024,000 / 20 | completed | L40. Test acc 0.8333, AUC 0.9834 |
| 20261029-35e96f7-lund-10pct-a | 2026-10-29 | train | ErwinParTv2Lund | 35e96f7 | 0.1 / 1,024,000 / 20 | preempted+resumed (4 attempts) | Mixed GPUs, last part L40. Test acc 0.8325, AUC 0.9832. Its timing record has 30 epoch entries, not 20: check `run_manifest.json` |
| 20261029-35e96f7-lund-10pct-b | 2026-10-29 | train | ErwinParTv2Lund | 35e96f7 | 0.1 / 1,024,000 / 20 | preempted+resumed (3 attempts) | Mixed GPUs, last part L40. Test acc 0.8338, AUC 0.9834 |
| 20261029-35e96f7-part-10pct | 2026-10-29 | train | ParT | 35e96f7 | 0.1 / 1,024,000 / 20 | preempted+resumed (5 attempts) | Mixed GPUs, last part RTX 3090. Same-budget ParT baseline. Test acc 0.8392, AUC 0.9843 |
| 20261102-04fe81e-m16-10pct | 2026-11-02 | train | ErwinParTv2, m = 16 | 04fe81e | 0.1 / 1,024,000 / 20 | completed | RTX 4090. Ball-size sweep. Test acc 0.8364, AUC 0.9838 |
| 20261102-04fe81e-m32-10pct | 2026-11-02 | train | ErwinParTv2, m = 32 | 04fe81e | 0.1 / 1,024,000 / 20 | completed | RTX 3090. Ball-size sweep. Test acc 0.8381, AUC 0.9841 |
| 20261102-04fe81e-m64-10pct | 2026-11-02 | train | ErwinParTv2, m = 64 | 04fe81e | 0.1 / 1,024,000 / 20 | completed | RTX 3090. Ball-size sweep (dense first level). Test acc 0.8380, AUC 0.9841 |

Filled in retrospectively on 2026-11-19 from `tests/s10_results/cluster_eval.json` and
`plots/new_comparison_plots/benchmark_comparison.md` on branch `ErwinParT`. Test metrics are on
2M test jets (10% of the test set), computed by `Cluster/scripts/eval_runs.py`; attempts count pod
restarts. Smoke and test-suite jobs were not recorded.

## Columns

- **Run ID** — `<yyyymmdd>-<short-sha>-<tag>`; also the job name (`vleept-<run-id>`)
  and the output directory (`/output/runs/<run-id>/`).
- **Scope** — `data-fraction` / `samples-per-epoch` / `epochs`.
- **Outcome** — completed | preempted+resumed | failed(reason). Record failures;
  they are the rows that stop the same mistake twice.
