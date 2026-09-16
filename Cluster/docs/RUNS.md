# Run ledger

Append-only. One row per cluster run. `run_manifest.json` in each run directory
holds the machine-readable version (SHA, branch, image, scope, torch/weaver
versions); this file is the human index.

Scope is 10% of JetClass unless a row says otherwise.

| Run ID | Date | Role | Model | SHA | Scope | Outcome | Notes |
|---|---|---|---|---|---|---|---|
| _(none yet)_ | | | | | | | Stage C has not been run |

## Columns

- **Run ID** — `<yyyymmdd>-<short-sha>-<tag>`; also the job name (`vleept-<run-id>`)
  and the output directory (`/output/runs/<run-id>/`).
- **Scope** — `data-fraction` / `samples-per-epoch` / `epochs`.
- **Outcome** — completed | preempted+resumed | failed(reason). Record failures;
  they are the rows that stop the same mistake twice.
