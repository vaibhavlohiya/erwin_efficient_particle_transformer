# kube/ — the original job manifests

These still work and are **not** deprecated in the sense of broken. They clone
the `ErwinParT` branch and reproduce the behaviour they always had.

For new work use **[`../Cluster/`](../Cluster/)**, which clones
`ErwinParTCluster` and adds what these manifests lack:

| | `kube/*.yml` | `Cluster/` |
|---|---|---|
| Output dir | `{auto}` — a new timestamp each start | stable `RUN_ID` |
| Prediction | cannot locate a trained checkpoint (`{auto}` re-expands; path is `/output/training/…` while training writes `/output/…`) | explicit `RUN_DIR` |
| Resume after preemption | no | `--load-epoch`, automatic |
| Training log / tensorboard | container-ephemeral, lost with the pod | on the output PVC |
| Preflight | constructs the model | constructs **and** runs fwd/bwd on sparse jets |
| Scope | full JetClass | 10% by default |
| Boilerplate | duplicated per model | one template, `jobs/job.yml` |

**Do not mix the two in one experiment** — they clone different branches, so
they can be running different code.
