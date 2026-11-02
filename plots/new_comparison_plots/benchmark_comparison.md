# Benchmark comparison: ErwinParTv2 on JetClass

JetClass, 10 classes, `full` feature set. Rejection is background (QCD) rejection
$1/\varepsilon_B$ at the stated signal efficiency, with ParT's discriminant
$s_i / (s_i + s_\mathrm{QCD})$ and ParT's working points.

## Main table

| **Model** | **Accuracy** | **AUC** | **H→bb̄<br>Rej 50%** | **H→cc̄<br>Rej 50%** | **H→gg<br>Rej 50%** | **H→4q<br>Rej 50%** | **H→ℓνqq′<br>Rej 99%** | **t→bqq′<br>Rej 50%** | **t→bℓν<br>Rej 99.5%** | **W→qq′<br>Rej 50%** | **Z→qq̄<br>Rej 50%** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **PFN** | 0.772 | 0.9714 | 2924 | 841 | 75 | 198 | 265 | 797 | 721 | 189 | 159 |
| **P-CNN** | 0.809 | 0.9789 | 4890 | 1276 | 88 | 474 | 947 | 2907 | 2304 | 241 | 204 |
| **ParticleNet** | 0.844 | 0.9849 | 7634 | 2475 | 104 | 954 | 3339 | 10526 | 11173 | 347 | 283 |
| **ParT (plain)** | 0.849 | 0.9859 | 9569 | 2911 | 112 | 1185 | 3868 | 17699 | 12987 | 384 | 311 |
| **ErwinParT** | 0.825 | 0.9819 | 8000 | 1681 | 97 | 590 | 1695 | 5556 | 4545 | 296 | 244 |
| **ParT** | **0.861** | **0.9877** | **10638** | **4149** | **123** | **1864** | **5479** | **32787** | **15873** | **543** | **402** |
| *New models: 10% of JetClass, 20 epochs, 2M test jets* | | | | | | | | | | | |
| **ErwinParTv2** (mean of 2 runs) | 0.8330 | 0.9833 | 6667 | 1983 | 96 | 725 | 2186 | 9129 | 5885 | 335 | 258 |
| **ErwinParTv2 + Lund** (1 run) | 0.8325 | 0.9832 | 7143 | 1835 | 96 | 697 | 1980 | 7407 | 5000 | 345 | 253 |

**The rows above the divider are not directly comparable with the new models.**
The new models trained on 10% of JetClass (about 1M jets per epoch) for 20 epochs,
and were tested on 2M test jets: 10% of the 20M-jet test set. The rows for ParT, ParticleNet,
P-CNN, PFN and ParT (plain) are as published for full JetClass, and the ErwinParT
(v1) row is as previously reported. Part of the gap to ParT comes from the smaller
training budget, not the architecture. The same-budget ParT run is still in progress
(see [Pending](#pending)).

The new models' rejections are geometric means over runs; accuracy and AUC are
arithmetic means.

## Per run

| Run | GPU | Accuracy | AUC | H→bb̄ | H→cc̄ | H→gg | H→4q | H→ℓνqq′ | t→bqq′ | t→bℓν | W→qq′ | Z→qq̄ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ErwinParTv2, run a | RTX 4090 | 0.8328 | 0.9833 | 6667 | 1887 | 96 | 712 | 2151 | 8333 | 5714 | 341 | 258 |
| ErwinParTv2, run b | L40 | 0.8333 | 0.9834 | 6667 | 2083 | 95 | 738 | 2222 | 10000 | 6061 | 328 | 258 |
| ErwinParTv2 + Lund, run a | L40 | 0.8325 | 0.9832 | 7143 | 1835 | 96 | 697 | 1980 | 7407 | 5000 | 345 | 253 |

## Precision of the rejection values

The 2M-jet test set holds 200,000 QCD jets. A rejection of 10,000 means only 20 of
them pass the cut, so the highest rejections carry large counting errors. The
relative error is roughly $1/\sqrt{N_\mathrm{pass}}$:

| | H→bb̄ | H→cc̄ | H→gg | H→4q | H→ℓνqq′ | t→bqq′ | t→bℓν | W→qq′ | Z→qq̄ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| QCD jets passing (ErwinParTv2) | 30 | ~100 | ~2,090 | ~275 | ~92 | 20–24 | 33–35 | ~600 | 775 |
| Relative error | ±18% | ±10% | ±2% | ±6% | ±10% | ±20–22% | ±17% | ±4% | ±4% |

So the differences between ErwinParTv2 and the Lund variant in H→bb̄, t→bqq′
and t→bℓν are within counting noise. The full 20M-jet test set would cut these
errors by about 3×.

## Cost

Forward FLOPs per jet, counted with `torch.utils.flop_counter.FlopCounterMode` on
512 real jets, with each model built by its weaver network config:

| Model | Parameters | GFLOPs / jet | vs ParT | Particles kept | Test jets truncated |
| --- | --- | --- | --- | --- | --- |
| ParT | 2.14M | 0.670 | 1× | 128 | 0.001% |
| ErwinParT (v1) | 2.65M | 0.286 | 2.3× less | 128 | 0.001% |
| **ErwinParTv2** | 2.21M | **0.191** | **3.5× less** | 64 | 6.3% |
| ErwinParTv2 + Lund | 2.26M | 0.254 | 2.6× less | 64 + Lund tokens | 6.3% |

The Lund clustering's elementwise operations add under 0.003 GFLOPs per jet; the
FLOP counter doesn't see them. At 64 particles, truncation falls hardest on
H→gg (24.9% of jets) and t→bqq′ (13.8%), against about 1% for W→qq′ and Z→qq̄.

## Takeaways

- **ErwinParTv2 improves on ErwinParT v1** in accuracy (0.833 against 0.825)
  and AUC (0.9833 against 0.9819), at two-thirds of v1's cost per jet and 3.5×
  less than ParT. Its 20 epochs are likely fewer than v1's: v1's checkpoints
  run to epoch 49, though which run produced the reported row isn't recorded.
- **Lund tokens add nothing at this scale.** The single Lund run lands inside the
  spread of the two ErwinParTv2 runs on accuracy and AUC, and is slightly behind
  on overall rejection, while costing 33% more FLOPs.
- Run-to-run spread for ErwinParTv2 is small: 0.0005 in accuracy, 0.0001 in AUC.

## Pending

These runs are queued on the NRP Nautilus cluster with the same 10%, 20-epoch setup:

- **ParT**: the same-budget reference. It will show how much of the gap to
  published ParT comes from the architecture rather than the training budget.
- **ErwinParTv2 + Lund, run b**: the second Lund run.
- **ErwinParTv2 with first-level ball sizes 16, 32 and 64**: whether ball locality
  costs accuracy. Size 64 makes the first level dense.

## Sources

- Published rows and the ErwinParT (v1) row: the table provided, also in
  `Plots/ErwinParT.py`.
- New models: `tests/s10_results/cluster_eval.json`, computed from the
  runs' test predictions by `Cluster/scripts/eval_runs.py` (branch `ErwinParTCluster`).
- FLOPs: `tests/s10_results/cluster_flops.json`, from `tests/flops_cluster_models.py`.
- Truncation: `tests/s10_results/test_multiplicity.json`, read from the 2M test jets.
- Plots: this folder; see `summary.csv` for the same numbers in machine-readable form.
