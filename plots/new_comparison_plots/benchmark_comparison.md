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
| **ParT, same budget** (1 run) | **0.8392** | **0.9843** | 7143 | 2174 | 106 | 909 | 2857 | 11111 | 7692 | 362 | 266 |

**The rows above the divider are not directly comparable with the new models.**
The new models trained on 10% of JetClass (about 1M jets per epoch) for 20 epochs,
and were tested on 2M test jets: 10% of the 20M-jet test set. The rows for ParT, ParticleNet,
P-CNN, PFN and ParT (plain) are as published for full JetClass, and the ErwinParT
(v1) row is as previously reported. Part of the gap to ParT comes from the smaller
training budget, not the architecture. The **ParT, same budget** row is the
like-for-like comparison: ParT trained exactly as the new models were.

The new models' rejections are geometric means over runs; accuracy and AUC are
arithmetic means.

## Per run

| Run | GPU | Accuracy | AUC | H→bb̄ | H→cc̄ | H→gg | H→4q | H→ℓνqq′ | t→bqq′ | t→bℓν | W→qq′ | Z→qq̄ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ErwinParTv2, run a | RTX 4090 | 0.8328 | 0.9833 | 6667 | 1887 | 96 | 712 | 2151 | 8333 | 5714 | 341 | 258 |
| ErwinParTv2, run b | L40 | 0.8333 | 0.9834 | 6667 | 2083 | 95 | 738 | 2222 | 10000 | 6061 | 328 | 258 |
| ErwinParTv2 + Lund, run a | mixed (last part L40) | 0.8325 | 0.9832 | 7143 | 1835 | 96 | 697 | 1980 | 7407 | 5000 | 345 | 253 |
| ParT, same budget | mixed (last part RTX 3090) | 0.8392 | 0.9843 | 7143 | 2174 | 106 | 909 | 2857 | 11111 | 7692 | 362 | 266 |

## ErwinParTv2 relative to ParT (same budget)

Background rejection of ErwinParTv2 (mean of 2 runs) divided by ParT's:

| H→bb̄ | H→cc̄ | H→gg | H→4q | H→ℓνqq′ | t→bqq′ | t→bℓν | W→qq′ | Z→qq̄ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.93 | 0.91 | **0.90** | **0.80** | 0.77 | 0.82 | 0.77 | 0.92 | 0.97 |

Overall, ErwinParTv2 keeps **99.3% of ParT's accuracy** (0.8330 against 0.8392,
−0.61 points) and **99.9% of its AUC** (−0.0010).

Error bars include both models' counting errors, from the QCD jets passing each
cut (`plots/fpga_tradeoff/rejection_ratio.png`). Two losses are well measured:
H→4q (0.80 ± 0.06) and H→gg (0.90 ± 0.03). W→qq′ and Z→qq̄ sit at 0.92 ± 0.05
and 0.97 ± 0.04. The lepton classes hint at a larger loss (H→ℓνqq′ 0.77 ± 0.11,
t→bℓν 0.77 ± 0.18), but only within about 2σ. H→bb̄, H→cc̄ and t→bqq′ can't
be resolved with this test set (±0.11 to ±0.23).

There's no clear link to truncation. H→gg, the most truncated class (24.9%),
does lose about 10%, but H→4q (8.1%) loses twice as much, and barely truncated
classes vary from 0.77 to 0.97. Telling ball locality apart from truncation
needs the ball-size sweep (see [Pending](#pending)) and, for the classes with few
QCD jets passing, the full 20M-jet test set.

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

Forward FLOPs per jet, counted with `torch.utils.flop_counter.FlopCounterMode`,
with each model built by its weaver network config:

- **Deployed:** padding trimming off, so every jet uses all 128 input slots.
  This is how weaver builds a model for inference or ONNX export
  (`enabled=trim and not for_inference`), and how a fixed-size hardware pipeline
  sized for the largest jet runs.
- **Training:** weaver trims each batch's padding down to its longest jet. This
  was measured on 512 real jets, the longest with 104 particles.

| Model | Parameters | GFLOPs / jet, deployed | vs ParT | GFLOPs / jet, training batch | vs ParT | Particles kept | Test jets truncated |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ParT | 2.14M | 0.670 | 1× | 0.512 | 1× | 128 | 0.001% |
| ErwinParT (v1) | 2.65M | 0.286 | 2.3× less | 0.276 | 1.9× less | 128 | 0.001% |
| **ErwinParTv2** | 2.21M | **0.191** | **3.5× less** | **0.185** | **2.8× less** | 64 | 6.3% |
| ErwinParTv2 + Lund | 2.26M | 0.254 | 2.6× less | 0.248 | 2.1× less | 64 + Lund tokens | 6.3% |

ErwinParTv2 always fills 64 fixed slots, so its cost barely depends on trimming.
ParT's cost follows the longest jet it sees. With per-jet trimming at batch 1,
ParT is the cheaper model for jets below about 40 particles; see
`plots/fpga_tradeoff/scaling.png`.

The Lund clustering's elementwise operations add under 0.003 GFLOPs per jet; the
FLOP counter doesn't see them. At 64 particles, truncation falls hardest on
H→gg (24.9% of jets) and t→bqq′ (13.8%), against about 1% for W→qq′ and Z→qq̄.

## Takeaways

- **Against ParT trained the same way, ErwinParTv2 gives up 0.6 points of
  accuracy and 0.001 of AUC** for 3.5× fewer FLOPs and 7.7× less peak activation
  memory when deployed (`plots/fpga_tradeoff/`).
- Of the 2.8-point gap to published ParT (0.861), 2.2 points come from the
  smaller training budget (ParT itself reaches 0.839 here) and 0.6 from the
  architecture.

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
