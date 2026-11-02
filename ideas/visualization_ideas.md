# Visualization ideas: efficiency vs accuracy against ParT

**Goal.** ParT is accurate but too expensive. The new models (ErwinParTv2 and its
Lund variant) aim for efficiency, ultimately to run on FPGAs, and we accept a
slight accuracy drop for far lower resource use and latency. These plots are
meant to show that trade-off: large efficiency gains against small accuracy
differences.

Each idea notes whether the data exists yet or needs new measurements.

## A caveat that affects every plot

A fair trade-off plot needs **ParT trained under the same conditions** as our
models. So far we only have ParT's published full-JetClass numbers (0.861
accuracy), against our 10%-of-JetClass, 20-epoch runs (0.833). That gap mixes
architecture with training budget, which exaggerates our accuracy loss. The
same-budget ParT run (`part-10pct`) is still queued, so the trade-off plots should
use it, with published numbers shown only as a labelled reference.

## Core trade-off plots

### 1. Accuracy vs compute, with the efficiency frontier (the headline plot)
- **x:** GFLOPs per jet (log scale). **y:** accuracy or AUC; background rejection
  as a variant.
- **Points:** ParT, ErwinParTv2, + Lund, ErwinParT v1. Optionally PFN, P-CNN and
  ParticleNet: their network files are in the repo, so their FLOPs can be counted
  the same way.
- A line joins the models no other model beats on both axes (the frontier).
  Message: "ParT-class accuracy at a fraction of the cost".
- **Data:** mostly in hand. Needs the same-budget ParT, and FLOP counts for the
  three published models.

### 2. Relative trade-off: % of ParT's accuracy against % of ParT's cost
- Both axes normalised so ParT = 100%. A shaded **target zone** marks a chosen
  budget, for example at least 98% of ParT's accuracy at no more than 30% of its
  cost.
- Turns the result into one sentence for slides: "99% of ParT's AUC for 28% of
  the compute".
- One panel per cost measure (FLOPs, parameters, memory, latency), not one
  crowded chart.
- **Data:** in hand except the same-budget ParT, plus memory and latency (5, 7).

### 3. Our own trade-off curve: the knobs we can turn
- The same accuracy-vs-cost axes, tracing ErwinParTv2 across its settings:
  - ball size m = 8, 16, 32, 64 (the cluster sweep, queued);
  - sequence length L = 32, 48, 64, 128.
- Shows efficiency is tunable: an FPGA design can pick an operating point to fit
  its resource budget, instead of one fixed model.
- **Data:** the ball-size sweep is queued. The L-sweep exists only locally; the
  cluster would need new runs.

## Where the efficiency comes from

### 4. Cost vs number of particles (scaling)
- Log-log: FLOPs or memory per jet against particles per jet. ParT grows as N²,
  ours roughly as N·m.
- The real JetClass multiplicity distribution sits shaded behind it, showing
  where jets actually fall and how the gap widens for denser jets or future
  detectors.
- **Data:** mostly exists (`tests/flops_compare.py`); recompute with the
  production models.

### 5. Memory footprint, broken down by component
- Stacked bars of peak activation memory per jet for ParT and v2: attention
  matrices, pair features, feed-forward layers, embeddings.
- Matters most for FPGAs, where on-chip memory (BRAM/URAM) is usually the hard
  limit. ParT's full N×N pair-feature tensor is the expensive part, and it's what
  ball attention removes.
- **Data:** new but cheap: tensor sizes from the architecture, or profiled.

### 6. Where the FLOPs go: ParT → ErwinParTv2 step by step
- A step chart: start from ParT's total, then show how much each design change
  removes (ball attention instead of dense attention, pair features only within
  balls, pooling into subjets, 64 slots instead of 128), ending at v2's total.
- Explains *why* we're cheaper, not just that we are.
- **Data:** partly exists (a FLOPs breakdown by component); recompute per step.

## Latency

### 7. Latency per jet against multiplicity, plus its distribution
- A scatter of single-jet (batch 1) latency against particles per jet, with a
  histogram of latencies alongside.
- Key FPGA/trigger point: our model always works on 64 fixed slots, so its
  latency is **constant and predictable**, while ParT's grows with the jet.
  Deterministic, bounded latency is a hard requirement for a trigger, so this may
  be a stronger argument than average speed.
- **Data:** new measurements at batch 1, on CPU or GPU. Quick.

### 8. Throughput within a latency budget
- Jets per second for each model at a fixed latency budget (for example 1 ms or
  10 µs per jet), or throughput against batch size.
- **Data:** new measurements.

## Where the accuracy loss sits

### 9. Rejection relative to ParT, per jet type
- Bars of (our rejection ÷ ParT's rejection) per class, on a log axis centred at 1,
  with the statistical error bars from `benchmark_comparison.md`.
- Paired with each class's share of jets lost to the 64-particle cap. The loss
  should concentrate in H→gg (25% truncated) and t→bqq′ (14%). If it does, the
  drop is a known, fixable cost of L = 64 rather than of ball attention.
- **Data:** needs the same-budget ParT; everything else is in hand.

### 10. ROC curves with a ratio panel underneath
- The standard HEP layout: rejection vs signal efficiency on top, the ratio
  ours/ParT below, per class. Shows at which working points the loss appears, for
  example only at high signal efficiency.
- **Data:** needs the same-budget ParT.

## FPGA-specific (new work)

### 11. Estimated FPGA resources
- DSPs, LUTs, BRAM, flip-flops and latency in clock cycles for each model, each as
  a share of a target device (for example a Xilinx VU13P), one small panel per
  resource. ParT may simply not fit, which is a strong result in itself.
- Two levels of effort:
  - **Analytic estimates** from multiply-accumulate counts and buffer sizes: days,
    approximate.
  - **hls4ml synthesis**: real numbers, but weeks of work, and attention layers
    have limited hls4ml support.

### 12. Accuracy vs numeric precision
- Accuracy against fixed-point bit width (16, 12, 10, 8, 6 bits) for ParT and v2.
  FPGAs run in low-precision fixed point, so a model that holds its accuracy at 8
  bits is worth more on hardware than its 32-bit accuracy suggests.
- **Data:** new experiments. Post-training quantization is quick;
  quantization-aware training is more work.

## A summary for slides

### 13. Scorecard table
- One row per measure (accuracy, AUC, H→bb̄ rejection, GFLOPs, parameters, peak
  memory, latency, latency spread); columns ParT, ErwinParTv2 and "ratio". Green
  when we win, neutral when within noise.
- **Data:** grows as the measurements above come in.

## Recommended order

1. **With data we have or will have soon:** #1 frontier (headline), #2 relative to
   ParT, #9 per-class rejection ratio with truncation, #13 scorecard. These all use
   the same-budget ParT run, so build them once it finishes.
2. **Cheap new measurements:** #7 latency and its determinism, #5 memory footprint,
   #4 scaling with particle count. Together they make the FPGA case directly:
   predictable latency, smaller memory, and a gap that widens with denser jets.
3. **A separate phase:** #11 FPGA resources and #12 quantization. They need real
   engineering effort, and they're what an FPGA reviewer will eventually ask for.

## Open questions

- Is there a **target FPGA device** or a **latency budget**? It would let the plots
  show a concrete pass line instead of a generic one.

## Design notes for all plots

- Cost axes on a log scale; ParT as the reference line or 100% mark.
- No dual axes: one measure per panel, small multiples for several.
- Accuracy axes zoomed but clearly labelled; bars only for quantities that start
  at zero (FLOPs, memory, latency).
- Published full-JetClass numbers are always labelled as a different training
  regime.
