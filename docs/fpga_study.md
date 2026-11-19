# FPGA compression study: handoff notes

Status as of 2026-11-19, branch `feat/fpga-compression-workflow` (pushed; last results commit `76b647e`).
Read this before resuming the study. The human-readable report, with charts, is the shared doc
"ErwinParTLite FPGA study: full local report": https://claude.ai/artifact/CA8bYdBuT8SDGa4s36d5et

## Goal

Compress ErwinParTv2 (ball-tree attention ParT, 2.21M params, 95.7M MACs) into an FPGA-friendly
JetClass tagger, ErwinParTLite (`networks/erwin_part/fpga/`), keeping as much accuracy as possible.
Plan: step 0 audit, 1 hardware refactor, 2 Lite model, 3 width/depth sweep, 4 distillation,
5 quantization-aware training, 6 summary, then STOP for the user's review.

## Ground rules (from the user)

- **Local data first, hard gate:** train only on `JetClass_example_100k.root`. Do not launch anything
  on the 10% JetClass dataset or the cluster until the user has reviewed the local results.
- Commit per step on this branch. **Push only when the user asks.**
- Seed noise is about +-0.005 accuracy; gaps smaller than that are not findings.

## Where things are

| What | Path |
| --- | --- |
| Model (LiteConfig, LiteMSA, PairNet, ...) | `networks/erwin_part/fpga/lite_model.py` |
| Fixed-point emulation (ap_fixed, QAT, BOPs) | `networks/erwin_part/fpga/quant.py` |
| Lookup-table logs, bit-field frexp | `networks/erwin_part/fpga/approx.py` |
| Weaver plugin | `networks/example_ErwinParTLite.py` |
| Checks (37, all pass) | `tests/s15_lite.py` (in `tests/run_all.sh`) |
| Local trainer (KD, QAT, --seq-len) | `tests/s16_lite_train.py` |
| Sweep stages | `tests/s16_sweep.sh {variants,repeat,decide,width,depth,kd,qat,particles}` |
| Plots | `tests/plot_fpga_compression.py` -> `plots/fpga_compression/` (predate QAT/particles) |
| Results | `tests/s10_results/lite_<tag>_{result.json,log.txt}`; `_best.pt` is gitignored, local only |

Run with the weaver env: `/opt/homebrew/Caskroom/miniconda/base/envs/weaver/bin/python`.
Long stages: `caffeinate -i -s bash tests/s16_sweep.sh <stage> > <log> 2>&1` in the background.

## Chosen design

Morton tiles of 8 with half-cell shift in alternate blocks, hierarchy 64->32->16 (summed 4-momenta),
physics ball summaries, LUT pair features, masked BatchNorm, multi-scale class-attention readout,
64 particles. Flags: `--ball-summary physics --pair-features lut --norm batchnorm`.

## Key results (local test set, seed 0 unless noted)

| Model | Params | MACs | Float acc | 8-bit acc |
| --- | --- | --- | --- | --- |
| ErwinParTv2 (baseline) | 2.21M | 95.7M | 0.767-0.770 | - |
| D=64, FFN 2, depths 2,2,2 (`d64_phys_lut_bn_s0`) | 376k | 15.5M | 0.7769 (2-seed mean 0.7746) | 0.7730 |
| D=32, FFN 4, depths 1,1,1 (`dp32f4_111`), FPGA candidate | 93k | 4.2M | 0.7632 | 0.7598 |

- Width: D=128 overfits locally (no better than D=64); D=32 loses ~0.015-0.02, D=16 ~0.04.
- Distillation from `erwin_m64` (alpha 0.5, T 4): +0.001 to +0.006, small because the teacher
  memorized the local sample (0.828 train vs 0.770 test).
- QAT (6 epochs, lr 3e-4, `--dist-cap 16`): 8 bits ~free; 6 bits -0.025..-0.03; uniform 4 bits
  -0.12..-0.19. Attention floor of 8 bits only matters at 4 bits (+0.02..+0.035).
  BOPs: D=64 1.00G/0.56G/0.25G at 8/6/4 bits; D=32 0.27G/0.15G/0.07G.
- Particles (float): 64->32 costs ~0.014, ->16 another ~0.028; Hbb/top rejection fall most.
  Narrowing D is the cheaper cut than dropping particles.

## Lessons that cost time

- **Self-pair bug (fixed, `60fa762`):** the diagonal pair (ln dR = -18.4) dominated PairNet's BatchNorm
  stats. Stats now use distinct pairs; self-pairs get one learned bias. Symptom: LUT beat exact features.
- **Fixed-point calibration:** one-particle tiles have radius 0, so distances ~1e6 wrecked integer-bit
  calibration (8-bit PTQ 0.54). Fixed by `--dist-cap 16` (float changes <= 0.0001), row-max subtraction
  before softmax, fixed score type ap_fixed<., 5>, probabilities ap_ufixed<., 0>.
- `torch.frexp` has no MPS kernel; `approx._frexp` rebuilds it from float bits.
- On battery the Mac sleeps or throttles despite `caffeinate -s`; keep it on mains power.
- Never edit `s16_sweep.sh` while a stage is running: bash reads scripts incrementally.
- A `pgrep -f "<pattern>"` wait loop matches its own command line; use a PID or `pgrep -x`.
- Local test set has only 1,017 QCD jets: rejection above ~1000 shows as "1017+"; treat rej90 as a trend.

## Still to do

Waiting on the user's go-ahead (local):
1. Rerun the pre-fix comparisons at seeds 0 and 1: Morton vs tree (`--partition tree`) and hierarchy
   vs flat pT (`--partition pt --flat --depths 6 --ball-summary plain`).
2. Second seed for the 8- and 6-bit QAT runs of both candidates.
3. Check whether BatchNorm + the self-pair fix help ErwinParTv2 itself.

Then (local): mixed precision (4 bits only where tolerated); quantize inputs and pair features;
mass sculpting and rejection at several efficiencies; refresh plots; update `CLAUDE.md`.

Blocked on the user:
- **Cluster, 10% dataset:** after review. Recommended runs: D=64 and D=32 one-block (float, then 8-bit),
  plus ErwinParTv2 retrained on the same split. Needs a job YAML cloning this branch (copy
  `kube/erwinpartv2-job.yml` or use `Cluster/`).
- **Hardware:** a Linux machine with Vitis is available. Need: SSH access, Vitis/Vivado version, target
  FPGA part, whether hls4ml is installed. Plan: export the 8-bit D=32 model to HLS (custom C++ for tile
  attention and the Morton sort), check bit-exactness against `quant.py`, Vitis HLS synthesis (latency,
  II, DSP/LUT/FF/BRAM), Vivado place and route (timing, power).
