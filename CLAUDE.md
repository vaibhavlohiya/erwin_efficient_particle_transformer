# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A fork of the official [Particle Transformer (ParT)](https://arxiv.org/abs/2202.03772) jet-tagging repo, extended with *efficient* attention variants (Linformer, Reformer, Mamba, Erwin/ball attention, pair-attention ablations) to reduce the O(N²) cost of ParT's pairwise-interaction attention. Models train on JetClass / QuarkGluon / TopLandscape.

Training is not run by code here — it delegates to the **weaver** framework (`pip install 'weaver-core>=0.4'`), which provides the `weaver` CLI, ROOT data loading/transformation, and the training loop. This repo supplies network definitions, dataset configs, and launcher scripts.

The active work is **bringing Erwin's ball-tree attention into ParT**. Two generations coexist and confusing them is the easy mistake:

| | `ErwinParT` (v1) | `ErwinParTv2` |
|---|---|---|
| Code | `networks/EfficientParticleTransformer.py`, `attn_type: 'erwin'` | package `networks/erwin_part/` + `networks/example_ErwinParTv2.py` |
| Partition | 1-D ΔR ordering cut into fixed chunks | real 2-D recursive median-split ball tree |
| ParT pair bias | dropped | kept, evaluated on within-ball pairs only |
| Hierarchy | none | constituents → subjets → subjets (U-Net style) |
| Batching | per-jet Python loop → `--batch-size 16` | fully vectorised → ParT's own `--batch-size 512` |

v2 supersedes v1. Leave v1 in place as the baseline it is; new work goes in `networks/erwin_part/`.

## Common commands

```bash
# Download a dataset (updates env.sh with the path)
./get_datasets.py [JetClass|QuarkGluon|TopLandscape] [-d DATA_DIR]

# Train on JetClass: arg1 = model name, arg2 = feature set (kin|kinpid|full)
./train_JetClass.sh ErwinParTv2 full
./train_JetClass.sh ParT full

# Extra args forward to weaver and override the script's defaults;
# `-o key value` overrides a get_model() config key
./train_JetClass.sh ParT full --batch-size 256 --gpus 0,1 --num-workers 1
./train_JetClass.sh ErwinParTv2 full -o tree_on_cpu False -o readout fine

# DistributedDataParallel multi-GPU (batch size is per-GPU)
DDP_NGPUS=4 ./train_JetClass.sh ParT full --batch-size 256

# Evaluate / predict (writes pred.root; expects /output paths, used in kube jobs)
./test_JetClass.sh ParT full

# Other datasets (support -FineTune variants starting from models/*.pt)
./train_QuarkGluon.sh [ParT|ParT-FineTune|PN|PN-FineTune|PFN|PCNN] [kin|kinpid|kinpidplus]
./train_TopLandscape.sh [ParT|ParT-FineTune|PN|PN-FineTune|PFN|PCNN] [kin]
```

Model names accepted by `train_JetClass.sh`: `ParT`, `ParTNoPairs`, `LinformerParT`, `LinformerPairWise`, `ErwinParT`, `ErwinParTv2`, `ReformerParT`, `MambaParT`, `PairAttnParT`, `MorePairAttnParT`, `PN`, `PFN`, `PCNN`. Each maps to a `networks/example_*.py` plus a tuned batch size / LR in an `elif` branch — adding a model means adding both. (The `Single` / `SinglePairs` branches point at files that don't exist; they're dead.)

Per-model batch sizes are tuned to memory cost, not uniform: most use 512, `PairAttnParT` 256, `MorePairAttnParT` 128, `PFN`/`PCNN` 4096, and **`ErwinParT` (v1) 16** — expect v1 to be much slower per epoch.

`env.sh` exports `DATADIR_JetClass` etc. and is sourced by the train scripts. `COMMENT=<suffix>` tags the output dir/tensorboard run. Checkpoints go to `training/JetClass/Pythia/<feature>/<model>/{auto}<suffix>/`.

## Interpreters — three, deliberately not interchangeable

| Interpreter | Has | Used for |
|---|---|---|
| conda env `weaver` (`/opt/homebrew/Caskroom/miniconda/base/envs/weaver/bin/python`) | weaver-core, torch, uproot — **no `balltree`** | all training, all tests but S2 |
| system `python3` | `balltree`, torch, uproot — **no weaver** | the ball-tree parity oracle only |
| `../erwin`'s `uv` venv (Py 3.9) | `balltree-erwin` | the upstream Erwin repo, not this one |

`tests/run_all.sh` splits across the first two on purpose (`WEAVER_PY` / `ORACLE_PY` override the paths). Don't try to merge them.

A trap S0 exists to catch: the sibling `../weaver/` fork directory has no `__init__.py`, so in an interpreter *without* weaver-core, `import weaver` still succeeds as an empty namespace package. Check `weaver.__file__`, not importability.

## Testing

No linter and no pytest. Tests are standalone scripts that print `[OK ...]` lines; `run_all.sh` counts those markers and reports PASS/FAIL per script — **a new check must emit `[OK ...]` to be counted**.

```bash
./tests/run_all.sh                                    # full S0-S9 suite
QUICK=1 ./tests/run_all.sh                            # smaller sweep + shorter overfit
$WEAVER_PY tests/s6_leakage.py                        # one check (weaver env)
python3 tests/s2_tree_parity.py                       # the oracle check (system python)

# v1 smoke tests (weaver env, from repo root)
python test_erwin_block.py        # imports assume a `particle_transformer/` parent layout
python test_cls_integration.py    # self-contained; padding isolation, grads, leakage
```

What each stage covers: `s0_env` weaver install · `s1_geometry` jet-frame coords · `s2_tree_parity` our pure-torch tree vs Erwin's compiled `balltree` · `s3_s4_invariance` permutation/rotation · `s5_pair_parity` ball pair features vs ParT's `PairEmbed` · `s6_leakage` cross-jet and padding leakage · `s7_gradients` · `s8_complexity` scaling · `s9_overfit` trainability.

Beyond the suite:

```bash
python tests/s10_compare.py --epochs 30 --batch 256   # ErwinParTv2 vs ParT, matched everything
python tests/s10_compare.py --models erwin --ball-size 16 --tag _m16
bash tests/s11_ballsweep.sh                           # level-0 ball size m ∈ {8,16,32,64}
bash tests/s11_ablations.sh                           # rot0/noU/noD/fine/coarse/logpolar
SEED=1 bash tests/s11_seedrepeat.sh                   # load-bearing arms at a second seed
python tests/flops_compare.py                         # FlopCounterMode, matched-L and production
python tests/make_plots.py                            # tests/s10_results/*.json -> plots/*.png
python tests/make_pdf.py && python tests/merge_pdf.py # -> results/*.pdf
```

S10 and S11 write `tests/s10_results/<tag>_{result,history,roc}.json` + `_best.pt`; `make_plots.py` only reads those — it never re-trains, so a number not in the JSON is not plotted. All of this runs on the checked-in `JetClass_example_100k.root` (100k jets), so treat the numbers as a controlled small-scale comparison, not published results.

Standalone v1 analysis scripts (weaver env, repo root, each saves a PNG): `benchmark_erwin_complexity.py` → `erwin_complexity.png`; `visualize_erwin_balls.py` → `erwin_balls.png`. `smoke_test_erwin.ipynb` trains v1 against the local sample.

## Architecture

**Weaver plugin contract** — every `networks/example_*.py` is loaded by path via `--network-config` and must expose:
- `get_model(data_config, **kwargs) -> (model, model_info)` — builds from the YAML data config (input dims from `data_config.input_dicts`, classes from `data_config.label_value`) and returns ONNX-export metadata. CLI `-o key value` arrives as `kwargs` and overrides the config dict.
- `get_loss(data_config, **kwargs) -> nn.Module`

The `forward` signature is fixed by the data config inputs: `forward(points, features, lorentz_vectors, mask)`.

**Data configs** (`data/<Dataset>/*.yaml`) are weaver YAML. The three JetClass variants (`kin`, `kinpid`, `full`) differ only in which particle features feed `pf_features`; the network input dim adapts automatically.

**`networks/EfficientParticleTransformer.py`** — one configurable model where `block_params['attn_type']` picks the attention: `LinBlock` dispatches `linformer`/`performer`/`reformer`/`mamba`/`pairs`, optionally with `PairEmbedFull` (ParT's attention bias); `'erwin'` instead uses `ParticleErwinBlock` (v1 ΔR-sorted chunks, learned distance bias, SwiGLU FFN, alternating `shift=True` roll by `ball_size // 2`, no pair embedding). The original ParT lives in weaver itself (`weaver.nn.model.ParticleTransformer`).

**`networks/erwin_part/` (ErwinParTv2)** — read the module docstrings; they carry the reasoning, not just the API. Import layering is load-bearing: `geometry.py` and `balltree_torch.py` depend only on torch so the tree can be verified against the compiled `balltree` in a *different interpreter*; `__init__.py` therefore resolves the weaver-backed names lazily via PEP 562. **Don't add an eager weaver import to those two modules or to `__init__.py`** — it breaks S2.

- `geometry.py` — jet-frame (Δη, Δφ) coordinates. Two invariants it exists to enforce: φ is periodic (all angular quantities are wrapped differences against the jet axis, never absolute φ), and padded slots are not particles (every reduction takes `valid`). Works in `(N, P, C)`; the `(N, 4, P)` / `(P, N, C)` conversions happen at call sites.
- `balltree_torch.py` — batched recursive median split, returns a permutation, so **same-ball points are contiguous at every level**: ball attention is a `.view`, pooling a strided reduction. Fixed `L` (power of two) per jet, unlike the ragged `next_pow2(n)` of Erwin's compiled package. Nothing in the training path imports `balltree`; it is only the test oracle.
- `ball_attention.py` — `softmax(QKᵀ/√d + U_ball − softplus(σ)·d̂ + keypad)V`, block-diagonal within balls. Two departures from the references: σ goes through `softplus` so the distance bias is negative by construction, and distances are normalised per ball so one σ init works at every level. Masked keys use `-1e9`, not `-inf` (a fully-masked softmax row is NaN on MPS). Residual structure is ParT's; only the FFN is Erwin's SwiGLU.
- `pair_features.py` — ParT's `U` without the O(N²). `PairEmbed` is pointwise `Conv1d(k=1)` over the pair axis, so it can run on within-ball pairs only (`L·m` instead of `L²` ≈ 32× fewer). Its leading BatchNorm is replaced by a **fixed per-level standardisation** (`PAIR_FEATURE_NORM`) because within-ball pairs are a biased (collinear) subsample and the statistics shift ~6 units across levels. Layer structure matches `PairEmbed` exactly so weights transfer both ways (`from_part_pair_embed`). The constants are geometry-specific — re-measure if `seq_len`/`ball_sizes`/`strides` change (note: `tests/calibrate_pair_norm.py`, referenced in the docstring, is not currently present).
- `hierarchy.py` — `BallPooling` (Eq. 12) sums children's **4-momenta**, so a coarse node is literally a subjet and coarse pair features are QCD-splitting observables. Invalid children are zeroed and an occupancy channel is appended so the projection can tell a full ball from a half-empty one. `BallUnpooling` exists but is unused: jet tagging runs encoder → bottleneck with `decode=False`.
- `model.py` — `SequenceTrimmer`/`Embed` from ParT, then L=64 pt-sorted slots → per-level trees → `BallBlock` stacks → ParT class attention. Three choices that diverge from a parent architecture and shouldn't be "fixed" casually: (1) **the bottleneck is a single ball** — with Erwin's stock `[8,8,8]` the 16 coarse nodes still split in two, so a two-prong decay can stay separated at every level; (2) **the readout is multi-scale class attention** over all 112 tokens, not a mean-pool, because a jet tag rides on a few hard constituents; (3) **widths are constant** across levels, so ParT hyperparameters transfer unchanged. Ablation switches (`use_pair_bias`, `use_dist_bias`, `rotate`, `readout`, `tree_space`) are wired through to `tests/s11_*.sh`.

Geometry for v2 comes from `pf_points` = `(part_deta, part_dphi)` as JetClass ships them (already φ-wrapped, in JetClass's own η sign convention `(η−η_jet)·sign(η_jet)`). Don't substitute positions recomputed from the Lorentz vectors unless you reproduce both — the tree would then partition in a different frame from the one the features describe.

**Cluster jobs** (`kube/`): Kubernetes Jobs for NRP Nautilus, namespace `cms-ml`. They `git clone` this fork at a branch into `/code`, `pip install -r requirements.txt`, symlink `/output` → `/code/training`, then run `train_JetClass.sh`; `*-pred-job.yml` variants run `test_JetClass.sh`. `erwinpartv2-job.yml` is the best template for a new job — it adds a pre-flight `get_model(...)` check against the data config so a broken plugin fails in seconds instead of after the dataset loads. Prefer copying the nearest existing YAML; the affinity/toleration/PVC boilerplate is the fiddly part. `NCCL_P2P_DISABLE` / `NCCL_IB_DISABLE` in `train_JetClass.sh` work around multi-GPU sync on that cluster.

**A job only runs code that is committed and pushed to the branch its YAML clones** (`BRANCH`, default `erwin_block`, in `erwinpartv2-job.yml`). Much of the v2 work is currently untracked — check `git status` before assuming a cluster run exercises what you just edited.

**Pre-trained models** (`models/*.pt`): JetClass-trained checkpoints for the `-FineTune` modes and as evaluation baselines.

## Workspace context

This repo sits in the `HEP-NN` workspace (not itself a repo) alongside `../erwin/` (upstream reference implementation and ball-tree source of truth), `../weaver/` (an unrelated older fork of the training framework — *not* the pip `weaver-core` this repo trains against), and `../hww-tagger/`. `../CLAUDE.md` covers what spans them; `../erwin/CLAUDE.md` covers ball-tree internals upstream.
