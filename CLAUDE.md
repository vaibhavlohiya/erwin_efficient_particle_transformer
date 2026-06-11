# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A fork of the official [Particle Transformer (ParT)](https://arxiv.org/abs/2202.03772) jet-tagging repo, extended with *efficient* attention variants (Linformer, Reformer, Mamba, Erwin/BallMSA, pair-attention ablations) to reduce the O(N²) cost of ParT's pairwise-interaction attention. Models are trained on the JetClass / QuarkGluon / TopLandscape datasets.

Training is not run by code in this repo directly — it delegates to the **weaver** framework (`pip install 'weaver-core>=0.4'`), which provides the `weaver` CLI, data loading/transformation from ROOT files, and the training loop. This repo supplies the network definitions, dataset configs, and launcher scripts.

## Common commands

```bash
# Download a dataset (updates env.sh with the path)
./get_datasets.py [JetClass|QuarkGluon|TopLandscape] [-d DATA_DIR]

# Train on JetClass: arg1 = model name, arg2 = feature set (kin|kinpid|full)
./train_JetClass.sh ErwinParT full
./train_JetClass.sh ParT full

# Extra args are forwarded to weaver and override the script's defaults
./train_JetClass.sh ParT full --batch-size 256 --gpus 0,1 --num-workers 1

# DistributedDataParallel multi-GPU (batch size is per-GPU)
DDP_NGPUS=4 ./train_JetClass.sh ParT full --batch-size 256

# Evaluate / predict with a trained model (writes pred.root; expects /output paths, used in kube jobs)
./test_JetClass.sh ParT full

# Other datasets (support -FineTune variants starting from models/*.pt)
./train_QuarkGluon.sh [ParT|ParT-FineTune|PN|PN-FineTune|PFN|PCNN] [kin|kinpid|kinpidplus]
./train_TopLandscape.sh [ParT|ParT-FineTune|PN|PN-FineTune|PFN|PCNN] [kin]

# Smoke-test the Erwin block (no dataset needed; run from the weaver conda env)
python test_erwin_block.py
```

Model names accepted by `train_JetClass.sh`: `ParT`, `ParTNoPairs`, `LinformerParT`, `LinformerPairWise`, `ErwinParT`, `ReformerParT`, `MambaParT`, `PairAttnParT`, `MorePairAttnParT`, `PN`, `PFN`, `PCNN`. Each maps to a `networks/example_*.py` file and a tuned batch size / LR inside the script — adding a new model means adding a new `networks/example_<Name>.py` plus an `elif` branch there.

`env.sh` exports `DATADIR_JetClass` etc. and is sourced by the train scripts. `COMMENT=<suffix>` tags the output dir/tensorboard run. Checkpoints go to `training/JetClass/Pythia/<feature>/<model>/{auto}<suffix>/`.

There is no linter or test framework configured; `test_erwin_block.py` is a standalone smoke script (note: its imports reference a `particle_transformer` parent-dir layout and may need the repo checked out under that name, or the import path fixed).

## Architecture

**Weaver plugin contract** — every `networks/example_*.py` is passed to weaver via `--network-config` and must expose two functions:
- `get_model(data_config, **kwargs) -> (model, model_info)`: builds the network from the YAML data config (input dims from `data_config.input_dicts`, classes from `data_config.label_value`) and returns ONNX-export metadata. CLI `-o key value` options arrive as `kwargs` and override the config dict.
- `get_loss(data_config, **kwargs) -> nn.Module`

The model `forward` signature is fixed by the data config inputs: `forward(points, features, lorentz_vectors, mask)`.

**Data configs** (`data/<Dataset>/*.yaml`) are weaver YAML files defining input variables, preprocessing, and labels. The three JetClass variants (`kin`, `kinpid`, `full`) only differ in which particle features feed `pf_features`; the network input dim adapts automatically.

**`networks/EfficientParticleTransformer.py`** is the core of this fork — a single configurable model where `block_params['attn_type']` selects the attention mechanism inside each block:
- `LinBlock` dispatches `linformer` / `performer` / `reformer` / `mamba` / `pairs` attention, optionally combined with `PairEmbedFull` pairwise interaction features (the ParT attention bias).
- `attn_type: 'erwin'` instead uses `ParticleErwinBlock` / `ErwinTransformerBlock` (BallMSA: particles sorted by eta are grouped into balls of `ball_size` with a learned distance-based attention bias, plus a SwiGLU FFN; no pair embedding). Particle (eta, phi) positions are derived from the Lorentz-vector input by `_compute_positions`.
- The thin `example_*.py` wrappers each pin one `attn_type` + hyperparameters; the original ParT lives in weaver itself (`weaver.nn.model.ParticleTransformer`), imported by `example_ParticleTransformer.py`.

**Cluster jobs** (`kube/`): Kubernetes Job specs for the NRP Nautilus cluster. They mount a persistent volume, `pip install -e` weaver-core from the volume, and invoke the same `train_JetClass.sh` entry points. The `*-pred-job.yml` variants run `test_JetClass.sh`. The NCCL overrides exported in `train_JetClass.sh` (`NCCL_P2P_DISABLE`, `NCCL_IB_DISABLE`) exist to work around multi-GPU sync issues on that cluster.

**Pre-trained models** (`models/*.pt`): JetClass-trained checkpoints used by the `-FineTune` modes of the QuarkGluon/TopLandscape scripts and as evaluation baselines.
