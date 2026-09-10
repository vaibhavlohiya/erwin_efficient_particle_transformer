"""ErwinParTv2 - weaver network config.

Particle Transformer with Erwin's hierarchical ball-tree attention. See
`networks/erwin_part/` for the modules and `tests/` for the S0-S11 verification
suite.

    ./train_JetClass.sh ErwinParTv2 full

What differs from the v1 `ErwinParT` config in this repo:

  * a real 2-D ball tree (recursive median split in (delta_eta, delta_phi)),
    not a 1-D delta_R ordering cut into chunks; verified to partition
    identically to Erwin's compiled `balltree` on 512/512 real jets
  * ParT's pairwise interaction bias is KEPT, evaluated on within-ball pairs
    only - 32x fewer pairs than ParT's dense L=128 table
  * a U-Net hierarchy: constituents -> subjets -> subjets, where a coarse node
    carries the summed 4-momentum of its children, so the pair features between
    coarse nodes are the observables of a QCD splitting
  * class attention reads all three levels at once (112 tokens < ParT's 128)
  * the whole batch is vectorised - no Python loop over jets, so this runs at
    ParT's batch size rather than v1's 16

Geometry comes from `pf_points` = (part_deta, part_dphi), which JetClass ships
already wrapped in phi and in its own eta sign convention. Do not substitute
positions recomputed from the Lorentz vectors unless you reproduce both.
"""
import os
import sys

import torch

# The other example_*.py in this repo import via `particle_transformer.networks...`,
# which only resolves if the checkout happens to be named `particle_transformer`.
# weaver loads this file by path, so anchor the import to this file's own repo
# instead - that works whatever the directory is called.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from networks.erwin_part.model import ErwinParticleTransformerV2  # noqa: E402
from weaver.utils.logger import _logger  # noqa: E402


class ErwinParTv2Wrapper(torch.nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.mod = ErwinParticleTransformerV2(**kwargs)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'mod.cls_token', 'mod.level_embed'}

    def forward(self, points, features, lorentz_vectors, mask):
        return self.mod(features, v=lorentz_vectors, mask=mask, points=points)


def get_model(data_config, **kwargs):
    cfg = dict(
        input_dim=len(data_config.input_dicts['pf_features']),
        num_classes=len(data_config.label_value),
        embed_dims=[128, 512, 128],
        num_heads=8,
        # 64 slots -> 8 balls of 8 -> 4 balls of 8 -> ONE ball of 16.
        # The single-ball bottleneck is load-bearing: with Erwin's stock
        # [8, 8, 8] the 16 coarse nodes still split in two, and a two-prong
        # decay can sit in different balls at every level and never interact.
        seq_len=64,
        ball_sizes=[8, 8, 16],
        strides=[2, 2],
        depths=[2, 2, 2],
        num_cls_layers=2,
        pair_embed_dims=[64, 64],
        rotate=45.0,
        block_params={'dropout': 0.1, 'attn_dropout': 0.1, 'activation_dropout': 0.1},
        cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0},
        fc_params=[],
        activation='gelu',
        trim=True,
        for_inference=False,
    )
    cfg.update(**kwargs)
    _logger.info('Model config: %s' % str(cfg))

    model = ErwinParTv2Wrapper(**cfg)

    model_info = {
        'input_names': list(data_config.input_names),
        'input_shapes': {k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names': ['softmax'],
        'dynamic_axes': {
            **{k: {0: 'N', 2: 'n_' + k.split('_')[0]} for k in data_config.input_names},
            **{'softmax': {0: 'N'}},
        },
    }
    return model, model_info


def get_loss(data_config, **kwargs):
    return torch.nn.CrossEntropyLoss()
