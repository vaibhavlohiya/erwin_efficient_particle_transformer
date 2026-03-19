import torch
from particle_transformer.networks.EfficientParticleTransformer import EfficientParticleTransformer
from weaver.utils.logger import _logger

'''
Erwin-based Particle Transformer.

Replaces the LinBlock attention blocks with ErwinTransformerBlock (ICML 2025):
  - Ball Multi-Head Self-Attention (BallMSA) groups nearby particles (sorted
    by eta) into balls of `ball_size` particles and applies scaled dot-product
    attention with a learned, distance-based attention bias (sigma_att).
  - A SwiGLU MLP replaces the standard FFN.
  - The pair embedding is dropped; spatial locality is handled entirely by
    BallMSA's distance-based attention bias.

Key parameters in block_params:
  attn_type  : 'erwin'   -- selects the ErwinTransformerBlock path
  ball_size  : int       -- particles per ball (default 16)
  ffn_ratio  : int       -- SwiGLU hidden-dim ratio (default 4)

Particle positions (eta, phi) are computed automatically from the
Lorentz-vector input (v: px, py, pz, energy).
'''


class EfficientParticleTransformerWrapper(torch.nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.mod = EfficientParticleTransformer(**kwargs)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'mod.cls_token'}

    def forward(self, points, features, lorentz_vectors, mask):
        return self.mod(features, v=lorentz_vectors, mask=mask)


def get_model(data_config, **kwargs):

    cfg = dict(
        input_dim=len(data_config.input_dicts['pf_features']),
        num_classes=len(data_config.label_value),
        # network configurations
        embed_dims=[128, 512, 128],
        num_heads=8,
        num_layers=8,
        num_cls_layers=2,
        block_params={
            'attn_type': 'erwin',
            'ball_size': 16,   # particles per BallMSA attention ball
            'ffn_ratio': 4,    # SwiGLU hidden-dim ratio
        },
        cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0},
        fc_params=[],
        activation='gelu',
        # misc
        trim=True,
        for_inference=False,
    )
    cfg.update(**kwargs)
    _logger.info('Model config: %s' % str(cfg))

    model = EfficientParticleTransformerWrapper(**cfg)

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
