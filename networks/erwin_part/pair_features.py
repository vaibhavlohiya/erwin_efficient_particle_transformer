"""Ball-local pairwise interaction features - ParT's U bias, without the O(N^2).

Stock ParT builds a dense interaction bias U of shape (N, H, S, S) from
`PairEmbed`, then adds it inside the softmax. The intermediate that MLP runs on
is (N, 64, S, S): at N=512, S=128, fp16 that is ~1.07 GB, and it - not the
attention matrix - is what actually dominates ParT's memory.

Ball attention only ever consumes U at within-ball index pairs. `PairEmbed` is a
stack of Conv1d(kernel=1), i.e. *pointwise over the pair axis*, so it can be
evaluated on any subset of pairs rather than on the full outer product. That is
the whole trick here: build the (i, j) pairs inside each ball first, then run the
same MLP on just those.

    dense   :  L * L        = 16384 pairs/jet at ParT's L=128
    ball    :  n_balls * m^2 = L * m = 512 pairs/jet at L=64, m=8
                                              -> 32x fewer, 32x less memory

Why the input normalisation is not BatchNorm
--------------------------------------------
`PairEmbed` begins with a BatchNorm1d over the pair features, calibrated across
*all* pairs. Within-ball pairs are a biased subsample - they are the collinear
ones - so those statistics no longer describe the input. Worse, the statistics
move with the level: measured on 4000 real jets, ln(m^2) runs

    level 0 (constituents) -1.96      level 1  +1.42      level 2  +4.34

a 6.3-unit shift, far more than one running mean can absorb. Each level
therefore gets its own *fixed* standardisation, measured from data, in the same
spirit as the hand-tuned constants in the weaver data config.

The module keeps `PairEmbed`'s exact layer structure so weights transfer
verbatim in both directions - see `from_part_pair_embed`, which is also how the
pre-trained ParT pair MLP gets reused for fine-tuning.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from weaver.nn.model.ParticleTransformer import pairwise_lv_fts

# (mean, std) of [ln kt, ln z, ln deltaR, ln m^2] over within-ball pairs,
# measured on 4000 JetClass jets with L=64, ball_sizes=[8,8,16], strides=[2,2].
# Recompute with tests/calibrate_pair_norm.py if that geometry changes.
PAIR_FEATURE_NORM = (
    ((-3.496, -1.436, -4.665, -1.957), (5.828, 0.770, 5.396, 4.343)),   # level 0
    ((-2.449, -1.347, -4.428, +1.424), (6.311, 0.683, 5.563, 1.909)),   # level 1
    ((-0.263, -1.378, -3.005, +4.342), (5.703, 0.709, 4.862, 1.788)),   # level 2
)


def within_ball_pair_features(p4_balls: torch.Tensor,
                              num_outputs: int = 4) -> torch.Tensor:
    """(N, n_balls, m, 4) -> (N, num_outputs, n_balls * m * m).

    Uses weaver's own `pairwise_lv_fts`, so the features are bit-identical to
    ParT's: [ln kt, ln z, ln deltaR, ln m^2], computed in rapidity with a
    properly wrapped delta_phi. They are constants of the input, hence no_grad -
    only the MLP downstream carries gradient.

    Pair index order is (ball, i, j), matching `ball_view`.
    """
    N, nb, m, _ = p4_balls.shape
    b4 = p4_balls.permute(0, 3, 1, 2)                              # (N, 4, nb, m)
    xi = b4.unsqueeze(-1).expand(N, 4, nb, m, m).reshape(N, 4, -1)
    xj = b4.unsqueeze(-2).expand(N, 4, nb, m, m).reshape(N, 4, -1)
    with torch.no_grad():
        return pairwise_lv_fts(xi, xj, num_outputs=num_outputs)


class PairFeatureEmbed(nn.Module):
    """ParT's pair MLP, evaluated on within-ball pairs.

    Args:
        dims:        hidden widths; the last entry must be `num_heads` so the
                     output is one bias channel per attention head.
        pair_dim:    number of pair features (4 = ln kt, ln z, ln dR, ln m^2).
        level:       which row of PAIR_FEATURE_NORM to standardise with.
        input_norm:  'fixed'  - frozen standardisation from PAIR_FEATURE_NORM
                                (default; see module docstring)
                     'batch'  - a live BatchNorm1d, exactly as ParT does it.
                                Use for parity against a stock PairEmbed.
                     'none'   - no input normalisation.
        use_pre_activation_pair: drop the final activation, as ParT does, so the
                     bias enters the softmax pre-activation.

    Layer structure is identical to `weaver...PairEmbed.embed`, so state dicts
    are interchangeable.
    """

    def __init__(self, dims: list[int], pair_dim: int = 4, level: int = 0,
                 input_norm: str = "fixed",
                 use_pre_activation_pair: bool = True,
                 activation: str = "gelu"):
        super().__init__()
        if input_norm not in ("fixed", "batch", "none"):
            raise ValueError(f"input_norm must be fixed|batch|none, got {input_norm}")
        self.pair_dim = pair_dim
        self.out_dim = dims[-1]
        self.input_norm = input_norm

        layers: list[nn.Module] = []
        if input_norm != "none":
            layers.append(nn.BatchNorm1d(pair_dim))
        in_dim = pair_dim
        for dim in dims:
            layers += [nn.Conv1d(in_dim, dim, 1), nn.BatchNorm1d(dim),
                       nn.GELU() if activation == "gelu" else nn.ReLU()]
            in_dim = dim
        if use_pre_activation_pair:
            layers = layers[:-1]
        self.embed = nn.Sequential(*layers)

        if input_norm == "fixed":
            mean, std = PAIR_FEATURE_NORM[min(level, len(PAIR_FEATURE_NORM) - 1)]
            bn: nn.BatchNorm1d = self.embed[0]
            with torch.no_grad():
                bn.running_mean.copy_(torch.tensor(mean[:pair_dim]))
                bn.running_var.copy_(torch.tensor(std[:pair_dim]) ** 2)
                bn.weight.fill_(1.0)
                bn.bias.zero_()
            bn.weight.requires_grad_(False)
            bn.bias.requires_grad_(False)

    def train(self, mode: bool = True):
        """Keep the fixed input standardiser in eval mode even while training."""
        super().train(mode)
        if self.input_norm == "fixed":
            self.embed[0].eval()
        return self

    @classmethod
    def from_part_pair_embed(cls, pair_embed, level: int = 0,
                             input_norm: str = "fixed") -> "PairFeatureEmbed":
        """Build from a trained weaver `PairEmbed`, copying its weights.

        This is the fine-tuning path: the pair MLP in `models/ParT_full.pt` has
        exactly this shape, so a JetClass-pretrained interaction bias transfers
        into ErwinParTv2 unchanged.
        """
        src = pair_embed.embed
        convs = [m for m in src if isinstance(m, nn.Conv1d)]
        dims = [c.out_channels for c in convs]
        has_in_norm = isinstance(src[0], nn.BatchNorm1d)
        pair_dim = convs[0].in_channels
        use_pre_act = not isinstance(src[-1], (nn.GELU, nn.ReLU))

        out = cls(dims, pair_dim=pair_dim, level=level, input_norm=input_norm,
                  use_pre_activation_pair=use_pre_act)
        # Skip the source's leading BatchNorm when we substitute our own.
        src_sd, dst_sd = src.state_dict(), out.embed.state_dict()
        offset = 0 if (has_in_norm and input_norm != "none") else 0
        for k in dst_sd:
            if input_norm == "fixed" and k.startswith("0."):
                continue                       # keep the measured standardiser
            if k in src_sd and src_sd[k].shape == dst_sd[k].shape:
                dst_sd[k] = src_sd[k].clone()
        out.embed.load_state_dict(dst_sd)
        return out

    def forward(self, p4_balls: torch.Tensor) -> torch.Tensor:
        """(N, n_balls, m, 4) -> (N, n_balls, H, m, m) additive attention bias."""
        N, nb, m, _ = p4_balls.shape
        feats = within_ball_pair_features(p4_balls, self.pair_dim)   # (N, D, nb*m*m)
        u = self.embed(feats)                                        # (N, H, nb*m*m)
        return u.view(N, self.out_dim, nb, m, m).permute(0, 2, 1, 3, 4)
