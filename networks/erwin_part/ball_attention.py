"""Ball-local multi-head self-attention, carrying ParT's interaction bias.

One attention block computes

    softmax( QK^T/sqrt(d)  +  U_ball  -  softplus(sigma_h) * d_hat  +  keypad ) V

block-diagonally within balls, where

    U_ball      ParT's pairwise interaction bias, evaluated only on within-ball
                pairs (pair_features.PairFeatureEmbed)
    d_hat       Erwin Eq. 10's distance bias, normalised by the ball radius
    keypad      -1e9 on padded keys

plus Erwin Eq. 9's relative position embedding folded into the values.

Two deliberate departures from the reference implementations:

*   **sigma is passed through softplus.** Erwin initialises `sigma_att` near -1
    and multiplies raw, so the bias is negative only because sigma happens to
    start negative - nothing keeps it there. `-softplus(sigma)` is negative by
    construction, so the bias always decays with distance.
*   **Distances are normalised per ball.** Balls at coarse levels are physically
    much wider (measured: mean intra-ball deltaR grows from 0.13 at level 0), so
    an unnormalised distance bias would swamp U at the top of the hierarchy and
    vanish at the bottom. Dividing by each ball's own radius makes sigma
    scale-free, so one initialisation works at every level.

The residual structure is ParT's, not Erwin's: `c_attn` head scaling, `w_resid`,
and pre/post LayerNorm are what keep deep ParT stacks trainable, and they let
ParT's hyperparameters transfer unchanged. Only the FFN is Erwin's SwiGLU.

`-1e9` rather than `-inf` for masked keys: on MPS a fully-masked softmax row
returns NaN, and a large finite penalty behaves identically everywhere else.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry import delta_r, masked_centroid

NEG = -1e9


class SwiGLU(nn.Module):
    """W_3( W_2 x  *  SiLU(W_1 x) ) - Erwin's feed-forward."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden)
        self.w2 = nn.Linear(dim, hidden)
        self.w3 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(self.w2(x) * F.silu(self.w1(x)))


def ball_distance_bias(pos_balls: torch.Tensor, valid_balls: torch.Tensor) -> torch.Tensor:
    """(N, nb, m, D), (N, nb, m) -> (N, nb, 1, m, m) normalised pairwise distance.

    Distances are angular (delta_R with a wrapped delta_phi) and divided by each
    ball's own radius - the mean distance of its real members from its centroid -
    so the scale is comparable across levels. Padded members contribute nothing
    to the radius.
    """
    with torch.no_grad():
        d = delta_r(pos_balls.unsqueeze(-2), pos_balls.unsqueeze(-3))   # (N, nb, m, m)
        centre = masked_centroid(pos_balls, valid_balls)                # (N, nb, 1, D)
        rad = delta_r(pos_balls, centre.expand_as(pos_balls))           # (N, nb, m)
        w = valid_balls.to(rad.dtype)
        radius = (rad * w).sum(-1) / w.sum(-1).clamp(min=1.0)           # (N, nb)
        d = d / radius.clamp(min=1e-4).unsqueeze(-1).unsqueeze(-1)

        # Zero any entry touching a padded slot. Padded positions sit at the
        # origin, which is nowhere near a jet's real constituents, so leaving
        # them in would put values of O(1000) in a tensor whose real entries
        # average 1.3. Those entries are masked or discarded downstream either
        # way; keeping them finite and small costs nothing and keeps the tensor
        # safe to inspect, autocast, and reason about.
        pair_valid = valid_balls.unsqueeze(-1) & valid_balls.unsqueeze(-2)
        return (d * pair_valid.to(d.dtype)).unsqueeze(2)


class BallMSA(nn.Module):
    """Multi-head self-attention within each ball."""

    def __init__(self, dim: int, num_heads: int, pos_dim: int = 2,
                 attn_dropout: float = 0.0):
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim {dim} not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attn_dropout = attn_dropout

        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.pe_proj = nn.Linear(pos_dim, dim)                    # W_pos, Eq. 9
        self.sigma = nn.Parameter(torch.zeros(1, num_heads, 1, 1))  # Eq. 10

    def forward(self, x: torch.Tensor, pos: torch.Tensor, valid: torch.Tensor,
                bias: torch.Tensor) -> torch.Tensor:
        """x (N, nb, m, C); pos (N, nb, m, D); valid (N, nb, m);
        bias (N, nb, H, m, m) = U_ball + keypad, already assembled.
        Returns (N, nb, m, C)."""
        N, nb, m, C = x.shape

        # Eq. 9: displacement from the ball's centre of mass, real members only.
        with torch.no_grad():
            rel = pos - masked_centroid(pos, valid)
        x = x + self.pe_proj(rel)

        qkv = self.qkv(x).view(N * nb, m, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)                      # (N*nb, H, m, hd)

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=bias.reshape(N * nb, self.num_heads, m, m),
            dropout_p=self.attn_dropout if self.training else 0.0)

        out = out.transpose(1, 2).reshape(N, nb, m, C)
        return self.proj(out)


class BallBlock(nn.Module):
    """ParT's block with ball-local attention and a SwiGLU feed-forward."""

    def __init__(self, embed_dim: int = 128, num_heads: int = 8, ffn_ratio: int = 4,
                 dropout: float = 0.1, attn_dropout: float = 0.1,
                 activation_dropout: float = 0.1, pos_dim: int = 2,
                 scale_attn: bool = True, scale_heads: bool = True,
                 scale_resids: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.pre_attn_norm = nn.LayerNorm(embed_dim)
        self.attn = BallMSA(embed_dim, num_heads, pos_dim, attn_dropout)
        self.post_attn_norm = nn.LayerNorm(embed_dim) if scale_attn else None
        self.dropout = nn.Dropout(dropout)

        self.pre_ffn_norm = nn.LayerNorm(embed_dim)
        self.ffn = SwiGLU(embed_dim, embed_dim * ffn_ratio)
        self.act_dropout = nn.Dropout(activation_dropout)

        self.c_attn = nn.Parameter(torch.ones(num_heads)) if scale_heads else None
        self.w_resid = nn.Parameter(torch.ones(embed_dim)) if scale_resids else None

    def forward(self, x: torch.Tensor, pos: torch.Tensor, valid: torch.Tensor,
                bias: torch.Tensor) -> torch.Tensor:
        """All inputs ball-shaped: x (N, nb, m, C). Returns the same shape."""
        N, nb, m, C = x.shape

        residual = x
        h = self.attn(self.pre_attn_norm(x), pos, valid, bias)
        if self.c_attn is not None:
            h = h.view(N, nb, m, self.num_heads, self.head_dim)
            h = torch.einsum("nbmhd,h->nbmhd", h, self.c_attn).reshape(N, nb, m, C)
        if self.post_attn_norm is not None:
            h = self.post_attn_norm(h)
        x = residual + self.dropout(h)

        residual = x
        h = self.act_dropout(self.ffn(self.pre_ffn_norm(x)))
        h = self.dropout(h)
        if self.w_resid is not None:
            residual = residual * self.w_resid
        x = residual + h

        # A padded slot must never carry state forward.
        return x * valid.unsqueeze(-1).to(x.dtype)


def assemble_bias(u_ball: torch.Tensor, valid_balls: torch.Tensor,
                  dist: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """U_ball + Erwin's distance bias + the padded-key penalty -> (N, nb, H, m, m).

    Built once per (level, rotation) and shared by every block at that level -
    the tree, the pair gather and the distance matrix are all expensive and all
    identical across blocks.
    """
    bias = u_ball - F.softplus(sigma) * dist
    key = valid_balls.unsqueeze(2).unsqueeze(3)                   # (N, nb, 1, 1, m)
    return bias.masked_fill(~key, NEG)
