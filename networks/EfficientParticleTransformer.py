""" Particle Transformer (ParT)

Paper: "Particle Transformer for Jet Tagging" - https://arxiv.org/abs/2202.03772
"""
import math
import random
import warnings
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from einops import rearrange

from weaver.utils.logger import _logger
from weaver.nn.model.ParticleTransformer import build_sparse_tensor, trunc_normal_, SequenceTrimmer, Embed, Block, pairwise_lv_fts


# ---------------------------------------------------------------------------
# Erwin components (adapted from erwin/models/erwin.py – ICML 2025)
# ---------------------------------------------------------------------------

class SwiGLU(nn.Module):
    """W_3 * SiLU(W_1 x) ⊗ W_2 x"""
    def __init__(self, in_dim: int, dim: int):
        super().__init__()
        self.w1 = nn.Linear(in_dim, dim)
        self.w2 = nn.Linear(in_dim, dim)
        self.w3 = nn.Linear(dim, in_dim)

    def forward(self, x: torch.Tensor):
        return self.w3(self.w2(x) * F.silu(self.w1(x)))


class BallMSA(nn.Module):
    """Ball Multi-Head Self-Attention (BMSA) with distance-based attention bias.

    Implements Eq. 9 (relative position embedding) and Eq. 10 (distance-based
    attention bias) of the Erwin paper:

        Eq. 9 :  X_B = X_B + (P_B - c_B) W_pos
        Eq. 10:  B_B = -sigma^2 * ||p_i - p_j||_2     (per head, learnable sigma)
    """
    def __init__(self, dim: int, num_heads: int, ball_size: int, dimensionality: int = 2):
        super().__init__()
        self.num_heads = num_heads
        self.ball_size = ball_size

        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        # W_pos of Eq. 9: projects the (P_B - c_B) relative coordinates to dim
        self.pe_proj = nn.Linear(dimensionality, dim)
        # sigma of Eq. 10, one per head. The bias is formed as -sigma^2 * dist,
        # so it is non-positive and decays with distance by construction,
        # whatever sign sigma drifts to during training.
        self.sigma_att = nn.Parameter(1.0 + 0.01 * torch.randn((1, num_heads, 1, 1)))

    @torch.no_grad()
    def pairwise_distances(self, pos: torch.Tensor):
        """Euclidean distance ||p_i - p_j||_2 between members of each ball."""
        pos = rearrange(pos, '(n m) d -> n m d', m=self.ball_size)   # (n_balls, B, d)
        return torch.cdist(pos, pos, p=2).unsqueeze(1)               # (n_balls, 1, B, B)

    @torch.no_grad()
    def compute_rel_pos(self, pos: torch.Tensor):
        """(P_B - c_B) of Eq. 9: c_B is the centroid (mean coordinate) of each ball."""
        num_balls = pos.shape[0] // self.ball_size
        pos = pos.view(num_balls, self.ball_size, pos.shape[1])      # (n_balls, B, d)
        return (pos - pos.mean(dim=1, keepdim=True)).view(-1, pos.shape[2])

    def forward(self, x: torch.Tensor, pos: torch.Tensor):
        # Eq. 9: add the projected ball-relative coordinates to the features
        x = x + self.pe_proj(self.compute_rel_pos(pos))              # (n_balls*B, C)
        q, k, v = rearrange(self.qkv(x), "(n m) (H E K) -> K n H m E",
                             H=self.num_heads, m=self.ball_size, K=3)
        # Eq. 10: structurally -sigma^2 * distance (negative, distance-decaying)
        attn_bias = -(self.sigma_att ** 2) * self.pairwise_distances(pos)  # (n_balls, H, B, B)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        x = rearrange(x, "n H m E -> (n m) (H E)", H=self.num_heads, m=self.ball_size)
        return self.proj(x)


class ErwinTransformerBlock(nn.Module):
    """Full Erwin transformer block: pre-norm BallMSA + pre-norm SwiGLU MLP."""
    def __init__(self, dim: int, num_heads: int, ball_size: int,
                 mlp_ratio: int = 4, dimensionality: int = 2):
        super().__init__()
        self.ball_size = ball_size
        self.norm1 = nn.RMSNorm(dim)
        self.norm2 = nn.RMSNorm(dim)
        self.BMSA = BallMSA(dim, num_heads, ball_size, dimensionality)
        self.swiglu = SwiGLU(dim, dim * mlp_ratio)

    def forward(self, x: torch.Tensor, pos: torch.Tensor):
        x = x + self.BMSA(self.norm1(x), pos)
        return x + self.swiglu(self.norm2(x))


# ---------------------------------------------------------------------------
# Position helper and ParticleErwinBlock
# ---------------------------------------------------------------------------

def _compute_positions(v: torch.Tensor) -> torch.Tensor:
    """
    Compute (eta, phi) pseudo-rapidity and azimuthal angle from 4-momentum.

    Args:
        v: (N, 4, P)  [px, py, pz, energy]
    Returns:
        pos: (N, P, 2) [eta, phi]
    """
    px, py, pz = v[:, 0], v[:, 1], v[:, 2]   # (N, P)
    pt = torch.sqrt(px ** 2 + py ** 2).clamp(min=1e-8)
    eta = torch.asinh(pz / pt)
    phi = torch.atan2(py, px)
    return torch.stack([eta, phi], dim=-1)     # (N, P, 2)


class ParticleErwinBlock(nn.Module):
    """
    Adapter that runs ErwinTransformerBlock inside EfficientParticleTransformer.

    For each jet (batch element):
      1. Extracts real (non-padded) particles.
      2. Sorts them by ascending Delta-R distance from the jet axis (the
         unweighted (eta, phi) centroid of the real particles), so consecutive
         particles fall into the same angular annulus around the jet core.
      3. Pads to a multiple of ball_size by repeating the last particle.
      4. If `shift=True` (cross-ball attention, Erwin paper Sec. 3.2), rolls
         the sorted sequence by ball_size // 2 so the new balls straddle two
         neighbouring balls of the unshifted layout; alternating shifted /
         unshifted layers propagates information across balls while keeping
         the O(P * ball_size) cost.
      5. Stacks all jets and calls ErwinTransformerBlock in one forward pass.
      6. Un-rolls, un-sorts and scatters the output back to (P, N, C).

    Interface matches LinBlock:
        forward(x, pos, padding_mask) -> x   with x: (P, N, C)
    """
    def __init__(self, embed_dim: int, num_heads: int,
                 ball_size: int = 16, mlp_ratio: int = 4, dimensionality: int = 2,
                 shift: bool = False):
        super().__init__()
        self.ball_size = ball_size
        self.shift = shift
        self.half_ball = ball_size // 2
        self.block = ErwinTransformerBlock(embed_dim, num_heads, ball_size,
                                           mlp_ratio, dimensionality)

    @staticmethod
    def _delta_r_sort_idx(real_pos: torch.Tensor) -> torch.Tensor:
        """
        Sorting indices by ascending angular distance from the jet axis.

        jet axis:  eta_jet = mean(eta_i), phi_jet = mean(phi_i)  (real particles)
        Delta R_i = sqrt((eta_i - eta_jet)^2 + (phi_i - phi_jet)^2)

        Args:
            real_pos: (num_real, 2)  [eta, phi] of the real particles of one jet
        Returns:
            sort_idx: (num_real,)    indices that sort by ascending Delta R
        """
        jet_axis = real_pos.mean(dim=0, keepdim=True)            # (1, 2)
        delta_r = torch.linalg.norm(real_pos - jet_axis, dim=-1)  # (num_real,)
        return torch.argsort(delta_r)

    def forward(self, x: torch.Tensor, pos: torch.Tensor,
                padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x:            (P, N, C)  seq_len × batch × embed_dim
            pos:          (N, P, 2)  eta, phi per particle per jet
            padding_mask: (N, P)     True = padded slot
        Returns:
            (P, N, C)
        """
        P, N, C = x.shape
        x_t = x.permute(1, 0, 2)   # (N, P, C)

        all_x_in, all_pos_in, scatter_info = [], [], []

        for n in range(N):
            if padding_mask is not None:
                real_mask = ~padding_mask[n]
            else:
                real_mask = torch.ones(P, dtype=torch.bool, device=x.device)

            num_real = int(real_mask.sum().item())
            if num_real == 0:
                scatter_info.append(None)
                continue

            real_x   = x_t[n, real_mask]       # (num_real, C)
            real_pos = pos[n, real_mask]        # (num_real, 2)

            # Sort by Delta R from the jet axis so consecutive particles are
            # spatially close (same annulus around the jet core)
            sort_idx = self._delta_r_sort_idx(real_pos)
            real_x   = real_x[sort_idx]
            real_pos = real_pos[sort_idx]

            # Pad to a multiple of ball_size by repeating the last particle
            pad_size   = (self.ball_size - num_real % self.ball_size) % self.ball_size
            padded_len = num_real + pad_size
            if pad_size > 0:
                real_x   = torch.cat([real_x,   real_x[-1:].expand(pad_size, C)], dim=0)
                real_pos = torch.cat([real_pos, real_pos[-1:].expand(pad_size, 2)], dim=0)

            # Cross-ball attention: cyclically roll the sorted sequence by half
            # a ball, so each ball of this layer straddles two balls of the
            # unshifted layers (positions roll together with their features).
            # With a single ball this permutes members within the ball, which
            # leaves the (permutation-equivariant) attention output unchanged.
            if self.shift:
                real_x   = torch.roll(real_x,   shifts=-self.half_ball, dims=0)
                real_pos = torch.roll(real_pos, shifts=-self.half_ball, dims=0)

            all_x_in.append(real_x)
            all_pos_in.append(real_pos)
            scatter_info.append((num_real, sort_idx, real_mask, padded_len))

        if not all_x_in:
            return x

        x_cat   = torch.cat(all_x_in,   dim=0)   # (total_padded, C)
        pos_cat = torch.cat(all_pos_in,  dim=0)   # (total_padded, 2)
        out_cat = self.block(x_cat, pos_cat)       # (total_padded, C)

        x_out  = x_t.clone()
        offset = 0
        for n in range(N):
            info = scatter_info[n]
            if info is None:
                continue
            num_real, sort_idx, real_mask, padded_len = info
            out_n = out_cat[offset:offset + padded_len]              # (padded_len, C)
            if self.shift:
                # Undo the cross-ball roll before dropping padding / un-sorting
                out_n = torch.roll(out_n, shifts=self.half_ball, dims=0)
            out_n = out_n[:num_real]                                  # (num_real, C)
            x_out[n, real_mask] = out_n[torch.argsort(sort_idx)]
            offset += padded_len

        return x_out.permute(1, 0, 2)   # (P, N, C)


# ---------------------------------------------------------------------------
# Analysis utilities: trainable parameters & theoretical FLOPs
# ---------------------------------------------------------------------------

def count_trainable_params(module: nn.Module) -> int:
    """
    Exact number of trainable parameters in `module`.

    For ParticleErwinBlock with embed dim C, heads H, MLP ratio r and
    coordinate dimensionality d, the closed form is:

        BallMSA :  qkv     3C^2 + 3C
                   proj     C^2 +  C
                   pe_proj   dC +  C
                   sigma            H
        SwiGLU  :  w1, w2  2(rC^2 + rC)
                   w3       rC^2 +  C
        RMSNorm :  x2              2C

        total = (4 + 3r) C^2 + (d + 2r + 8) C + H

    e.g. C=128, r=4, d=2, H=8  ->  16*128^2 + 18*128 + 8 = 264,456.
    """
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def erwin_block_flops(N: int, P: int, C: int, num_heads: int = 8,
                      ball_size: int = 16, mlp_ratio: int = 4,
                      dimensionality: int = 2) -> dict:
    """
    Theoretical FLOPs of one forward pass through ParticleErwinBlock.

    Conventions:
      * one multiply-accumulate = 2 FLOPs;
      * T = N * P tokens (assumes P is a multiple of B; ball padding adds at
        most B-1 tokens per jet, a negligible correction);
      * B = ball_size, H = num_heads, r = mlp_ratio, d = dimensionality.

    Derivation (every term is linear in P):

      RPE, Eq. 9          Linear(d -> C)                       2 T d C
      QKV projection      Linear(C -> 3C)                      6 T C^2
      Distance bias       B^2 pairs/ball, d dims, ~3 ops each  3 T B d
      Attention scores    Q K^T per ball: H * (B x E)(E x B),
                          H E = C                              2 T B C
      Softmax             ~5 ops per score, H heads            5 T B H
      Attention values    softmax(.) V                         2 T B C
      Output projection   Linear(C -> C)                       2 T C^2
      SwiGLU MLP          w1, w2: Linear(C -> rC),
                          w3: Linear(rC -> C)                  6 r T C^2
                          SiLU + gating product                4 r T C
      RMSNorm x2          ~4 ops per element each              8 T C

      total ~= N P [ (8 + 6r) C^2 + 4 B C + lower-order ]

    The attention cost is O(N * P * B * C): *linear* in sequence length P,
    versus O(N * P^2 * C) for full pairwise attention.

    Returns a dict with the per-component breakdown and 'total'.
    """
    T, B, H, r, d = N * P, ball_size, num_heads, mlp_ratio, dimensionality
    flops = {
        'rpe_projection':    2 * T * d * C,
        'qkv_projection':    6 * T * C * C,
        'distance_bias':     3 * T * B * d,
        'attention_scores':  2 * T * B * C,
        'softmax':           5 * T * B * H,
        'attention_values':  2 * T * B * C,
        'output_projection': 2 * T * C * C,
        'swiglu_linear':     6 * r * T * C * C,
        'swiglu_gating':     4 * r * T * C,
        'rmsnorm':           8 * T * C,
    }
    flops['total'] = sum(flops.values())
    return flops


def analyze_erwin_block(block: 'ParticleErwinBlock', N: int = 1, P: int = 128) -> dict:
    """Print and return the parameter count and FLOPs breakdown of `block`."""
    C = block.block.norm1.weight.shape[0]
    H = block.block.BMSA.num_heads
    d = block.block.BMSA.pe_proj.in_features
    r = block.block.swiglu.w1.out_features // C

    n_params = count_trainable_params(block)
    flops = erwin_block_flops(N, P, C, num_heads=H, ball_size=block.ball_size,
                              mlp_ratio=r, dimensionality=d)

    print(f"ParticleErwinBlock  (C={C}, H={H}, B={block.ball_size}, r={r}, d={d})")
    print(f"  trainable parameters : {n_params:,}")
    print(f"  forward FLOPs (N={N}, P={P}):")
    for name, val in flops.items():
        marker = '  total ->' if name == 'total' else '          '
        print(f"  {marker} {name:<18s} {val:>14,}")
    return {'params': n_params, 'flops': flops}


def to_qtypedderr(x):
    # x: (N, 17, ...),
    # dim1: [pt_log, e_log, logptrel, logerel, deltaR,
    # charge, isChargedHadron, isNeutralHadron, isPhoton, isElectron, isMuon,
    # d0, d0err, dz, dzerr,
    # deta, dphi]
    kin, qtype, d0, d0err, dz, dzerr, detaphi = x.split((5, 6, 1, 1, 1, 1, 2), dim=1)
    return qtype, torch.cat((d0, dz), dim=1), torch.cat((d0err, dzerr), dim=1)

def pairwise_x_fts(xi, xj, num_outputs=10, eps=1e-8):
    qtypei, di, derri = to_qtypedderr(xi)
    qtypej, dj, derrj = to_qtypedderr(xj)
    qtype = qtypei + qtypej
    d = (di + dj) / (1 + di*dj + eps)
    derr = torch.sqrt(derri**2 + derrj**2)
    outputs = torch.cat([qtype, d, derr], dim=1)
    assert num_outputs == outputs.size(1)
    return outputs

class PairEmbedFull(nn.Module):
    def __init__(
            self, pairwise_lv_dim, pairwise_x_dim, pairwise_input_dim, dims,
            remove_self_pair=False, use_pre_activation_pair=True, mode='concat',
            normalize_input=True, activation='gelu', eps=1e-8,
            for_onnx=False):
        super().__init__()

        self.pairwise_lv_dim = pairwise_lv_dim
        self.pairwise_x_dim = pairwise_x_dim
        self.pairwise_input_dim = pairwise_input_dim
        self.is_symmetric = (pairwise_lv_dim <= 5) and (pairwise_input_dim == 0)
        self.remove_self_pair = remove_self_pair
        self.mode = mode
        self.for_onnx = for_onnx
        self.pairwise_lv_fts = partial(pairwise_lv_fts, num_outputs=pairwise_lv_dim, eps=eps, for_onnx=for_onnx)
        self.pairwise_x_fts = partial(pairwise_x_fts, num_outputs=pairwise_x_dim)
        self.out_dim = dims[-1]

        if self.mode == 'concat':
            input_dim = pairwise_lv_dim + pairwise_x_dim + pairwise_input_dim
            module_list = [nn.BatchNorm1d(input_dim)] if normalize_input else []
            for dim in dims:
                module_list.extend([
                    nn.Conv1d(input_dim, dim, 1),
                    nn.BatchNorm1d(dim),
                    nn.GELU() if activation == 'gelu' else nn.ReLU(),
                ])
                input_dim = dim
            if use_pre_activation_pair:
                module_list = module_list[:-1]
            self.embed = nn.Sequential(*module_list)
        elif self.mode == 'sum':
            if pairwise_lv_dim > 0:
                input_dim = pairwise_lv_dim
                module_list = [nn.BatchNorm1d(input_dim)] if normalize_input else []
                for dim in dims:
                    module_list.extend([
                        nn.Conv1d(input_dim, dim, 1),
                        nn.BatchNorm1d(dim),
                        nn.GELU() if activation == 'gelu' else nn.ReLU(),
                    ])
                    input_dim = dim
                if use_pre_activation_pair:
                    module_list = module_list[:-1]
                self.embed = nn.Sequential(*module_list)
            if pairwise_x_dim > 0:
                input_dim = pairwise_x_dim
                module_list = [nn.BatchNorm1d(input_dim)] if normalize_input else []
                for dim in dims:
                    module_list.extend([
                        nn.Conv1d(input_dim, dim, 1),
                        nn.BatchNorm1d(dim),
                        nn.GELU() if activation == 'gelu' else nn.ReLU(),
                    ])
                    input_dim = dim
                if use_pre_activation_pair:
                    module_list = module_list[:-1]
                self.x_embed = nn.Sequential(*module_list)

            if pairwise_input_dim > 0:
                input_dim = pairwise_input_dim
                module_list = [nn.BatchNorm1d(input_dim)] if normalize_input else []
                for dim in dims:
                    module_list.extend([
                        nn.Conv1d(input_dim, dim, 1),
                        nn.BatchNorm1d(dim),
                        nn.GELU() if activation == 'gelu' else nn.ReLU(),
                    ])
                    input_dim = dim
                if use_pre_activation_pair:
                    module_list = module_list[:-1]
                self.fts_embed = nn.Sequential(*module_list)
        else:
            raise RuntimeError('`mode` can only be `sum` or `concat`')

    def forward(self, x, z, uu=None):
        # x: (batch, v_dim, seq_len)
        # z: (batch, x_dim, seq_len)
        # uu: (batch, v_dim, seq_len, seq_len)
        assert (x is not None or uu is not None)
        with torch.no_grad():
            if x is not None:
                batch_size, _, seq_len = x.size()
            else:
                batch_size, _, seq_len, _ = uu.size()
            if self.is_symmetric and not self.for_onnx:
                i, j = torch.tril_indices(seq_len, seq_len, offset=-1 if self.remove_self_pair else 0,
                                          device=(x if x is not None else uu).device)
                if x is not None:
                    x = x.unsqueeze(-1).repeat(1, 1, 1, seq_len)
                    xi = x[:, :, i, j]  # (batch, dim, seq_len*(seq_len+1)/2)
                    xj = x[:, :, j, i]
                    x = self.pairwise_lv_fts(xi, xj)
                if z is not None:
                    z = z.unsqueeze(-1).repeat(1, 1, 1, seq_len)
                    zi = z[:, :, i, j]
                    zj = z[:, :, j, i]
                    z = self.pairwise_x_fts(zi, zj)
                if uu is not None:
                    # (batch, dim, seq_len*(seq_len+1)/2)
                    uu = uu[:, :, i, j]
            else:
                if x is not None:
                    x = self.pairwise_lv_fts(x.unsqueeze(-1), x.unsqueeze(-2))
                    if self.remove_self_pair:
                        i = torch.arange(0, seq_len, device=x.device)
                        x[:, :, i, i] = 0
                    x = x.view(-1, self.pairwise_lv_dim, seq_len * seq_len)
                if uu is not None:
                    uu = uu.view(-1, self.pairwise_input_dim, seq_len * seq_len)
            if self.mode == 'concat':
                pair_fts = []
                if x is not None:
                    pair_fts.append(x)
                if z is not None:
                    pair_fts.append(z)
                if uu is not None:
                    pair_fts.append(uu)
                if len(pair_fts) == 1:
                    pair_fts = pair_fts[0]
                else:
                    pair_fts = torch.cat(pair_fts, dim=1)

        if self.mode == 'concat':
            elements = self.embed(pair_fts)  # (batch, embed_dim, num_elements)
        elif self.mode == 'sum':
            elements = 0
            if x is not None:
                elements += self.embed(x)
            if z is not None:
                elements += self.x_embed(z)
            elif uu is not None:
                elements += self.fts_embed(x)

        if self.is_symmetric and not self.for_onnx:
            y = torch.zeros(batch_size, self.out_dim, seq_len, seq_len, dtype=elements.dtype, device=elements.device)
            y[:, :, i, j] = elements
            y[:, :, j, i] = elements
        else:
            y = elements.view(-1, self.out_dim, seq_len, seq_len)
        return y

class PairAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask):
        # x: (P, N, C)
        # attn_mask: (N*num_heads, P, P)
        # output: (P, N, C)
        seq_len = x.size(0)
        v = self.v_proj(x).view(-1, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (N, num_heads, P, head_dim)
        attn = self.dropout(torch.softmax(attn_mask, dim=-1).view(-1, self.num_heads, seq_len, seq_len))  # (N, num_heads, P, P)
        output = torch.matmul(attn, v).permute(2, 0, 1, 3).contiguous().view(seq_len, -1, self.embed_dim)  # (P, N, C)
        return output, attn

class LinBlock(nn.Module):
    def __init__(
        self,
        embed_dim=128,
        num_heads=8,
        max_seq_len=128,
        attn_type="linformer",
        compressed=4,
        bucket_size=32,
        n_hashes=4,
        d_state=16,
        d_conv=4,
        expand=2,
        ffn_ratio=4,
        dropout=0.1,
        attn_dropout=0.1,
        activation_dropout=0.1,
        add_bias_kv=False,
        activation="gelu",
        scale_fc=True,
        scale_attn=True,
        scale_heads=True,
        scale_resids=True,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.compressed = compressed
        self.head_dim = embed_dim // num_heads
        self.ffn_dim = embed_dim * ffn_ratio
        self.attn_type = attn_type

        self.pre_attn_norm = nn.LayerNorm(embed_dim)
        if self.attn_type == "linformer":
            from particle_transformer.networks.multihead_linear_attention import MultiheadLinearAttention
            shared_compress_layer = nn.Linear(max_seq_len, max_seq_len // compressed, bias=False)
            self.attn = MultiheadLinearAttention(
                embed_dim,
                num_heads,
                dropout=attn_dropout,
                add_bias_kv=add_bias_kv,
                max_seq_len=max_seq_len,
                compressed=compressed,
                self_attention=True,
                shared_kv_compressed=1,
                shared_compress_layer=shared_compress_layer,
            )
        elif self.attn_type == "reformer":
            from reformer_pytorch import LSHSelfAttention
            self.attn = LSHSelfAttention(
                embed_dim,
                heads=num_heads,
                bucket_size=bucket_size,
                n_hashes=n_hashes,
                causal=False,
                dropout=attn_dropout,
            )
        elif self.attn_type == "mamba":
            from mamba_ssm import Mamba
            self.attn = Mamba(
                d_model=embed_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        elif self.attn_type == "pairs":
            self.attn = PairAttention(embed_dim, num_heads, dropout=attn_dropout)
        self.post_attn_norm = nn.LayerNorm(embed_dim) if scale_attn else None
        self.dropout = nn.Dropout(dropout)

        self.pre_fc_norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, self.ffn_dim)
        self.act = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.act_dropout = nn.Dropout(activation_dropout)
        self.post_fc_norm = nn.LayerNorm(self.ffn_dim) if scale_fc else None
        self.fc2 = nn.Linear(self.ffn_dim, embed_dim)

        self.c_attn = (
            nn.Parameter(torch.ones(num_heads), requires_grad=True)
            if scale_heads
            else None
        )
        self.w_resid = (
            nn.Parameter(torch.ones(embed_dim), requires_grad=True)
            if scale_resids
            else None
        )

    def forward(self, x, padding_mask=None, attn_mask=None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, seq_len)` where padding
                elements are indicated by ``1``.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """

        residual = x
        x = self.pre_attn_norm(x)
        if self.attn_type == "linformer":
            x = self.attn(x, x, x, key_padding_mask=padding_mask, attn_mask=attn_mask)[
                0
            ]  # (seq_len, batch, embed_dim)
        elif self.attn_type == "performer":
            x = self.attn(x, x, input_mask=padding_mask, attn_mask=attn_mask)[
                0
            ]  # (seq_len, batch, embed_dim)
        elif self.attn_type == "reformer":
            x = self.attn(x)
        elif self.attn_type == "mamba":
            x = self.attn(x)
        elif self.attn_type == "pairs":
            x = self.attn(x, attn_mask)[0]

        if self.c_attn is not None:
            tgt_len = x.size(0)
            x = x.view(tgt_len, -1, self.num_heads, self.head_dim)
            x = torch.einsum("tbhd,h->tbdh", x, self.c_attn)
            x = x.reshape(tgt_len, -1, self.embed_dim)
        if self.post_attn_norm is not None:
            x = self.post_attn_norm(x)
        x = self.dropout(x)
        x += residual

        residual = x
        x = self.pre_fc_norm(x)
        x = self.act(self.fc1(x))
        x = self.act_dropout(x)
        if self.post_fc_norm is not None:
            x = self.post_fc_norm(x)
        x = self.fc2(x)
        x = self.dropout(x)
        if self.w_resid is not None:
            residual = torch.mul(self.w_resid, residual)
        x += residual

        return x


class EfficientParticleTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        num_classes=None,
        # network configurations
        pair_input_dim=4,
        pair_more_input_dim=0,
        pair_extra_dim=0,
        remove_self_pair=False,
        use_pre_activation_pair=True,
        embed_dims=[128, 512, 128],
        pair_embed_dims=None,  # [64, 64, 64],
        num_heads=8,
        num_layers=8,
        num_cls_layers=2,
        block_params=None,
        cls_block_params={"dropout": 0, "attn_dropout": 0, "activation_dropout": 0},
        fc_params=[],
        activation="gelu",
        # misc
        trim=True,
        for_inference=False,
        use_amp=False,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

        self.trimmer = SequenceTrimmer(enabled=trim and not for_inference)
        self.for_inference = for_inference
        self.use_amp = use_amp

        embed_dim = embed_dims[-1] if len(embed_dims) > 0 else input_dim
        default_cfg = dict(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ffn_ratio=4,
            dropout=0.1,
            attn_dropout=0.1,
            activation_dropout=0.1,
            add_bias_kv=False,
            activation=activation,
            scale_fc=True,
            scale_attn=True,
            scale_heads=True,
            scale_resids=True,
        )

        cfg_block = copy.deepcopy(default_cfg)
        if block_params is not None:
            cfg_block.update(block_params)
        _logger.info("cfg_block: %s" % str(cfg_block))

        cfg_cls_block = copy.deepcopy(default_cfg)
        if cls_block_params is not None:
            cfg_cls_block.update(cls_block_params)
        _logger.info("cfg_cls_block: %s" % str(cfg_cls_block))

        self.pair_extra_dim = pair_extra_dim
        self.embed = (
            Embed(input_dim, embed_dims, activation=activation)
            if len(embed_dims) > 0
            else nn.Identity()
        )
        self.pair_more_input_dim = pair_more_input_dim

        # When using Erwin blocks the pair embedding is not used (BallMSA handles
        # spatial interactions via its built-in distance-based attention bias).
        if cfg_block.get('attn_type') == 'erwin':
            self.use_erwin_blocks = True
            self.pair_embed = None
            ball_size = cfg_block.get('ball_size', 16)
            mlp_ratio  = cfg_block.get('ffn_ratio', 4)
            # Alternate unshifted / shifted ball layouts (cross-ball attention,
            # Erwin paper Sec. 3.2): odd layers roll the sorted sequence by
            # ball_size // 2 so information propagates across ball boundaries.
            self.blocks = nn.ModuleList([
                ParticleErwinBlock(embed_dim, num_heads, ball_size=ball_size,
                                   mlp_ratio=mlp_ratio, shift=(i % 2 == 1))
                for i in range(num_layers)
            ])
        else:
            self.use_erwin_blocks = False
            self.pair_embed = PairEmbedFull(
                pair_input_dim, pair_more_input_dim, pair_extra_dim,
                pair_embed_dims + [cfg_block['num_heads']],
                remove_self_pair=remove_self_pair,
                use_pre_activation_pair=use_pre_activation_pair,
                for_onnx=for_inference,
            ) if pair_embed_dims is not None and pair_input_dim + pair_more_input_dim + pair_extra_dim > 0 else None
            self.blocks = nn.ModuleList([LinBlock(**cfg_block) for _ in range(num_layers)])
        self.cls_blocks = nn.ModuleList(
            [Block(**cfg_cls_block) for _ in range(num_cls_layers)]
        )
        self.norm = nn.LayerNorm(embed_dim)

        if fc_params is not None:
            fcs = []
            in_dim = embed_dim
            for out_dim, drop_rate in fc_params:
                fcs.append(
                    nn.Sequential(
                        nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(drop_rate)
                    )
                )
                in_dim = out_dim
            fcs.append(nn.Linear(in_dim, num_classes))
            self.fc = nn.Sequential(*fcs)
        else:
            self.fc = None

        # init
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        trunc_normal_(self.cls_token, std=0.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            "cls_token",
        }

    def forward(self, x, v=None, mask=None, uu=None, uu_idx=None):
        # x: (N, C, P)
        # v: (N, 4, P) [px,py,pz,energy]
        # mask: (N, 1, P) -- real particle = 1, padded = 0
        # for pytorch: uu (N, C', num_pairs), uu_idx (N, 2, num_pairs)
        # for onnx: uu (N, C', P, P), uu_idx=None

        with torch.no_grad():
            if not self.for_inference:
                if uu_idx is not None:
                    uu = build_sparse_tensor(uu, uu_idx, x.size(-1))
            x, v, mask, uu = self.trimmer(x, v, mask, uu)
            padding_mask = ~mask.squeeze(1)  # (N, P)

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            x_in = x if self.pair_more_input_dim > 0 else None
            # input embedding
            x = self.embed(x).masked_fill(~mask.permute(2, 0, 1), 0)  # (P, N, C)

            # transform
            if self.use_erwin_blocks:
                # Compute (eta, phi) from 4-momentum for BallMSA positional encoding
                assert v is not None, (
                    "Erwin blocks require the 4-momentum tensor (v) to compute "
                    "(eta, phi) positions for BallMSA."
                )
                pos = _compute_positions(v.float())  # (N, P, 2), always float32
                for block in self.blocks:
                    x = block(x, pos=pos, padding_mask=padding_mask)
            else:
                attn_mask = None
                if (v is not None or x_in is not None or uu is not None) and self.pair_embed is not None:
                    attn_mask = self.pair_embed(v, x_in, uu).view(-1, v.size(-1), v.size(-1))  # (N*num_heads, P, P)
                for block in self.blocks:
                    x = block(x, padding_mask=padding_mask, attn_mask=attn_mask)

            # extract class token
            cls_tokens = self.cls_token.expand(1, x.size(1), -1)  # (1, N, C)
            for block in self.cls_blocks:
                cls_tokens = block(x, x_cls=cls_tokens, padding_mask=padding_mask)

            x_cls = self.norm(cls_tokens).squeeze(0)

            # fc
            if self.fc is None:
                return x_cls
            output = self.fc(x_cls)
            if self.for_inference:
                output = torch.softmax(output, dim=1)
            # print('output:\n', output)
            return output


class EfficientParticleTransformerTagger(nn.Module):
    def __init__(
        self,
        pf_input_dim,
        sv_input_dim,
        num_classes=None,
        # network configurations
        embed_dims=[128, 512, 128],
        num_heads=8,
        num_layers=8,
        num_cls_layers=2,
        block_params=None,
        cls_block_params={"dropout": 0, "attn_dropout": 0, "activation_dropout": 0},
        fc_params=[],
        activation="gelu",
        # misc
        trim=True,
        for_inference=False,
        use_amp=False,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

        self.use_amp = use_amp

        self.pf_trimmer = SequenceTrimmer(enabled=trim and not for_inference)
        self.sv_trimmer = SequenceTrimmer(enabled=trim and not for_inference)

        self.pf_embed = Embed(pf_input_dim, embed_dims, activation=activation)
        self.sv_embed = Embed(sv_input_dim, embed_dims, activation=activation)

        self.part = EfficientParticleTransformer(
            input_dim=embed_dims[-1],
            num_classes=num_classes,
            # network configurations
            embed_dims=[],
            num_heads=num_heads,
            num_layers=num_layers,
            num_cls_layers=num_cls_layers,
            block_params=block_params,
            cls_block_params=cls_block_params,
            fc_params=fc_params,
            activation=activation,
            # misc
            trim=False,
            for_inference=for_inference,
            use_amp=use_amp,
        )

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            "part.cls_token",
        }

    def forward(
        self, pf_x, pf_v=None, pf_mask=None, sv_x=None, sv_v=None, sv_mask=None
    ):
        # x: (N, C, P)
        # v: (N, 4, P) [px,py,pz,energy]
        # mask: (N, 1, P) -- real particle = 1, padded = 0

        with torch.no_grad():
            pf_x, pf_v, pf_mask, _ = self.pf_trimmer(pf_x, pf_v, pf_mask)
            sv_x, sv_v, sv_mask, _ = self.sv_trimmer(sv_x, sv_v, sv_mask)
            v = torch.cat([pf_v, sv_v], dim=2)
            mask = torch.cat([pf_mask, sv_mask], dim=2)

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            pf_x = self.pf_embed(pf_x)  # after embed: (seq_len, batch, embed_dim)
            sv_x = self.sv_embed(sv_x)
            x = torch.cat([pf_x, sv_x], dim=0)

            return self.part(x, v, mask)


if __name__ == "__main__":
    # Standalone analysis: parameters & theoretical FLOPs of ParticleErwinBlock
    # (matches the ErwinParT config in networks/example_ErwinParticleTransformer.py)
    #   python networks/EfficientParticleTransformer.py
    C, H, B, r = 128, 8, 32, 4
    block = ParticleErwinBlock(C, H, ball_size=B, mlp_ratio=r)
    analyze_erwin_block(block, N=1, P=128)
