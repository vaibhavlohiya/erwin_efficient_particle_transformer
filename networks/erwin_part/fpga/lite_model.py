"""ErwinParTLite - ErwinParTv2's hierarchy in a fixed-size, FPGA-friendly form.

What it keeps from ErwinParTv2, because that is what sets it apart from flat
FPGA jet taggers (SCRAMJet, LEAF, PHAT-JeT, JetFormer):

* spatial tiles of `ball` particles with ParT's pairwise physics bias U_ij
  inside every tile, and a distance bias scaled to each tile;
* a hierarchy: pooling sums the children's 4-momenta, so coarser levels are
  physical subjets and their pair features are splitting observables;
* a single global ball at the top level and a multi-scale class-attention readout.

What it changes for hardware (all switchable, see LiteConfig):

* every size is fixed at construction - L slots, tile size, levels - and there is
  no Python control flow on tensor values, so torch.fx can trace it (except the
  'tree' reference layout, which reuses ErwinParTv2's recursive median split);
* no SequenceTrimmer, no Lund branch;
* tiles come from one Morton (Z-order) sort, with a half-cell-shifted grid in
  alternate blocks in place of the rotated tree ('morton'); or from the pT order
  untouched ('pt', the flat comparison baseline);
* optional ball summaries: every query also attends to one summary token per
  tile of its level ('plain'), optionally with the pair features between the
  query and the tile's summed 4-momentum as their bias ('physics');
* pair features exact (ParT's) or from table logarithms ('lut', see approx.py);
* the normalisation is selectable (norms.py);
* width, FFN ratio, depths, class-attention depth and the pair network's size
  are all parameters, so the model can shrink for a trigger budget.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.fx
import torch.nn as nn
import torch.nn.functional as F

from .approx import _log2_table, exact_features, lut_features, lut_log, lut_pt
from .norms import make_norm
from .partition import gather_slots, morton_order

# ParT's exact pair features call weaver's TorchScript `delta_r2`, which fx cannot
# trace into; record the call as one leaf node instead. (The FPGA path is 'lut'.)
torch.fx.wrap("exact_features")

NEG = -1e9


@dataclass
class LiteConfig:
    input_dim: int
    num_classes: int = 10
    seq_len: int = 64                    # slots kept, top pT first
    D: int = 128                         # width
    num_heads: int = 0                   # 0 -> max(1, D // 16)
    ffn_ratio: int = 4
    embed_ratio: int = 4                 # embedding MLP C -> D -> ratio*D -> D
    ball: int = 8                        # tile size on every level but the top
    strides: tuple = (2, 2)              # pooling factor between levels
    depths: tuple = (2, 2, 2)            # blocks per level
    hierarchy: bool = True               # False: one level, len(depths) == 1
    cls_layers: int = 2
    partition: str = "morton"            # morton | pt | tree
    morton_bits: int = 5
    shift: bool = True                   # half-cell-shifted tiles in alternate blocks
    ball_summary: str = "off"            # off | plain | physics
    pair_features: str = "exact"         # exact | lut | none
    pair_hidden: tuple = (64, 64)
    lut_bits: int = 6
    use_dist_bias: bool = True
    dist_cap: float = 0.0                # >0: saturate the tile-normalised distance (see _dist)
    norm: str = "layernorm"              # layernorm | batchnorm | affine | none
    dropout: float = 0.1
    attn_dropout: float = 0.1
    level_sizes: list = field(default_factory=list, init=False)
    ball_sizes: list = field(default_factory=list, init=False)

    def __post_init__(self):
        if self.partition not in ("morton", "pt", "tree"):
            raise ValueError(f"partition must be morton|pt|tree, got {self.partition!r}")
        if self.ball_summary not in ("off", "plain", "physics"):
            raise ValueError(f"ball_summary must be off|plain|physics, got {self.ball_summary!r}")
        if self.pair_features not in ("exact", "lut", "none"):
            raise ValueError(f"pair_features must be exact|lut|none, got {self.pair_features!r}")
        if self.num_heads <= 0:
            self.num_heads = max(1, self.D // 16)
        if self.D % self.num_heads:
            raise ValueError(f"D={self.D} not divisible by num_heads={self.num_heads}")
        if not self.hierarchy and len(self.depths) != 1:
            raise ValueError("a flat model (hierarchy=False) takes one entry in depths")
        n = self.seq_len
        self.level_sizes = [n]
        if self.hierarchy:
            if len(self.strides) != len(self.depths) - 1:
                raise ValueError("strides must have len(depths) - 1 entries")
            for s in self.strides:
                n //= s
                self.level_sizes.append(n)
        # every level but the top uses `ball`; the top is one ball (global), as in ErwinParTv2
        self.ball_sizes = [self.ball] * (len(self.level_sizes) - 1) + (
            [self.level_sizes[-1]] if self.hierarchy else [self.ball])
        for L, m in zip(self.level_sizes, self.ball_sizes):
            if L % m:
                raise ValueError(f"tile size {m} does not divide level size {L}")


# ------------------------------------------------------------------ pieces --
class SwiGLU(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.w1, self.w2, self.w3 = nn.Linear(dim, hidden), nn.Linear(dim, hidden), nn.Linear(hidden, dim)

    def forward(self, x):
        return self.w3(self.w2(x) * F.silu(self.w1(x)))


class PairNet(nn.Module):
    """ParT's pair MLP on (..., 4) features -> (..., heads), with masked
    (real-pair-only) normalisation of its input and hidden layers."""

    def __init__(self, hidden, heads):
        super().__init__()
        dims = [4, *hidden, heads]
        self.in_norm = make_norm("batchnorm", 4)
        # No bias: the BatchNorm right after each layer subtracts the mean, so a bias
        # would cancel exactly and never learn; the norm's own shift takes its place.
        self.lins = nn.ModuleList(nn.Linear(a, b, bias=False) for a, b in zip(dims[:-1], dims[1:]))
        self.norms = nn.ModuleList(make_norm("batchnorm", b) for b in dims[1:])

    def forward(self, f, valid, stats=None):
        """`stats` (default `valid`) selects the pairs the statistics come from and
        that carry their features; the rest of `valid` (a particle with itself)
        enters as a zero feature vector, so it gets one learned self-bias."""
        stats = valid if stats is None else stats
        h = self.in_norm(f, stats) * stats.unsqueeze(-1).to(f.dtype)
        last = len(self.lins) - 1
        for i, (lin, nrm) in enumerate(zip(self.lins, self.norms)):
            h = nrm(lin(h), stats)
            if i < last:                       # pre-activation output, as ParT
                h = F.gelu(h)
        return h * valid.unsqueeze(-1).to(h.dtype)


class LiteMSA(nn.Module):
    """Tile attention with the assembled bias, plus optional ball-summary keys."""

    def __init__(self, D, H, attn_dropout, summary):
        super().__init__()
        self.D, self.H, self.hd = D, H, D // H
        self.summary = summary
        self.attn_dropout = attn_dropout
        self.qkv = nn.Linear(D, 3 * D)
        self.proj = nn.Linear(D, D)
        self.pe_proj = nn.Linear(2, D)
        self.sigma = nn.Parameter(torch.zeros(1, 1, H, 1, 1))

    def forward(self, x, ctx, nb, m):
        """x (N, nb, m, D) normalised; ctx from ErwinParTLite._context."""
        D, H, hd = self.D, self.H, self.hd
        x = x + self.pe_proj(ctx["rel"])
        q, k, v = self.qkv(x).view(-1, nb, m, 3, H, hd).permute(3, 0, 1, 4, 2, 5)   # (N, nb, H, m, hd)
        K = m
        aq = getattr(self, "aq", None)              # fixed-point emulation, set by quant.quantize_model
        if self.summary:
            w = ctx["valid"].unsqueeze(-1).to(x.dtype)
            s = (x * w).sum(2) / w.sum(2).clamp(min=1.0)                            # (N, nb, D)
            if aq is not None:                      # through the quantized layer (its Q part unused)
                kv = self.qkv(s)[..., D:].reshape(-1, nb, 2, H, hd)
            else:
                kv = F.linear(s, self.qkv.weight[D:], self.qkv.bias[D:]).view(-1, nb, 2, H, hd)
            ks = kv[:, :, 0].permute(0, 2, 1, 3).unsqueeze(1).expand(-1, nb, H, nb, hd)
            vs = kv[:, :, 1].permute(0, 2, 1, 3).unsqueeze(1).expand(-1, nb, H, nb, hd)
            k = torch.cat([k, ks], dim=3)
            v = torch.cat([v, vs], dim=3)
            K = m + nb
        bias = ctx["u"] - F.softplus(self.sigma) * ctx["dist"]                        # (N, nb, H, m, K)
        if aq is not None:
            out = aq.attend(q.reshape(-1, H, m, hd), k.reshape(-1, H, K, hd), v.reshape(-1, H, K, hd),
                            bias.reshape(-1, H, m, K), ctx["key_valid"].reshape(-1, 1, 1, K),
                            self.attn_dropout, self.training)
            return self.proj(out.view(-1, nb, H, m, hd).permute(0, 1, 3, 2, 4).reshape(-1, nb, m, D))
        bias = bias.masked_fill(~ctx["key_valid"].unsqueeze(2).unsqueeze(3), NEG)
        out = F.scaled_dot_product_attention(
            q.reshape(-1, H, m, hd), k.reshape(-1, H, K, hd), v.reshape(-1, H, K, hd),
            attn_mask=bias.reshape(-1, H, m, K), dropout_p=self.attn_dropout if self.training else 0.0)
        return self.proj(out.view(-1, nb, H, m, hd).permute(0, 1, 3, 2, 4).reshape(-1, nb, m, D))


class LiteBlock(nn.Module):
    """ParT's residual structure (head and residual scales, pre/post norms), SwiGLU FFN."""

    def __init__(self, cfg: LiteConfig, summary: bool):
        super().__init__()
        D, H = cfg.D, cfg.num_heads
        self.H, self.hd = H, D // H
        self.pre_attn_norm = make_norm(cfg.norm, D)
        self.attn = LiteMSA(D, H, cfg.attn_dropout, summary)
        self.post_attn_norm = make_norm(cfg.norm, D)
        self.pre_ffn_norm = make_norm(cfg.norm, D)
        self.ffn = SwiGLU(D, D * cfg.ffn_ratio)
        self.dropout = nn.Dropout(cfg.dropout)
        self.act_dropout = nn.Dropout(cfg.dropout)
        self.c_attn = nn.Parameter(torch.ones(H))
        self.w_resid = nn.Parameter(torch.ones(D))

    def forward(self, x, ctx, nb, m):
        valid = ctx["valid"]
        h = self.attn(self.pre_attn_norm(x, valid), ctx, nb, m)
        h = (h.view(-1, nb, m, self.H, self.hd) * self.c_attn.view(self.H, 1)).reshape(-1, nb, m, self.H * self.hd)
        x = x + self.dropout(self.post_attn_norm(h, valid))
        h = self.dropout(self.act_dropout(self.ffn(self.pre_ffn_norm(x, valid))))
        x = x * self.w_resid + h
        out_q = getattr(self, "out_q", None)        # residual stream at fixed point (quant.py)
        if out_q is not None:
            x = out_q(x)
        return x * valid.unsqueeze(-1).to(x.dtype)


class LitePool(nn.Module):
    """Merge `stride` consecutive children: summed 4-momenta (a physical subjet),
    positions about the parent centre, occupancy; as ErwinParTv2's BallPooling."""

    def __init__(self, D, stride, norm):
        super().__init__()
        self.s = stride
        self.proj = nn.Linear(stride * D + stride * 2 + 1, D)
        self.norm = make_norm(norm, D)

    def forward(self, x, pos, p4, valid, n_parent):
        s = self.s
        D = x.shape[-1]
        w = valid.view(-1, n_parent, s, 1).to(x.dtype)
        xg = x.view(-1, n_parent, s, D) * w
        pg = pos.view(-1, n_parent, s, 2)
        centre = (pg * w).sum(2, keepdim=True) / w.sum(2, keepdim=True).clamp(min=1.0)
        rel = ((pg - centre) * w).reshape(-1, n_parent, s * 2)
        occ = w.mean(2)
        pv = valid.view(-1, n_parent, s).any(-1)
        out = self.norm(self.proj(torch.cat([xg.reshape(-1, n_parent, s * D), rel, occ], -1)), pv)
        return (out * pv.unsqueeze(-1).to(out.dtype), centre.squeeze(2),
                (p4.view(-1, n_parent, s, 4) * w).sum(2), pv)


class ClsBlock(nn.Module):
    """ParT's class-attention block: the class token attends to [itself; tokens]."""

    def __init__(self, cfg: LiteConfig):
        super().__init__()
        D = cfg.D
        self.H, self.hd = cfg.num_heads, D // cfg.num_heads
        self.norm1 = make_norm(cfg.norm, D)
        self.q, self.kv, self.proj = nn.Linear(D, D), nn.Linear(D, 2 * D), nn.Linear(D, D)
        self.norm2 = make_norm(cfg.norm, D)
        self.ffn = nn.Sequential(nn.Linear(D, D * cfg.ffn_ratio), nn.GELU(), nn.Linear(D * cfg.ffn_ratio, D))
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, cls, tokens, tok_valid):
        H, hd = self.H, self.hd
        u = torch.cat([cls, tokens], 1)                                  # (N, 1+T, D)
        uv = torch.cat([torch.ones_like(tok_valid[:, :1]), tok_valid], 1)
        un = self.norm1(u, uv)
        q = self.q(un[:, :1]).view(-1, 1, H, hd).transpose(1, 2)
        k, v = self.kv(un).view(-1, un.shape[1], 2, H, hd).permute(2, 0, 3, 1, 4)
        aq = getattr(self, "aq", None)
        if aq is not None:
            a = aq.attend(q, k, v, None, uv[:, None, None, :], 0.0, self.training)
        else:
            mask = torch.zeros_like(uv, dtype=u.dtype).masked_fill(~uv, NEG)[:, None, None, :]
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        cls = cls + self.dropout(self.proj(a.transpose(1, 2).reshape(-1, 1, H * hd)))
        return cls + self.dropout(self.ffn(self.norm2(cls, torch.ones_like(tok_valid[:, :1]))))


# ------------------------------------------------------------------- model --
class ErwinParTLite(nn.Module):
    def __init__(self, cfg: LiteConfig):
        super().__init__()
        self.cfg = cfg
        D, H = cfg.D, cfg.num_heads
        dims = [cfg.input_dim, D, cfg.embed_ratio * D, D]
        self.embed_norms = nn.ModuleList(make_norm(cfg.norm, d) for d in dims[:-1])
        self.embed_lins = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        n_lv = len(cfg.level_sizes)
        summary_at = [cfg.ball_summary != "off" and L // m > 1 for L, m in zip(cfg.level_sizes, cfg.ball_sizes)]
        self.summary_at = summary_at
        self.levels = nn.ModuleList(
            nn.ModuleList(LiteBlock(cfg, summary_at[i]) for _ in range(cfg.depths[i])) for i in range(n_lv))
        self.pools = nn.ModuleList(LitePool(D, s, cfg.norm) for s in (cfg.strides if cfg.hierarchy else ()))
        self.pair_nets = nn.ModuleList(PairNet(cfg.pair_hidden, H) for _ in range(n_lv))
        self.summary_nets = nn.ModuleList(
            PairNet(cfg.pair_hidden, H) if (summary_at[i] and cfg.ball_summary == "physics") else nn.Identity()
            for i in range(n_lv))
        self.register_buffer("log2_table", _log2_table(cfg.lut_bits), persistent=False)
        self.level_embed = nn.Parameter(torch.zeros(n_lv, D))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, D))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.cls_blocks = nn.ModuleList(ClsBlock(cfg) for _ in range(cfg.cls_layers))
        self.out_norm = make_norm(cfg.norm, D)
        self.fc = nn.Linear(D, cfg.num_classes)

    # ------------------------------------------------------------ geometry --
    def _pair_feats(self, pos_i, pos_j, p4_i, p4_j):
        """(..., 4) pair features between broadcast-compatible (..., 2)/(..., 4) inputs."""
        c = self.cfg
        if c.pair_features == "exact":
            return exact_features(p4_i, p4_j)
        pt_i, lnpt_i = lut_pt(p4_i, self.log2_table, c.lut_bits)
        pt_j, lnpt_j = lut_pt(p4_j, self.log2_table, c.lut_bits)
        return lut_features(pos_i, pos_j, lnpt_i, lnpt_j, pt_i, pt_j, self.log2_table, c.lut_bits, p4_i, p4_j)

    def _dist(self, pos_i, pos_j, radius2):
        """Distance normalised by the tile's own size: dR / r (exact) or dR^2 / r^2
        (lut mode: no square root at all). radius2 = mean squared distance to centre."""
        d = pos_i - pos_j
        dr2 = (d * d).sum(-1)
        if self.cfg.pair_features == "lut":
            out = dr2 / radius2.clamp(min=1e-6)
        else:
            out = dr2.clamp(min=1e-12).sqrt() / radius2.clamp(min=1e-6).sqrt()
        # A one-particle tile has radius 0, so its distances to other tiles' summaries
        # come out ~1e6. In float that only switches those keys off; in fixed point it
        # stretches the bias's range and costs every other value its resolution. A cap
        # (in the same units: dR^2/r^2 for lut, dR/r otherwise) saturates it, as hardware would.
        if self.cfg.dist_cap > 0:
            out = out.clamp(max=self.cfg.dist_cap)
        return out

    def _context(self, li, pos, p4, valid):
        """Tile-shaped geometry and the bias shared by every block at one level/ordering."""
        c = self.cfg
        L, m = c.level_sizes[li], c.ball_sizes[li]
        nb = L // m
        H = c.num_heads
        pb, qb, vb = pos.view(-1, nb, m, 2), p4.view(-1, nb, m, 4), valid.view(-1, nb, m)
        w = vb.unsqueeze(-1).to(pos.dtype)
        centre = (pb * w).sum(2, keepdim=True) / w.sum(2, keepdim=True).clamp(min=1.0)    # (N, nb, 1, 2)
        rel = (pb - centre) * w
        radius2 = ((rel * rel).sum(-1) * vb).sum(-1) / vb.sum(-1).clamp(min=1)            # (N, nb)
        pair_valid = vb.unsqueeze(-1) & vb.unsqueeze(-2)                                  # (N, nb, m, m)
        zl = torch.zeros_like(pb[..., 0])[:, :, None, :, None]                            # (N, nb, 1, m, 1)
        if c.pair_features == "none":
            u = zl.expand(-1, -1, H, m, m)
        else:
            # A particle paired with itself has dR = 0, so ln dR and ln kT sit at the
            # log floor (-18 exact, -9 lut) against ~-3 for real pairs. In a tile of 8
            # that is 1 pair in 8 (1 in 128 in ParT): left in, it dominates the pair
            # norm's statistics and squashes every real pair together. So the
            # statistics and the features come from distinct pairs only.
            distinct = pair_valid & ~torch.eye(m, dtype=torch.bool, device=pb.device)
            f = self._pair_feats(pb.unsqueeze(3), pb.unsqueeze(2), qb.unsqueeze(3), qb.unsqueeze(2))
            u = self.pair_nets[li](f, pair_valid, distinct).permute(0, 1, 4, 2, 3)        # (N, nb, H, m, m)
        if c.use_dist_bias:
            dist = self._dist(pb.unsqueeze(3), pb.unsqueeze(2), radius2[..., None, None])
            dist = (dist * pair_valid).unsqueeze(2)                                       # (N, nb, 1, m, m)
        else:
            dist = torch.zeros_like(u[:, :, :1])
        key_valid = vb                                                                     # (N, nb, m)
        if self.summary_at[li]:
            tile_valid = vb.any(-1)                                                        # (N, nb)
            tile_p4 = (qb * w).sum(2)                                                      # (N, nb, 4) subjet
            tile_pos = centre.squeeze(2)                                                   # (N, nb, 2)
            sv = vb.unsqueeze(-1) & tile_valid[:, None, None, :]                           # (N, nb, m, nb)
            if c.ball_summary == "physics" and c.pair_features != "none":
                fs = self._pair_feats(pb.unsqueeze(3), tile_pos[:, None, None], qb.unsqueeze(3),
                                      tile_p4[:, None, None])
                us = self.summary_nets[li](fs, sv).permute(0, 1, 4, 2, 3)                 # (N, nb, H, m, nb)
                ds = self._dist(pb.unsqueeze(3), tile_pos[:, None, None], radius2[..., None, None])
                ds = (ds * sv).unsqueeze(2)
            else:
                us = zl.expand(-1, -1, H, m, nb)
                ds = torch.zeros_like(us[:, :, :1])
            u = torch.cat([u, us], -1)
            dist = torch.cat([dist, ds], -1)
            key_valid = torch.cat([key_valid, tile_valid[:, None, :].expand(-1, nb, nb)], -1)
        return {"rel": rel, "valid": vb, "u": u, "dist": dist, "key_valid": key_valid}

    def _orders(self, pos, valid):
        """Permutation into the level's base tile order (or None) and the shifted one."""
        c = self.cfg
        if c.partition == "morton":
            return (morton_order(pos, valid, c.morton_bits),
                    morton_order(pos, valid, c.morton_bits, shift=True) if c.shift else None)
        if c.partition == "tree":
            from ..balltree_torch import build_ball_tree, rotate_positions
            return (build_ball_tree(pos, valid),
                    build_ball_tree(rotate_positions(pos, 45.0), valid) if c.shift else None)
        return None, None                                                                   # pt: keep the order

    # ------------------------------------------------------------- forward --
    def forward(self, x, v=None, mask=None, points=None):
        c = self.cfg
        valid = mask[:, 0].bool()
        feats, p4, pos = x.permute(0, 2, 1), v.permute(0, 2, 1), points.permute(0, 2, 1)
        pt = (p4[..., 0] * p4[..., 0] + p4[..., 1] * p4[..., 1]).sqrt()
        top = torch.argsort(torch.where(valid, pt, -torch.ones_like(pt)), dim=1, descending=True,
                            stable=True)[:, :c.seq_len]
        valid = torch.gather(valid, 1, top)
        w = valid.unsqueeze(-1).to(x.dtype)
        feats, p4, pos = gather_slots(feats, top) * w, gather_slots(p4, top) * w, gather_slots(pos, top) * w

        h = feats
        for i, (nrm, lin) in enumerate(zip(self.embed_norms, self.embed_lins)):
            h = F.gelu(lin(nrm(h, valid)))
        h = h * w

        tokens, tok_valid = [], []
        for li, blocks in enumerate(self.levels):
            L, m = c.level_sizes[li], c.ball_sizes[li]
            nb = L // m
            base, shifted = self._orders(pos, valid) if nb > 1 else (None, None)
            if c.partition == "morton" and li > 0:
                base = None          # pooled order is already Z-ordered: no re-sort in hardware
            if base is not None:
                h, pos, p4, valid = (gather_slots(h, base), gather_slots(pos, base),
                                     gather_slots(p4, base), torch.gather(valid, 1, base))
            ctx = {False: self._context(li, pos, p4, valid)}
            inv = None
            if shifted is not None:
                # shifted is an order of the ORIGINAL slots; re-express it relative to `base`
                rel = torch.argsort(base, dim=1) if base is not None else None   # slot -> place in base order
                sh = torch.gather(rel, 1, shifted) if rel is not None else shifted
                inv = torch.argsort(sh, dim=1)
                ctx[True] = self._context(li, gather_slots(pos, sh), gather_slots(p4, sh),
                                          torch.gather(valid, 1, sh))
            for bi, blk in enumerate(blocks):
                use_shift = inv is not None and bi % 2 == 1
                if use_shift:
                    h = gather_slots(h, sh)
                hv = blk(h.view(-1, nb, m, c.D), ctx[use_shift], nb, m).reshape(-1, L, c.D)
                h = gather_slots(hv, inv) if use_shift else hv
            tokens.append(h + self.level_embed[li])
            tok_valid.append(valid)
            if li < len(self.pools):
                h, pos, p4, valid = self.pools[li](h, pos, p4, valid, c.level_sizes[li + 1])

        toks, tv = torch.cat(tokens, 1), torch.cat(tok_valid, 1)
        cls = self.cls_token.expand(toks.shape[0], -1, -1)
        for blk in self.cls_blocks:
            cls = blk(cls, toks, tv)
        return self.fc(self.out_norm(cls, torch.ones_like(tv[:, :1]))[:, 0])
