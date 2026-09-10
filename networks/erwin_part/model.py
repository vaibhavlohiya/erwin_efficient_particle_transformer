"""ErwinParTv2 - Particle Transformer with hierarchical ball-tree attention.

    pf_features + pf_points + pf_vectors + mask
      -> SequenceTrimmer, Embed                       (ParT, unchanged)
      -> fixed L=64 slot layout, pt-sorted            (geometry)
      -> ball tree per level, canonical + rotated     (balltree_torch)
      -> level 0: 64 nodes,  8 balls of 8   } BallBlocks, alternating rotation,
         level 1: 32 subjets, 4 balls of 8  } each carrying ParT's U on
         level 2: 16 subjets, 1 ball of 16  } within-ball pairs only
      -> class attention over every level at once     (ParT's CaiT blocks)
      -> 10 logits

Three design choices worth stating plainly, because they diverge from one or
other parent architecture:

**The bottleneck is a single ball.** With Erwin's stock `ball_sizes=[8,8,8]` the
16 bottleneck nodes still split into two balls, so a two-prong topology such as
H->bb can land its prongs in different balls at *every* level and never interact.
Setting the top-level ball size to the remaining node count makes the coarsest
layer globally connected, which is what makes the hierarchy work at all.

**The readout is multi-scale, not a mean.** A jet tag rides on a handful of hard
constituents, so mean-pooling the coarse latent (as Erwin's own JetClass wrapper
does) dilutes the signal. ParT's class-attention head is kept and pointed at the
union of all three levels - 64 + 32 + 16 = 112 tokens, fewer than ParT's own 128,
so the readout costs less than ParT's while seeing constituents, subjets and the
whole jet at once.

**Widths are constant.** Every level runs at `embed_dim`, so ParT's block
hyperparameters transfer unchanged and the class-attention blocks can read all
levels without projection.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from weaver.nn.model.ParticleTransformer import Block, Embed, SequenceTrimmer, trunc_normal_

from .ball_attention import BallBlock, assemble_bias, ball_distance_bias
from .balltree_torch import build_levels
from .geometry import relative_positions, sort_truncate_pad, take_slots
from .hierarchy import BallPooling
from .pair_features import PairFeatureEmbed


class ErwinParticleTransformerV2(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int | None = None,
        embed_dims: list[int] = [128, 512, 128],
        num_heads: int = 8,
        ball_sizes: list[int] = [8, 8, 16],
        strides: list[int] = [2, 2],
        depths: list[int] = [2, 2, 2],
        num_cls_layers: int = 2,
        pair_embed_dims: list[int] = [64, 64],
        seq_len: int = 64,
        rotate: float = 45.0,
        tree_on_cpu: bool = True,
        use_pair_bias: bool = True,
        use_dist_bias: bool = True,
        readout: str = "multi",
        tree_space: str = "etaphi",
        ffn_ratio: int = 4,
        block_params: dict | None = None,
        cls_block_params: dict | None = None,
        fc_params: list = [],
        activation: str = "gelu",
        trim: bool = True,
        for_inference: bool = False,
        use_amp: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if len(ball_sizes) != len(depths) or len(strides) != len(ball_sizes) - 1:
            raise ValueError("need len(ball_sizes) == len(depths) == len(strides) + 1")

        self.seq_len = seq_len
        self.ball_sizes = ball_sizes
        self.strides = strides
        self.rotate = rotate
        self.tree_on_cpu = tree_on_cpu
        self.use_pair_bias = use_pair_bias      # ParT's U (ablation)
        self.use_dist_bias = use_dist_bias      # Erwin Eq. 10 (ablation)
        if readout not in ("multi", "fine", "coarse"):
            raise ValueError("readout must be multi|fine|coarse")
        self.readout = readout
        if tree_space not in ("etaphi", "logpolar"):
            raise ValueError("tree_space must be etaphi|logpolar")
        self.tree_space = tree_space
        self.for_inference = for_inference
        self.use_amp = use_amp
        self.num_heads = num_heads

        self.trimmer = SequenceTrimmer(enabled=trim and not for_inference)
        self.embed = Embed(input_dim, embed_dims, activation=activation) \
            if embed_dims else nn.Identity()
        dim = embed_dims[-1] if embed_dims else input_dim
        self.embed_dim = dim

        bp = dict(embed_dim=dim, num_heads=num_heads, ffn_ratio=ffn_ratio,
                  dropout=0.1, attn_dropout=0.1, activation_dropout=0.1)
        bp.update(block_params or {})

        # One pair MLP per level: the within-ball pair statistics move by several
        # units between levels (ln m^2: -1.96 -> +1.42 -> +4.34), so they cannot
        # share an input normalisation, and the biases they must produce differ.
        # Not instantiated when the bias is ablated away: dead parameters would
        # still be weight-decayed by AdamW and would misstate the model size.
        self.pair_embeds = nn.ModuleList([
            PairFeatureEmbed(list(pair_embed_dims) + [num_heads], level=i)
            for i in range(len(ball_sizes))]) if use_pair_bias else None

        self.levels = nn.ModuleList([
            nn.ModuleList([BallBlock(**bp) for _ in range(d)]) for d in depths])
        self.pools = nn.ModuleList([BallPooling(dim, s) for s in strides])

        cbp = dict(embed_dim=dim, num_heads=num_heads, ffn_ratio=ffn_ratio,
                   dropout=0, attn_dropout=0, activation_dropout=0)
        cbp.update(cls_block_params or {})
        self.cls_blocks = nn.ModuleList([Block(**cbp) for _ in range(num_cls_layers)])
        self.level_embed = nn.Parameter(torch.zeros(len(ball_sizes), dim))
        self.norm = nn.LayerNorm(dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.level_embed, std=0.02)

        if fc_params is not None and num_classes is not None:
            fcs, in_dim = [], dim
            for out_dim, drop in fc_params:
                fcs.append(nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(),
                                         nn.Dropout(drop)))
                in_dim = out_dim
            fcs.append(nn.Linear(in_dim, num_classes))
            self.fc = nn.Sequential(*fcs)
        else:
            self.fc = None

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"cls_token", "level_embed"}

    # ---------------------------------------------------------------- helpers --
    @staticmethod
    def _gather(t: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
        return torch.gather(t, 1, perm.unsqueeze(-1).expand(-1, -1, t.shape[-1]))

    def _level_context(self, li, pos, p4, valid, order):
        """Ball-shaped pos/valid and the shared attention bias for one ordering."""
        bs = self.ball_sizes[li]
        N, L = valid.shape
        if order is not None:
            pos = self._gather(pos, order)
            p4 = self._gather(p4, order)
            valid = torch.gather(valid, 1, order)
        nb = L // bs
        pb = pos.view(N, nb, bs, pos.shape[-1])
        vb = valid.view(N, nb, bs)
        qb = p4.view(N, nb, bs, 4)
        if self.use_pair_bias:
            u = self.pair_embeds[li](qb)                          # (N, nb, H, m, m)
        else:
            u = torch.zeros(N, nb, self.num_heads, bs, bs,
                            device=p4.device, dtype=p4.dtype)
        dist = (ball_distance_bias(pb, vb) if self.use_dist_bias
                else torch.zeros(N, nb, 1, bs, bs, device=p4.device, dtype=p4.dtype))
        return pb, vb, u, dist

    def _tree_coords(self, pos):
        """Coordinates the tree is *partitioned* in (attention geometry is unchanged).

        'logpolar' warps each point radially by ln(1 + dR/eps0)/dR, which expands
        the collinear core where QCD piles up soft radiation and compresses the
        periphery. It stays Cartesian, so there is no phi seam to split across -
        the failure mode a literal (ln dR, phi) parametrisation would introduce.
        The distance bias and the relative-position embedding continue to use the
        true (delta_eta, delta_phi); only the partition changes.
        """
        if self.tree_space == "etaphi":
            return pos
        eps0 = 0.01
        r = pos.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return pos * (torch.log1p(r / eps0) / r)

    def _build_tree(self, posL, vL):
        """Build the ball tree, on CPU when that is faster than the model's device.

        The tree is a chain of small sequential sorts and gathers - almost no
        arithmetic, one kernel launch after another. On MPS that launch overhead
        dominates: measured on a 128-jet batch, building the tree on-device costs
        124.8 ms of a 210.3 ms forward pass (59%), against 6.9 ms on CPU. Moving
        it off-device is a ~2x forward speedup and changes nothing numerically -
        the tree is integer index math under no_grad.

        Positions are fixed inputs, so this is also precomputable in dataloader
        workers; doing so removes the cost from the training loop entirely.
        """
        dev = posL.device
        posT = self._tree_coords(posL)
        if not (self.tree_on_cpu and dev.type == "mps"):
            return build_levels(posT, vL, self.ball_sizes, self.strides,
                                rotate=self.rotate)
        levels = build_levels(posT.cpu(), vL.cpu(), self.ball_sizes, self.strides,
                              rotate=self.rotate)
        return [{k: (t.to(dev, non_blocking=True) if torch.is_tensor(t) else t)
                 for k, t in lv.items()} for lv in levels]

    # ---------------------------------------------------------------- forward --
    def forward(self, x, v=None, mask=None, points=None, uu=None, uu_idx=None):
        """x (N, C, P); v (N, 4, P) [px,py,pz,E]; mask (N, 1, P); points (N, 2, P)."""
        with torch.no_grad():
            # SequenceTrimmer randomly PERMUTES and truncates while training.
            # `points` must ride along through the same permutation or positions
            # silently decouple from features - a bug that would only ever appear
            # in training mode. Carrying it as extra channels of x guarantees the
            # identical gather.
            n_ch = x.shape[1]
            if points is not None:
                x = torch.cat([x, points[:, :2].to(x.dtype)], dim=1)
            x, v, mask, _ = self.trimmer(x, v, mask)
            if points is not None:
                x, points = x[:, :n_ch], x[:, n_ch:]
            valid = mask.squeeze(1).bool()                        # (N, P)

        with torch.amp.autocast("cuda", enabled=self.use_amp):
            h = self.embed(x)                                     # (P, N, C)
            h = h.permute(1, 0, 2)                                # (N, P, C)

            with torch.no_grad():
                p4 = v.permute(0, 2, 1).float()                   # (N, P, 4)
                if points is not None:
                    # pf_points ships (delta_eta, delta_phi) already wrapped and
                    # in JetClass's eta sign convention - the authoritative
                    # geometry. Recomputing from p4 is the fallback.
                    pos = points.permute(0, 2, 1).float()
                else:
                    pos = relative_positions(p4, valid)

                pt = torch.hypot(p4[..., 0], p4[..., 1])
                idx, vL = sort_truncate_pad(valid, pt, self.seq_len)
                posL = take_slots(pos, idx) * vL.unsqueeze(-1)
                p4L = take_slots(p4, idx) * vL.unsqueeze(-1)
                levels = self._build_tree(posL, vL)

            h = take_slots(h, idx) * vL.unsqueeze(-1).to(h.dtype)

            tokens, masks = [], []
            cur_pos, cur_p4, cur_valid = posL, p4L, vL

            for li, (blocks, lv) in enumerate(zip(self.levels, levels)):
                perm, bs = lv["perm"], self.ball_sizes[li]
                h = self._gather(h, perm)
                cur_pos = self._gather(cur_pos, perm)
                cur_p4 = self._gather(cur_p4, perm)
                cur_valid = torch.gather(cur_valid, 1, perm)

                # Built once per (level, ordering) and reused by every block here.
                ctx = {False: self._level_context(li, cur_pos, cur_p4, cur_valid, None)}
                if lv["rot_rel"] is not None:
                    ctx[True] = self._level_context(li, cur_pos, cur_p4, cur_valid,
                                                    lv["rot_rel"])

                N, L, C = h.shape
                nb = L // bs
                for bi, blk in enumerate(blocks):
                    rot = bool(bi % 2) and lv["rot_rel"] is not None
                    pb, vb, u, dist = ctx[rot]
                    hh = self._gather(h, lv["rot_rel"]) if rot else h
                    bias = assemble_bias(u, vb, dist, blk.attn.sigma)
                    hh = blk(hh.view(N, nb, bs, C), pb, vb, bias).view(N, L, C)
                    h = self._gather(hh, lv["rot_rel_inv"]) if rot else hh

                tokens.append(h + self.level_embed[li])
                masks.append(cur_valid)

                if li < len(self.pools):
                    h, cur_pos, cur_p4, cur_valid = self.pools[li](
                        h, cur_pos, cur_p4, cur_valid)

            # Multi-scale class attention: one query over constituents + subjets.
            if self.readout == "fine":
                tokens, masks = tokens[:1], masks[:1]     # constituents only
            elif self.readout == "coarse":
                tokens, masks = tokens[-1:], masks[-1:]   # bottleneck only
            seq = torch.cat(tokens, dim=1).permute(1, 0, 2)       # (S, N, C)
            padding_mask = ~torch.cat(masks, dim=1)               # (N, S)
            cls = self.cls_token.expand(1, seq.shape[1], -1)
            for blk in self.cls_blocks:
                cls = blk(seq, x_cls=cls, padding_mask=padding_mask)

            out = self.norm(cls).squeeze(0)
            if self.fc is None:
                return out
            out = self.fc(out)
            return torch.softmax(out, dim=1) if self.for_inference else out
