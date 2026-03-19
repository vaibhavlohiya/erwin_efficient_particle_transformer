"""
Quick smoke-test for the Erwin block integration in EfficientParticleTransformer.

Run from the repo root with the weaver conda env:
    conda run -n weaver python efficient_particle_transformer/test_erwin_block.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from particle_transformer.networks.EfficientParticleTransformer import (
    EfficientParticleTransformer,
    ParticleErwinBlock,
    ErwinTransformerBlock,
    _compute_positions,
)

# ── helpers ──────────────────────────────────────────────────────────────────
def make_batch(N=4, P=64, input_dim=17, seed=0):
    """Return (x, v, mask) with a random number of real particles per jet."""
    torch.manual_seed(seed)
    x    = torch.randn(N, input_dim, P)           # (N, C, P)
    v    = torch.randn(N, 4, P)                   # (N, 4, P) [px,py,pz,E]
    mask = torch.ones(N, 1, P, dtype=torch.bool)  # start fully unmasked
    # randomly mask out the last few particles in each jet
    for n in range(N):
        cutoff = torch.randint(P // 2, P, (1,)).item()
        mask[n, 0, cutoff:] = False
    return x, v, mask


def run(label, model, x, v, mask):
    model.eval()
    with torch.no_grad():
        out = model(x, v=v, mask=mask)
    print(f"[{label}] output shape: {out.shape}  ✓")


# ── test 1: ParticleErwinBlock in isolation ───────────────────────────────────
def test_particle_erwin_block():
    N, P, C = 4, 64, 128
    ball_size = 16
    num_heads = 8
    block = ParticleErwinBlock(C, num_heads, ball_size=ball_size)
    block.eval()

    x    = torch.randn(P, N, C)                # (P, N, C)
    pos  = torch.randn(N, P, 2)                # (N, P, 2) - eta, phi
    pmask = torch.zeros(N, P, dtype=torch.bool)
    pmask[:, 48:] = True                        # last 16 slots are padded

    with torch.no_grad():
        out = block(x, pos, padding_mask=pmask)

    assert out.shape == (P, N, C), f"Bad shape: {out.shape}"
    print(f"[ParticleErwinBlock] output shape: {out.shape}  ✓")


# ── test 2: _compute_positions ────────────────────────────────────────────────
def test_compute_positions():
    N, P = 4, 64
    v = torch.randn(N, 4, P)
    pos = _compute_positions(v)
    assert pos.shape == (N, P, 2), f"Bad shape: {pos.shape}"
    print(f"[_compute_positions] output shape: {pos.shape}  ✓")


# ── test 3: full EfficientParticleTransformer with Erwin blocks ───────────────
def test_full_model_erwin():
    model = EfficientParticleTransformer(
        input_dim=17,
        num_classes=10,
        embed_dims=[64, 256, 64],
        num_heads=4,
        num_layers=4,
        num_cls_layers=2,
        block_params={'attn_type': 'erwin', 'ball_size': 16, 'ffn_ratio': 4},
        cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0},
        fc_params=[],
        activation='gelu',
        trim=True,
    )
    print(f"[ErwinModel] parameters: {sum(p.numel() for p in model.parameters()):,}")

    x, v, mask = make_batch(N=4, P=64, input_dim=17)
    run("ErwinModel forward", model, x, v, mask)

    # verify pair_embed is None
    assert model.pair_embed is None, "pair_embed should be None for Erwin blocks"
    assert model.use_erwin_blocks, "use_erwin_blocks should be True"
    print("[ErwinModel] pair_embed is None (as expected)  ✓")


# ── test 4: original LinBlock path still works ────────────────────────────────
def test_full_model_linformer():
    model = EfficientParticleTransformer(
        input_dim=17,
        num_classes=10,
        embed_dims=[64, 256, 64],
        num_heads=4,
        num_layers=4,
        num_cls_layers=2,
        block_params={'attn_type': 'linformer', 'compressed': 4, 'max_seq_len': 128},
        cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0},
        fc_params=[],
        activation='gelu',
        trim=True,
    )
    x, v, mask = make_batch(N=4, P=64, input_dim=17)
    run("LinformerModel forward", model, x, v, mask)
    assert not model.use_erwin_blocks, "use_erwin_blocks should be False for Linformer"
    print("[LinformerModel] original path still works  ✓")


if __name__ == '__main__':
    print("=" * 60)
    print("Testing Erwin block integration")
    print("=" * 60)
    test_compute_positions()
    test_particle_erwin_block()
    test_full_model_erwin()
    test_full_model_linformer()
    print("=" * 60)
    print("All tests passed!")
