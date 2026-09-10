"""Hierarchical ball-tree attention for the Particle Transformer (ErwinParTv2).

Import layering is deliberate and load-bearing:

  geometry, balltree_torch   depend only on torch
  pair_features, ball_attention, hierarchy, model   also need weaver-core

The tree and the geometry stay weaver-free so they can be verified against
Erwin's compiled `balltree` package, which lives in a different interpreter from
weaver-core (see tests/s2_tree_parity.py). Importing this package must therefore
NOT drag weaver in eagerly - the weaver-backed names are resolved on first
access via PEP 562 instead.
"""
from .balltree_torch import ball_view, build_ball_tree, build_levels, rotate_positions
from .geometry import (delta_r, masked_centroid, relative_positions,
                       sort_truncate_pad, take_slots, wrap_phi)

_LAZY = {
    "BallBlock": "ball_attention", "BallMSA": "ball_attention",
    "assemble_bias": "ball_attention", "ball_distance_bias": "ball_attention",
    "BallPooling": "hierarchy", "BallUnpooling": "hierarchy",
    "PairFeatureEmbed": "pair_features", "within_ball_pair_features": "pair_features",
    "ErwinParticleTransformerV2": "model",
}

__all__ = sorted(_LAZY) + [
    "ball_view", "build_ball_tree", "build_levels", "delta_r", "masked_centroid",
    "relative_positions", "rotate_positions", "sort_truncate_pad", "take_slots",
    "wrap_phi",
]


def __getattr__(name):
    """Resolve the weaver-dependent names on first use, not at import time."""
    if name in _LAZY:
        import importlib
        mod = importlib.import_module(f".{_LAZY[name]}", __name__)
        value = getattr(mod, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
