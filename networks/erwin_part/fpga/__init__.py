"""ErwinParTLite: ErwinParTv2's hierarchy in a fixed-size, FPGA-friendly form.

See lite_model.py. torch-only except the 'exact' pair features, which import
weaver's pairwise_lv_fts lazily, so the 'lut' path runs without weaver.
"""
from .lite_model import ErwinParTLite, LiteConfig

__all__ = ["ErwinParTLite", "LiteConfig"]
