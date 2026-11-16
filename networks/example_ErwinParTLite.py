"""Weaver plugin for ErwinParTLite, the fixed-size FPGA-friendly ErwinParTv2.

    weaver --network-config networks/example_ErwinParTLite.py -o D 32 -o ffn_ratio 2 ...

Every LiteConfig field can be set with `-o key value` (tuples as "(16, 16)").
Defaults reproduce ErwinParTv2's size (D = 128, 2.21M parameters) with Morton
tiles; see networks/erwin_part/fpga/lite_model.py.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from erwin_part.fpga import ErwinParTLite, LiteConfig  # noqa: E402


class ErwinParTLiteWrapper(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.mod = ErwinParTLite(LiteConfig(**kwargs))

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"mod.cls_token", "mod.level_embed"}

    def forward(self, points, features, lorentz_vectors, mask):
        return self.mod(features, v=lorentz_vectors, mask=mask, points=points)


def get_model(data_config, **kwargs):
    kwargs.pop("use_amp", None)
    kwargs.pop("for_inference", None)
    cfg = dict(input_dim=len(data_config.input_dicts["pf_features"]),
               num_classes=len(data_config.label_value))
    cfg.update(kwargs)
    model = ErwinParTLiteWrapper(**cfg)
    model_info = {
        "input_names": list(data_config.input_names),
        "input_shapes": {k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        "output_names": ["softmax"],
        "dynamic_axes": {**{k: {0: "N", 2: "n_" + k.split("_")[0]} for k in data_config.input_names},
                         **{"softmax": {0: "N"}}},
    }
    return model, model_info


def get_loss(data_config, **kwargs):
    return torch.nn.CrossEntropyLoss()
