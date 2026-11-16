"""Normalisation options for ErwinParTLite. Every norm takes (x, valid).

'layernorm'  nn.LayerNorm - the baseline; needs a mean, a variance and a
             division per token at inference.
'batchnorm'  per-channel statistics over REAL particles only (padding excluded,
             so zero-filled slots cannot drag the statistics). At inference it is
             a fixed affine map and folds into the neighbouring Linear.
'affine'     learned per-channel scale and shift, no statistics at all.
'none'       identity.
"""
import torch
import torch.nn as nn


class _LayerNorm(nn.LayerNorm):
    def forward(self, x, valid=None):
        return super().forward(x)


class _Affine(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x, valid=None):
        return x * self.weight + self.bias


class _Identity(nn.Module):
    def forward(self, x, valid=None):
        return x


class MaskedBatchNorm(nn.Module):
    """BatchNorm over the last dim, statistics from rows where `valid` is True."""

    def __init__(self, dim, momentum=0.1, eps=1e-5):
        super().__init__()
        self.momentum, self.eps = momentum, eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.register_buffer("running_mean", torch.zeros(dim))
        self.register_buffer("running_var", torch.ones(dim))

    def forward(self, x, valid):
        if self.training:
            w = valid.reshape(-1, 1).to(x.dtype)
            flat = x.reshape(-1, x.shape[-1])
            n = w.sum().clamp(min=1.0)
            mean = (flat * w).sum(0) / n
            var = (((flat - mean) ** 2) * w).sum(0) / n
            with torch.no_grad():
                self.running_mean.lerp_(mean, self.momentum)
                self.running_var.lerp_(var * n / (n - 1).clamp(min=1.0), self.momentum)
        else:
            mean, var = self.running_mean, self.running_var
        return (x - mean) * torch.rsqrt(var + self.eps) * self.weight + self.bias


def make_norm(kind: str, dim: int) -> nn.Module:
    if kind == "layernorm":
        return _LayerNorm(dim)
    if kind == "batchnorm":
        return MaskedBatchNorm(dim)
    if kind == "affine":
        return _Affine(dim)
    if kind == "none":
        return _Identity()
    raise ValueError(f"norm must be layernorm|batchnorm|affine|none, got {kind!r}")
