"""Stable axis normalization for the existing SABL-style S/A head.

For finite logits this is the real-valued sigmoid/sum function. Floating-point
results and training trajectories are not promised to be bitwise identical.
"""
import torch
from torch.nn import functional as F


def normalized_sigmoid_axis(logits, axis):
    if logits.ndim != 4 or axis not in (2, 3) or logits.shape[axis] == 0:
        raise ValueError('SABL axis normalization shape/axis')
    if logits.dtype not in (torch.float32, torch.float64):
        raise ValueError('SABL axis normalization requires FP32 or FP64')
    if not bool(torch.isfinite(logits).all()):
        raise ValueError('nonfinite SABL normalization logits')
    weights = F.softmax(F.logsigmoid(logits), dim=axis)
    if not bool(torch.isfinite(weights).all()):
        raise ValueError('nonfinite SABL normalized weights')
    return weights
