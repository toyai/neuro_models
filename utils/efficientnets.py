"""
Utilities functions for EfficientNets
"""
import math
from typing import Sequence, Union

import torch
from torch import nn


def blocks_params():
    """Default Blocks Parameters."""
    return [
        {
            "kernel_size": 3,
            "repeats": 1,
            "in_channels": 32,
            "out_channels": 16,
            "expand_ratio": 1,
            "id_skip": True,
            "stride": 1,
            "se_ratio": 0.25,
        },
        {
            "kernel_size": 3,
            "repeats": 2,
            "in_channels": 16,
            "out_channels": 24,
            "expand_ratio": 6,
            "id_skip": True,
            "stride": 2,
            "se_ratio": 0.25,
        },
        {
            "kernel_size": 5,
            "repeats": 2,
            "in_channels": 24,
            "out_channels": 40,
            "expand_ratio": 6,
            "id_skip": True,
            "stride": 2,
            "se_ratio": 0.25,
        },
        {
            "kernel_size": 3,
            "repeats": 3,
            "in_channels": 40,
            "out_channels": 80,
            "expand_ratio": 6,
            "id_skip": True,
            "stride": 2,
            "se_ratio": 0.25,
        },
        {
            "kernel_size": 5,
            "repeats": 3,
            "in_channels": 80,
            "out_channels": 112,
            "expand_ratio": 6,
            "id_skip": True,
            "stride": 1,
            "se_ratio": 0.25,
        },
        {
            "kernel_size": 5,
            "repeats": 4,
            "in_channels": 112,
            "out_channels": 192,
            "expand_ratio": 6,
            "id_skip": True,
            "stride": 2,
            "se_ratio": 0.25,
        },
        {
            "kernel_size": 3,
            "repeats": 1,
            "in_channels": 192,
            "out_channels": 320,
            "expand_ratio": 6,
            "id_skip": True,
            "stride": 1,
            "se_ratio": 0.25,
        },
    ]


def compound_params(name: str = "efficientnet-b0"):
    """Compound parameters for Efficient Net Models with some extra parameters

    Args:
        name: Efficient Net Model Name. Default: ``efficientnet-b0``.

    Returns:
        A tuple of width, depth, resolution, dropout, momentum, epsilon.
    """

    compound_params_dict = {
        # width, depth, resolution, dropout, momentum, epsilon
        "efficientnet-b0": (1.0, 1.0, 224, 0.2, 0.99, 1e-3),
        "efficientnet-b1": (1.0, 1.1, 240, 0.2, 0.99, 1e-3),
        "efficientnet-b2": (1.1, 1.2, 260, 0.3, 0.99, 1e-3),
        "efficientnet-b3": (1.2, 1.4, 300, 0.3, 0.99, 1e-3),
        "efficientnet-b4": (1.4, 1.8, 380, 0.4, 0.99, 1e-3),
        "efficientnet-b5": (1.6, 2.2, 456, 0.4, 0.99, 1e-3),
        "efficientnet-b6": (1.8, 2.6, 528, 0.5, 0.99, 1e-3),
        "efficientnet-b7": (2.0, 3.1, 600, 0.5, 0.99, 1e-3),
        # "efficientnet-b8": (2.2, 3.6, 672, 0.5, 0.99, 1e-3),
        # "efficientnet-l2": (4.3, 5.3, 800, 0.5, 0.99, 1e-3),
    }
    return compound_params_dict[name]


def get_padding(kernel_size: Union[int, Sequence], stride: Union[int, Sequence]):
    """Calculates Padding Size from kernel_size and stride."""
    kh, kw = kernel_size if isinstance(kernel_size, Sequence) else (kernel_size,) * 2
    sh, sw = stride if isinstance(stride, Sequence) else (stride,) * 2
    pad_h = max(kh - sh, 0)
    pad_w = max(kw - sw, 0)
    return (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)


def round_repeats(repeats: int, depth: float):
    """Round number of repeats based on depth multiplier.
    https://github.com/tensorflow/tensorflow/blob/v2.3.0/tensorflow/python/keras/applications/efficientnet.py#L310
    """
    return int(math.ceil(depth * repeats))


def round_filters(filters: int, divisor: int, width: float):
    """
    Round number of filters based on width multiplier.
    https://github.com/tensorflow/tensorflow/blob/v2.3.0/tensorflow/python/keras/applications/efficientnet.py#L301
    """
    filters *= width
    new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)


class Swish(nn.Module):
    """Swish Activation.

    Return:
        input * sigmoid(input)
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)
