"""
Utilities functions for EfficientNets
"""
import logging
import math
from copy import deepcopy
from typing import Sequence, Union

import torch
from torch import nn

log = logging.getLogger(__name__)


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
    return (
        pad_w - pad_w // 2,
        pad_w - pad_w // 2,
        pad_h - pad_h // 2,
        pad_h - pad_h // 2,
    )


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


def load_weights(model: nn.Module, name: str, include_fc: bool):
    """
    Apache License © Luke Melas-Kyriazi

    Version 2.0, 2004

    Load weights from https://github.com/lukemelas/EfficientNet-PyTorch.
    """
    url = "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/"
    ckpt = {
        "efficientnet-b0": "efficientnet-b0-355c32eb.pth",
        "efficientnet-b1": "efficientnet-b1-f1951068.pth",
        "efficientnet-b2": "efficientnet-b2-8bb594d6.pth",
        "efficientnet-b3": "efficientnet-b3-5fb5a3c3.pth",
        "efficientnet-b4": "efficientnet-b4-6ed6700e.pth",
        "efficientnet-b5": "efficientnet-b5-b6417697.pth",
        "efficientnet-b6": "efficientnet-b6-c76e70fd.pth",
        "efficientnet-b7": "efficientnet-b7-dcc49843.pth",
    }
    state_dict = torch.hub.load_state_dict_from_url(url + ckpt[name], check_hash=True)
    keys = iter(model.state_dict().keys())
    tmp = deepcopy(state_dict)

    for i, tmp_key in enumerate(tmp.keys()):
        key = next(keys)
        if key.split(".")[-1] == tmp_key.split(".")[-1]:
            state_dict[key] = state_dict.pop(tmp_key)

    if include_fc:
        msg = model.load_state_dict(state_dict)
        assert (
            not msg.missing_keys
        ), f"Missing keys while loading pretrained weights: {msg.missing_keys}"
    else:
        state_dict.pop("classifier.3.weight")
        state_dict.pop("classifier.3.bias")
        msg = model.load_state_dict(state_dict)
        assert set(["classifier.3.weight", "classifier.3.bias"]) == set(
            msg.missing_keys
        ), f"Missing keys while loading pretrained weights: {msg.missing_keys}"

    assert (
        not msg.unexpected_keys
    ), f"Unexpected keys while loading pretrained weights: {msg.unexpected_keys}"
    log.info("Loaded pretrained weights from %s", url + ckpt[name])


class Swish(nn.Module):
    """Swish Activation.

    Return:
        input * sigmoid(input)
    """

    def forward(self, x):
        return x * torch.sigmoid(x)
