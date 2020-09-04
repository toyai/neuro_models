"""
EfficientNet B0-7,
based from tensorflow.keras.applications.efficientnet
"""
from typing import Sequence, Union

from torch import nn

from utils.efficientnets import (
    Swish,
    blocks_params,
    compound_params,
    get_padding,
    load_weights,
    round_filters,
    round_repeats,
)


class ConvBN(nn.Module):
    """Convolution - BatchNorm - Swish Layers Block.
    It can be used to create Expansion stage, Depthwise Conv and Pointwise Conv.

    Return:
        A Tensor which size is (N, C, H, W)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Sequence],
        epsilon: float,
        momentum: float,
        stride: Union[int, Sequence] = 1,
        groups: int = 1,
    ):
        super().__init__()
        padding = get_padding(kernel_size, stride)
        layers = []

        # zero padding for the same size output
        if all(x == 0 for x in padding):
            layers.append(nn.Identity())
        else:
            layers.append(nn.ZeroPad2d(padding))

        layers.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                groups=groups,
                bias=False,
            )
        )
        layers.append(
            nn.BatchNorm2d(num_features=out_channels, eps=epsilon, momentum=momentum)
        )
        layers.append(Swish())
        self.conv_bn = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_bn(x)


class SqueezeExcite(nn.Module):
    """Squeeze and Excitation stage of MBConvBlock.

    Return:
        A Tensor which size is (N, C, H, W)
    """

    def __init__(
        self,
        out_channels: int,
        squeeze_channels: int,
        kernel_size: Union[int, Sequence],
        stride: Union[int, Sequence],
    ):
        super().__init__()
        layers = []
        layers.append(nn.AdaptiveAvgPool2d(1))

        padding = get_padding(kernel_size, stride)
        # zero padding for the same size output
        if all(x == 0 for x in padding):
            layers.append(nn.Identity())
        else:
            layers.append(nn.ZeroPad2d(padding))

        layers.append(
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=squeeze_channels,
                kernel_size=kernel_size,
            )
        )
        layers.append(Swish())

        # zero padding for the same size output
        if all(x == 0 for x in padding):
            layers.append(nn.Identity())
        else:
            layers.append(nn.ZeroPad2d(padding))

        layers.append(
            nn.Conv2d(
                in_channels=squeeze_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
            )
        )
        layers.append(nn.Sigmoid())
        self.se = nn.Sequential(*layers)

    def forward(self, x):
        return x * self.se(x)


class MBConvBlock(nn.Module):
    """Mobile Inverted Residual Block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Sequence],
        stride: Union[int, Sequence],
        epsilon: float,
        momentum: float,
        expand_ratio: int,
        se_ratio: float,
        dropout_p_skip: float = 0.2,
    ):
        super().__init__()
        self.is_residual = in_channels == out_channels and stride == 1
        self.dropout_p_skip = dropout_p_skip  # dropout probability at skip connection

        # intermediate channels between blocks
        inter_channels = in_channels * expand_ratio
        # squeeze out_channels for SqueezeExcite
        squeeze_channels = max(1, int(in_channels * se_ratio))
        layers = []

        if in_channels != inter_channels:
            # expansion conv stage
            layers.append(
                ConvBN(
                    in_channels=in_channels,
                    out_channels=inter_channels,
                    kernel_size=1,
                    epsilon=epsilon,
                    momentum=momentum,
                )
            )

        layers.extend(
            [
                # depthwise conv stage
                ConvBN(
                    in_channels=inter_channels,
                    out_channels=inter_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    epsilon=epsilon,
                    momentum=momentum,
                    groups=inter_channels,
                ),
                # squeeze and excitation stage
                SqueezeExcite(
                    out_channels=inter_channels,
                    squeeze_channels=squeeze_channels,
                    kernel_size=1,
                    stride=1,
                ),
                # pointwise conv stage
                ConvBN(
                    in_channels=inter_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    epsilon=epsilon,
                    momentum=momentum,
                ),
            ]
        )
        self.mb_conv_blocks = nn.Sequential(*layers)

    def forward(self, x):
        if self.is_residual:
            return x + nn.Dropout(self.dropout_p_skip)(self.mb_conv_blocks(x))
        return self.mb_conv_blocks(x)


class EfficientNet(nn.Module):
    """EfficientNet Base Model. 🏡"""

    def __init__(
        self,
        name: str,
        num_classes: int = 1000,
        include_fc: bool = True,
        divisor: int = 8,
    ):
        super().__init__()
        self.include_fc = include_fc
        self.num_classes = num_classes
        blocks_args = blocks_params()
        width, depth, image_size, dropout_p, momentum, epsilon = compound_params(name)
        out_channels = round_filters(filters=32, divisor=divisor, width=width)

        # conv stem
        layers = [ConvBN(3, out_channels, 3, epsilon, 1 - momentum, stride=2)]
        in_channels = out_channels

        for i, args in enumerate(blocks_args):
            out_channels = round_filters(args["out_channels"], divisor, width)
            repeats = round_repeats(args["repeats"], depth)
            for i in range(repeats):
                stride = args["stride"] if i == 0 else 1
                # mobile inverted conv block
                layers.append(
                    MBConvBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=args["kernel_size"],
                        stride=stride,
                        epsilon=epsilon,
                        momentum=1 - momentum,
                        expand_ratio=args["expand_ratio"],
                        se_ratio=args["se_ratio"],
                    )
                )
                in_channels = out_channels

        final_out_channels = round_filters(1280, divisor, width)
        # conv head
        layers.append(ConvBN(in_channels, final_out_channels, 1, epsilon, 1 - momentum))
        self.convs = nn.Sequential(*layers)

        # final classification layers
        if include_fc and num_classes == 1000:
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(1),
                nn.Dropout(dropout_p),
                nn.Linear(final_out_channels, num_classes),
                Swish(),
            )

    def forward(self, x):
        x = self.convs(x)
        x = self.classifier(x) if self.include_fc and self.num_classes == 1000 else x

        return x

    @classmethod
    def from_pretrained(cls, name: str, include_fc: bool = True):
        """Load weights from https://github.com/lukemelas/EfficientNet-PyTorch."""
        model = cls(name, include_fc=include_fc)
        load_weights(model, name, include_fc)
        return model
