"""
DenseNet 121, 161, 169, 201
"""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

compound_params_dict = {
    # name: growthrate, block_config, init_features
    "densenet121": (32, (6, 12, 24, 16), 64),
    "densenet161": (48, (6, 12, 36, 24), 96),
    "densenet169": (32, (6, 12, 32, 32), 64),
    "densenet201": (32, (6, 12, 48, 32), 64)
}


class DenseNet(nn.Module):
    def __init__(
        self,
        name: str,
        num_classes: int = 80,
        bn_size: int = 4,
        drop_rate: int = 0
    ):
        super().__init__()
        growthrate, block_config, num_init_features = compound_params_dict[name]
        # 1st Conv
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features,
                                kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('bn0', nn.BatchNorm2d(num_init_features)),
            ('act0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        ]))
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            dense_block = DenseBlock(
                num_layers=num_layers,
                in_channels=num_features,
                bn_size=bn_size,
                growthrate=growthrate,
                drop_rate=drop_rate
            )
            self.features.add_module(f'denseblock-{i+1}', dense_block)
            num_features = num_features + num_layers * growthrate
            if i != len(block_config) - 1:
                transition = TransitionBlock(in_channels=num_features,
                                             out_channels=num_features // 2)
                self.features.add_module(f'transition{i+1}', transition)
                num_features = num_features // 2
        # Final BatchNorm
        self.features.add_module('bn5', nn.BatchNorm2d(num_features))
        # Linear Layer
        self.classifer = nn.Linear(num_features, num_classes)
        self.weight_init()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifer(out)
        return out


class BottleNeck(nn.Module):
    def __init__(
        self,
        in_channels: int,
        growthrate: float,
        bn_size: int,
        drop_rate: float
    ):
        super().__init__()
        # 1x1 Convolution
        self.add_module('bn1', nn.BatchNorm2d(in_channels))
        self.add_module('act1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv2d(in_channels,
                                           bn_size * growthrate,
                                           kernel_size=1,
                                           stride=1,
                                           bias=False))
        # 3x3 Convolution
        self.add_module('bn2', nn.BatchNorm2d(bn_size * growthrate))
        self.add_module('act2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv2d(bn_size * growthrate,
                                           growthrate,
                                           kernel_size=3,
                                           stride=1,
                                           padding=1,
                                           bias=False))
        self.drop_out = float(drop_rate)

    def forward(self, inputs):
        out = torch.cat(inputs, 1)
        out = self.conv1(self.act1(self.bn1(out)))
        out = self.conv2(self.act2(self.bn2(out)))
        if self.drop_out > 0:
            out = F.dropout(out, p=self.drop_out)
        return out


class TransitionBlock(nn.Sequential):
    """
        A building block of Transition Block

        Return:
            Output tensor for the block

    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int
    ):
        super().__init__()
        self.add_module('bn', nn.BatchNorm2d(in_channels))
        self.add_module('act', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(in_channels, out_channels,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseBlock(nn.ModuleDict):
    def __init__(
        self,
        num_layers: int,
        in_channels: int,
        bn_size: int,
        growthrate: float,
        drop_rate: float
    ):
        super().__init__()
        for i in range(num_layers):
            layer = BottleNeck(
                in_channels + i * growthrate,
                growthrate=growthrate,
                bn_size=bn_size,
                drop_rate=drop_rate
            )
            self.add_module(f'bottleneck-{i+1}', layer)

    def forward(self, init_features):
        features = [init_features]
        for _, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)
