import numpy as np
import torch
from torch import nn


class ConvLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layer(x)


class DownSamplingLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.layer = nn.Sequential(
            nn.MaxPool2d(2),
            ConvLayer(in_channels, out_channels)
        )

    def forward(self, x):
        return self.layer(x)


class UpSamplingLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mode: str = 'transposed'):
        """
        :param mode: 'transposed' for transposed convolution, or 'nearest', 'linear', 'bilinear', 'bicubic', 'trilinear'
        """
        super().__init__()

        if mode == 'transposed':
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        elif mode in {'nearest', 'linear', 'bilinear', 'bicubic', 'trilinear'}:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode=mode),
                nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
            )
        else:
            raise ValueError(f'Unsupported mode [{mode}], supported modes are "transposed", "nearest", "linear", "bilinear", "bicubic" or "trilinear"')

        self.conv = ConvLayer(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, depth: int = 3, start_filters: int = 16, up_mode: str = 'transposed'):
        super().__init__()

        self.inc = ConvLayer(in_channels, start_filters)

        # Contracting path
        self.down = nn.ModuleList([DownSamplingLayer(start_filters * 2 ** i, start_filters * 2 ** (i + 1))
                                   for i in range(depth)])

        # Expansive path
        self.up = nn.ModuleList([UpSamplingLayer(start_filters * 2 ** (i + 1), start_filters * 2 ** i, up_mode)
                                 for i in range(depth - 1, -1, -1)])

        self.outc = nn.Conv2d(start_filters, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.inc(x)

        outputs = []

        for module in self.down:
            outputs.append(x)
            x = module(x)

        for module, output in zip(self.up, reversed(outputs)):
            x = module(x, output)

        return self.outc(x)


