"""CNN trunk blocks for PyRat."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Pre-activation residual block.

    Architecture: BN -> ReLU -> Conv -> BN -> ReLU -> Conv + skip
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out: torch.Tensor = F.relu(self.bn1(x))
        out = self.conv1(out)
        out = F.relu(self.bn2(out))
        out = self.conv2(out)
        return out + identity


class GPoolResBlock(nn.Module):
    """Residual block with global pooling branch (KataGo-style, simplified).

    Two parallel paths combined with skip connection:
    - Regular path: BN -> ReLU -> Conv3x3 -> BN -> ReLU -> Conv3x3
    - Pool path: BN -> ReLU -> Conv1x1 -> mean+max pool -> Linear -> broadcast
    - Output: regular + pool + skip

    All ops are ONNX-safe (mean/max pool, broadcast via unsqueeze).
    """

    def __init__(self, channels: int, gpool_channels: int = 32) -> None:
        super().__init__()

        # Regular path
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)

        # Pool path
        self.pool_bn = nn.BatchNorm2d(channels)
        self.pool_conv = nn.Conv2d(channels, gpool_channels, kernel_size=1, bias=False)
        # mean + max pool → 2 * gpool_channels features
        self.pool_linear = nn.Linear(2 * gpool_channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        # Regular path
        regular: torch.Tensor = F.relu(self.bn1(x))
        regular = self.conv1(regular)
        regular = F.relu(self.bn2(regular))
        regular = self.conv2(regular)

        # Pool path
        pool = F.relu(self.pool_bn(x))
        pool = self.pool_conv(pool)  # (B, gpool_channels, H, W)
        # Global mean and max pooling over spatial dims
        pool_mean = pool.mean(dim=(2, 3))  # (B, gpool_channels)
        pool_max = pool.amax(dim=(2, 3))  # (B, gpool_channels)
        pool_cat = torch.cat([pool_mean, pool_max], dim=1)  # (B, 2*gpool_channels)
        pool_out = self.pool_linear(pool_cat)  # (B, channels)
        # Broadcast to spatial dims (ONNX-safe)
        pool_out = pool_out.unsqueeze(2).unsqueeze(3)  # (B, channels, 1, 1)

        out: torch.Tensor = regular + pool_out + identity
        return out
