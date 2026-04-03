"""3D U-Net model built on MONAI for cardiac segmentation."""

from __future__ import annotations

import torch
import torch.nn as nn
from monai.networks.nets import UNet


class UNet3D(nn.Module):
    """Configurable 3D U-Net for volumetric segmentation.

    Wraps MONAI's UNet with a clean interface and optional softmax output.

    Args:
        in_channels: Number of input channels (1 for MRI).
        out_channels: Number of output classes (2 for binary: background + LV).
        channels: Feature map sizes per encoder level.
        strides: Downsampling strides between encoder levels.
        num_res_units: Number of residual units per level.
        dropout: Dropout probability.
        norm: Normalization type ('batch', 'instance', 'group').
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
        channels: tuple[int, ...] = (32, 64, 128, 256),
        strides: tuple[int, ...] = (2, 2, 2),
        num_res_units: int = 2,
        dropout: float = 0.2,
        norm: str = "batch",
    ) -> None:
        super().__init__()

        norm_map = {
            "batch": ("batch", {"affine": True}),
            "instance": ("instance", {"affine": False}),
            "group": ("group", {"num_groups": 8}),
        }

        self.unet = UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units,
            dropout=dropout,
            norm=norm_map.get(norm, norm),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W, D).

        Returns:
            Logits tensor of shape (B, out_channels, H, W, D).
        """
        return self.unet(x)


def build_model(
    in_channels: int = 1,
    out_channels: int = 2,
    channels: list[int] | None = None,
    strides: list[int] | None = None,
    num_res_units: int = 2,
    dropout: float = 0.2,
    norm: str = "batch",
) -> UNet3D:
    """Factory function to build a 3D U-Net from config parameters.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output classes.
        channels: Feature map sizes per encoder level.
        strides: Downsampling strides.
        num_res_units: Number of residual units.
        dropout: Dropout probability.
        norm: Normalization type.

    Returns:
        Configured UNet3D instance.
    """
    if channels is None:
        channels = [32, 64, 128, 256]
    if strides is None:
        strides = [2, 2, 2]

    return UNet3D(
        in_channels=in_channels,
        out_channels=out_channels,
        channels=tuple(channels),
        strides=tuple(strides),
        num_res_units=num_res_units,
        dropout=dropout,
        norm=norm,
    )
