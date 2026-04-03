"""Unit tests for 3D U-Net model."""

from __future__ import annotations

import pytest
import torch

from cardioquant3d.models.unet3d import UNet3D, build_model


class TestUNet3D:
    """Tests for UNet3D model."""

    def test_output_shape(self) -> None:
        """Output shape matches input spatial dims with out_channels."""
        model = build_model(
            in_channels=1,
            out_channels=2,
            channels=[16, 32, 64],
            strides=[2, 2],
            num_res_units=1,
            dropout=0.0,
        )
        x = torch.randn(1, 1, 32, 32, 16)
        y = model(x)
        assert y.shape == (1, 2, 32, 32, 16)

    def test_different_configs(self) -> None:
        """Model can be created with different channel configs."""
        model = build_model(
            in_channels=1,
            out_channels=4,
            channels=[8, 16, 32, 64],
            strides=[2, 2, 2],
            num_res_units=2,
        )
        x = torch.randn(1, 1, 64, 64, 32)
        y = model(x)
        assert y.shape == (1, 4, 64, 64, 32)

    def test_batch_size(self) -> None:
        """Model handles batch size > 1."""
        model = build_model(
            channels=[8, 16],
            strides=[2],
            num_res_units=1,
        )
        x = torch.randn(2, 1, 32, 32, 16)
        y = model(x)
        assert y.shape[0] == 2

    def test_gradient_flow(self) -> None:
        """Gradients flow through the model."""
        model = build_model(
            channels=[8, 16],
            strides=[2],
            num_res_units=1,
        )
        x = torch.randn(1, 1, 32, 32, 16, requires_grad=True)
        y = model(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
