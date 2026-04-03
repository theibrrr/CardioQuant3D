"""Unit tests for loss functions."""

from __future__ import annotations

import pytest
import torch

from cardioquant3d.training.losses import CombinedSegmentationLoss, DiceOnlyLoss


class TestCombinedLoss:
    """Tests for combined Dice + CE loss."""

    def test_loss_is_scalar(self) -> None:
        """Loss output is a scalar tensor."""
        loss_fn = CombinedSegmentationLoss()
        pred = torch.randn(2, 2, 16, 16, 8)
        target = torch.randint(0, 2, (2, 1, 16, 16, 8)).float()
        loss = loss_fn(pred, target)
        assert loss.dim() == 0

    def test_loss_is_positive(self) -> None:
        """Loss is non-negative."""
        loss_fn = CombinedSegmentationLoss()
        pred = torch.randn(2, 2, 16, 16, 8)
        target = torch.randint(0, 2, (2, 1, 16, 16, 8)).float()
        loss = loss_fn(pred, target)
        assert loss.item() >= 0.0

    def test_gradient_flow(self) -> None:
        """Gradients flow through the loss."""
        loss_fn = CombinedSegmentationLoss()
        pred = torch.randn(2, 2, 16, 16, 8, requires_grad=True)
        target = torch.randint(0, 2, (2, 1, 16, 16, 8)).float()
        loss = loss_fn(pred, target)
        loss.backward()
        assert pred.grad is not None


class TestDiceOnlyLoss:
    """Tests for Dice-only loss."""

    def test_loss_is_scalar(self) -> None:
        """Loss output is a scalar tensor."""
        loss_fn = DiceOnlyLoss()
        pred = torch.randn(2, 2, 16, 16, 8)
        target = torch.randint(0, 2, (2, 1, 16, 16, 8)).float()
        loss = loss_fn(pred, target)
        assert loss.dim() == 0
