"""Loss functions for 3D cardiac segmentation."""

from __future__ import annotations

import torch
import torch.nn as nn
from monai.losses import DiceCELoss, DiceLoss, FocalLoss


class CombinedSegmentationLoss(nn.Module):
    """Combined Dice + Cross-Entropy loss for segmentation.

    Uses MONAI's DiceCELoss with softmax activation for multi-class output.

    Args:
        include_background: Whether to include background in loss computation.
        softmax: Apply softmax to predictions.
        lambda_dice: Weight for Dice component.
        lambda_ce: Weight for CE component.
    """

    def __init__(
        self,
        include_background: bool = False,
        softmax: bool = True,
        lambda_dice: float = 1.0,
        lambda_ce: float = 1.0,
    ) -> None:
        super().__init__()
        self.loss_fn = DiceCELoss(
            include_background=include_background,
            softmax=softmax,
            to_onehot_y=True,
            lambda_dice=lambda_dice,
            lambda_ce=lambda_ce,
        )

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute combined loss.

        Args:
            predictions: Model logits (B, C, H, W, D).
            targets: Ground truth labels (B, 1, H, W, D).

        Returns:
            Scalar loss tensor.
        """
        return self.loss_fn(predictions, targets)


class DiceOnlyLoss(nn.Module):
    """Pure Dice loss for segmentation.

    Args:
        include_background: Whether to include background.
        softmax: Apply softmax to predictions.
    """

    def __init__(
        self,
        include_background: bool = False,
        softmax: bool = True,
    ) -> None:
        super().__init__()
        self.loss_fn = DiceLoss(
            include_background=include_background,
            softmax=softmax,
            to_onehot_y=True,
        )

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Dice loss.

        Args:
            predictions: Model logits (B, C, H, W, D).
            targets: Ground truth labels (B, 1, H, W, D).

        Returns:
            Scalar loss tensor.
        """
        return self.loss_fn(predictions, targets)
