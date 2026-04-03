"""Unit tests for segmentation metrics."""

from __future__ import annotations

import numpy as np
import pytest

from cardioquant3d.evaluation.metrics import (
    compute_dice_score,
    compute_hausdorff_distance,
)


class TestDiceScore:
    """Tests for Dice score computation."""

    def test_perfect_overlap(self) -> None:
        """Dice = 1.0 for identical masks."""
        mask = np.zeros((32, 32, 16), dtype=np.float32)
        mask[10:20, 10:20, 5:12] = 1.0
        assert compute_dice_score(mask, mask) == pytest.approx(1.0)

    def test_no_overlap(self) -> None:
        """Dice = 0.0 for non-overlapping masks."""
        pred = np.zeros((32, 32, 16), dtype=np.float32)
        pred[0:5, 0:5, 0:5] = 1.0
        gt = np.zeros((32, 32, 16), dtype=np.float32)
        gt[20:25, 20:25, 10:15] = 1.0
        assert compute_dice_score(pred, gt) == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        """Dice is between 0 and 1 for partial overlap."""
        pred = np.zeros((32, 32, 16), dtype=np.float32)
        pred[5:15, 5:15, 3:10] = 1.0
        gt = np.zeros((32, 32, 16), dtype=np.float32)
        gt[10:20, 5:15, 3:10] = 1.0
        dice = compute_dice_score(pred, gt)
        assert 0.0 < dice < 1.0

    def test_both_empty(self) -> None:
        """Dice = 1.0 when both masks are empty."""
        pred = np.zeros((32, 32, 16), dtype=np.float32)
        gt = np.zeros((32, 32, 16), dtype=np.float32)
        assert compute_dice_score(pred, gt) == pytest.approx(1.0)

    def test_symmetry(self) -> None:
        """Dice is symmetric."""
        pred = np.zeros((32, 32, 16), dtype=np.float32)
        pred[5:15, 5:15, 3:10] = 1.0
        gt = np.zeros((32, 32, 16), dtype=np.float32)
        gt[10:20, 5:15, 3:10] = 1.0
        assert compute_dice_score(pred, gt) == pytest.approx(
            compute_dice_score(gt, pred)
        )


class TestHausdorffDistance:
    """Tests for Hausdorff distance computation."""

    def test_identical_masks(self) -> None:
        """HD = 0 for identical masks."""
        mask = np.zeros((32, 32, 16), dtype=np.float32)
        mask[10:20, 10:20, 5:12] = 1.0
        hd = compute_hausdorff_distance(mask, mask, voxel_spacing=(1.0, 1.0, 1.0))
        assert hd == pytest.approx(0.0, abs=0.5)

    def test_separated_masks(self) -> None:
        """HD > 0 for separated masks."""
        pred = np.zeros((32, 32, 16), dtype=np.float32)
        pred[0:5, 0:5, 0:5] = 1.0
        gt = np.zeros((32, 32, 16), dtype=np.float32)
        gt[25:30, 25:30, 10:15] = 1.0
        hd = compute_hausdorff_distance(pred, gt, voxel_spacing=(1.0, 1.0, 1.0))
        assert hd > 0

    def test_empty_pred(self) -> None:
        """HD = inf when prediction is empty."""
        pred = np.zeros((32, 32, 16), dtype=np.float32)
        gt = np.zeros((32, 32, 16), dtype=np.float32)
        gt[10:20, 10:20, 5:12] = 1.0
        hd = compute_hausdorff_distance(pred, gt, voxel_spacing=(1.0, 1.0, 1.0))
        assert hd == float("inf")

    def test_respects_spacing(self) -> None:
        """HD scales with voxel spacing."""
        pred = np.zeros((32, 32, 16), dtype=np.float32)
        pred[5:10, 5:10, 3:8] = 1.0
        gt = np.zeros((32, 32, 16), dtype=np.float32)
        gt[15:20, 5:10, 3:8] = 1.0

        hd1 = compute_hausdorff_distance(pred, gt, voxel_spacing=(1.0, 1.0, 1.0))
        hd2 = compute_hausdorff_distance(pred, gt, voxel_spacing=(2.0, 2.0, 2.0))
        assert hd2 > hd1
