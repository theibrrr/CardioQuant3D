"""Segmentation evaluation metrics: Dice Score and Hausdorff Distance."""

from __future__ import annotations

import numpy as np
import torch
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from scipy.ndimage import distance_transform_edt


def compute_dice_score(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
) -> float:
    """Compute Dice similarity coefficient between two binary masks.

    Args:
        prediction: Binary prediction mask.
        ground_truth: Binary ground truth mask.

    Returns:
        Dice score in [0, 1].
    """
    prediction = prediction.astype(bool)
    ground_truth = ground_truth.astype(bool)

    intersection = np.logical_and(prediction, ground_truth).sum()
    total = prediction.sum() + ground_truth.sum()

    if total == 0:
        return 1.0 if intersection == 0 else 0.0

    return float(2.0 * intersection / total)


def compute_hausdorff_distance(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    voxel_spacing: tuple[float, ...] = (1.0, 1.0, 1.0),
    percentile: float = 95.0,
) -> float:
    """Compute Hausdorff distance between two binary masks.

    Uses the specified percentile (default 95th) for robustness.

    Args:
        prediction: Binary prediction mask.
        ground_truth: Binary ground truth mask.
        voxel_spacing: Physical voxel dimensions (mm).
        percentile: Percentile for robust HD computation.

    Returns:
        Hausdorff distance in mm.
    """
    prediction = prediction.astype(bool)
    ground_truth = ground_truth.astype(bool)

    if not prediction.any() and not ground_truth.any():
        return 0.0
    if not prediction.any() or not ground_truth.any():
        return float("inf")

    # Compute distance transforms
    pred_border = prediction ^ _erode(prediction)
    gt_border = ground_truth ^ _erode(ground_truth)

    # If borders are empty (single-voxel structures), use the masks directly
    if not pred_border.any():
        pred_border = prediction
    if not gt_border.any():
        gt_border = ground_truth

    # Distance from prediction boundary to ground truth
    dt_gt = distance_transform_edt(~ground_truth, sampling=voxel_spacing)
    dt_pred = distance_transform_edt(~prediction, sampling=voxel_spacing)

    distances_pred_to_gt = dt_gt[pred_border]
    distances_gt_to_pred = dt_pred[gt_border]

    all_distances = np.concatenate([distances_pred_to_gt, distances_gt_to_pred])

    return float(np.percentile(all_distances, percentile))


def _erode(mask: np.ndarray) -> np.ndarray:
    """Simple binary erosion for border extraction.

    Args:
        mask: Binary 3D mask.

    Returns:
        Eroded binary mask.
    """
    from scipy.ndimage import binary_erosion

    return binary_erosion(mask, iterations=1).astype(bool)


def compute_dice_batch(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    include_background: bool = False,
) -> float:
    """Compute mean Dice score for a batch using MONAI.

    Args:
        predictions: Predicted one-hot tensor (B, C, H, W, D).
        labels: Ground truth one-hot tensor (B, C, H, W, D).
        include_background: Whether to include background class.

    Returns:
        Mean Dice score across batch and classes.
    """
    metric = DiceMetric(include_background=include_background, reduction="mean")
    metric(y_pred=predictions, y=labels)
    result = metric.aggregate().item()
    metric.reset()
    return float(result)


def compute_hausdorff_batch(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    include_background: bool = False,
    percentile: float = 95.0,
) -> float:
    """Compute mean Hausdorff distance for a batch using MONAI.

    Args:
        predictions: Predicted one-hot tensor (B, C, H, W, D).
        labels: Ground truth one-hot tensor (B, C, H, W, D).
        include_background: Whether to include background class.
        percentile: Percentile for robust HD.

    Returns:
        Mean Hausdorff distance across batch.
    """
    metric = HausdorffDistanceMetric(
        include_background=include_background,
        percentile=percentile,
        reduction="mean",
    )
    metric(y_pred=predictions, y=labels)
    result = metric.aggregate().item()
    metric.reset()
    return float(result)
