"""Preprocessing utilities for 3D cardiac MRI volumes."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom


def load_nifti(path: str | Path) -> tuple[np.ndarray, np.ndarray, tuple[float, ...]]:
    """Load a NIfTI file and return volume, affine matrix, and voxel spacing.

    Args:
        path: Path to .nii or .nii.gz file.

    Returns:
        Tuple of (volume ndarray, affine 4x4 matrix, voxel spacing tuple).
    """
    nii = nib.load(str(path))
    volume = np.asarray(nii.dataobj, dtype=np.float32)
    affine = nii.affine
    spacing = tuple(float(s) for s in nii.header.get_zooms()[:3])
    return volume, affine, spacing


def normalize_intensity(
    volume: np.ndarray,
    clip_percentile: tuple[float, float] = (0.5, 99.5),
) -> np.ndarray:
    """Clip and z-score normalize a volume.

    Args:
        volume: Input 3D array.
        clip_percentile: Lower and upper percentiles for intensity clipping.

    Returns:
        Normalized volume with zero mean and unit variance.
    """
    low = np.percentile(volume, clip_percentile[0])
    high = np.percentile(volume, clip_percentile[1])
    volume = np.clip(volume, low, high)

    mean = volume.mean()
    std = volume.std()
    if std > 0:
        volume = (volume - mean) / std
    else:
        volume = volume - mean

    return volume


def binarize_label(
    label: np.ndarray,
    target_label: int = 3,
) -> np.ndarray:
    """Convert multi-label mask to binary mask for a specific structure.

    Args:
        label: Multi-label segmentation mask.
        target_label: Integer label for the target structure.

    Returns:
        Binary mask (0 or 1) as float32.
    """
    return (label == target_label).astype(np.float32)


def resample_volume(
    volume: np.ndarray,
    target_shape: tuple[int, ...],
    order: int = 1,
) -> np.ndarray:
    """Resample volume to a target spatial shape using scipy zoom.

    Args:
        volume: Input 3D array.
        target_shape: Desired output shape.
        order: Interpolation order (0=nearest, 1=linear, 3=cubic).

    Returns:
        Resampled volume.
    """
    zoom_factors = tuple(t / s for t, s in zip(target_shape, volume.shape))
    return zoom(volume, zoom_factors, order=order).astype(np.float32)
