"""I/O utilities for NIfTI file handling."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np


def load_nifti(path: str | Path) -> tuple[np.ndarray, np.ndarray, tuple[float, ...]]:
    """Load a NIfTI file.

    Args:
        path: Path to .nii or .nii.gz file.

    Returns:
        Tuple of (volume, affine, spacing).
    """
    nii = nib.load(str(path))
    volume = np.asarray(nii.dataobj, dtype=np.float32)
    affine = nii.affine
    spacing = tuple(float(s) for s in nii.header.get_zooms()[:3])
    return volume, affine, spacing


def save_nifti(
    volume: np.ndarray,
    affine: np.ndarray,
    path: str | Path,
) -> None:
    """Save a volume as NIfTI file.

    Args:
        volume: 3D numpy array.
        affine: 4x4 affine matrix.
        path: Output path (.nii or .nii.gz).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nii = nib.Nifti1Image(volume.astype(np.float32), affine)
    nib.save(nii, str(path))


def save_mesh(
    mesh: "trimesh.Trimesh",
    path: str | Path,
    file_type: str = "stl",
) -> None:
    """Save a trimesh object to file.

    Args:
        mesh: trimesh.Trimesh object.
        path: Output path.
        file_type: Export format (stl, ply, obj).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(path), file_type=file_type)
