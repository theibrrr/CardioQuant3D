"""Geometric measurements from 3D cardiac meshes and masks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh
from sklearn.decomposition import PCA


@dataclass
class GeometricMeasurements:
    """Geometric measurements of a cardiac structure.

    Attributes:
        volume_ml: Volume in milliliters (1 ml = 1000 mm³).
        surface_area_mm2: Surface area in mm².
        long_axis_mm: PCA-based long-axis length in mm.
        sphericity_index: Sphericity index (1.0 = perfect sphere).
    """

    volume_ml: float
    surface_area_mm2: float
    long_axis_mm: float
    sphericity_index: float


def compute_voxel_volume(
    mask: np.ndarray,
    voxel_spacing: tuple[float, ...],
) -> float:
    """Compute volume from voxel count and spacing.

    Args:
        mask: Binary 3D mask.
        voxel_spacing: Physical voxel dimensions (mm).

    Returns:
        Volume in mm³.
    """
    voxel_vol = float(np.prod(voxel_spacing))
    return float(mask.astype(bool).sum()) * voxel_vol


def compute_long_axis_length(
    mask: np.ndarray,
    voxel_spacing: tuple[float, ...],
) -> float:
    """Estimate long-axis length using PCA on foreground voxel coordinates.

    The long axis is defined as the extent along the first principal component.

    Args:
        mask: Binary 3D mask.
        voxel_spacing: Physical voxel dimensions (mm).

    Returns:
        Long-axis length in mm.
    """
    coords = np.argwhere(mask.astype(bool))
    if len(coords) < 2:
        return 0.0

    # Scale to physical coordinates
    physical_coords = coords.astype(np.float64) * np.array(voxel_spacing)

    pca = PCA(n_components=1)
    projections = pca.fit_transform(physical_coords)

    long_axis = float(projections.max() - projections.min())
    return long_axis


def compute_surface_area(mesh: trimesh.Trimesh) -> float:
    """Extract surface area from mesh.

    Args:
        mesh: trimesh.Trimesh object in physical coordinates.

    Returns:
        Surface area in mm².
    """
    return float(mesh.area)


def compute_sphericity_index(
    volume_mm3: float,
    surface_area_mm2: float,
) -> float:
    """Compute sphericity index.

    Formula: ψ = (π^(1/3) * (6V)^(2/3)) / A

    A perfect sphere has ψ = 1.0.

    Args:
        volume_mm3: Volume in mm³.
        surface_area_mm2: Surface area in mm².

    Returns:
        Sphericity index (dimensionless).
    """
    if surface_area_mm2 < 1e-8 or volume_mm3 < 1e-8:
        return 0.0

    numerator = (np.pi ** (1.0 / 3.0)) * ((6.0 * volume_mm3) ** (2.0 / 3.0))
    return float(numerator / surface_area_mm2)


def compute_geometric_measurements(
    mesh: trimesh.Trimesh,
    mask: np.ndarray,
    voxel_spacing: tuple[float, ...],
) -> GeometricMeasurements:
    """Compute all geometric measurements for a cardiac structure.

    Uses voxel-counting for volume (more reliable than mesh volume),
    mesh for surface area, PCA for long axis, and derived sphericity.

    Args:
        mesh: trimesh.Trimesh in physical coordinates.
        mask: Binary 3D mask.
        voxel_spacing: Physical voxel dimensions (mm).

    Returns:
        GeometricMeasurements dataclass.
    """
    volume_mm3 = compute_voxel_volume(mask, voxel_spacing)
    volume_ml = volume_mm3 / 1000.0  # 1 ml = 1000 mm³

    surface_area_mm2 = compute_surface_area(mesh)
    long_axis_mm = compute_long_axis_length(mask, voxel_spacing)
    sphericity = compute_sphericity_index(volume_mm3, surface_area_mm2)

    return GeometricMeasurements(
        volume_ml=volume_ml,
        surface_area_mm2=surface_area_mm2,
        long_axis_mm=long_axis_mm,
        sphericity_index=sphericity,
    )
