"""3D mesh generation from binary segmentation masks using Marching Cubes."""

from __future__ import annotations

import numpy as np
import trimesh
from skimage.measure import marching_cubes


def create_mesh_from_mask(
    mask: np.ndarray,
    voxel_spacing: tuple[float, ...] = (1.0, 1.0, 1.0),
    level: float = 0.5,
    step_size: int = 1,
) -> trimesh.Trimesh:
    """Convert a binary 3D mask to a triangle mesh using Marching Cubes.

    Voxel spacing is applied to scale vertices into physical (mm) coordinates.

    Args:
        mask: Binary 3D numpy array.
        voxel_spacing: Physical voxel dimensions (sx, sy, sz) in mm.
        level: Iso-surface level for marching cubes.
        step_size: Step size for marching cubes (higher = faster but coarser).

    Returns:
        trimesh.Trimesh object in physical coordinates.

    Raises:
        ValueError: If the mask has no foreground voxels.
    """
    mask = mask.astype(np.float32)

    if mask.sum() == 0:
        raise ValueError("Cannot create mesh from an empty mask (no foreground voxels).")

    # Run Marching Cubes
    vertices, faces, normals, _ = marching_cubes(
        mask,
        level=level,
        spacing=voxel_spacing,
        step_size=step_size,
    )

    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_normals=normals,
        process=True,
    )

    return mesh


def mesh_surface_area(mesh: trimesh.Trimesh) -> float:
    """Compute the surface area of a mesh.

    Args:
        mesh: trimesh.Trimesh object.

    Returns:
        Surface area in mm².
    """
    return float(mesh.area)


def mesh_volume(mesh: trimesh.Trimesh) -> float:
    """Compute the enclosed volume of a watertight mesh.

    Falls back to voxel-based volume if the mesh is not watertight.

    Args:
        mesh: trimesh.Trimesh object.

    Returns:
        Volume in mm³.
    """
    if mesh.is_watertight:
        return float(abs(mesh.volume))

    # Fallback: use convex hull
    try:
        return float(abs(mesh.convex_hull.volume))
    except Exception:
        return 0.0
