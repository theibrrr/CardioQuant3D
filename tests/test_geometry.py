"""Unit tests for geometry module."""

from __future__ import annotations

import numpy as np
import pytest

from cardioquant3d.geometry.measurements import (
    compute_long_axis_length,
    compute_sphericity_index,
    compute_voxel_volume,
)
from cardioquant3d.geometry.mesh import create_mesh_from_mask, mesh_surface_area, mesh_volume


class TestMeshCreation:
    """Tests for mesh generation from binary masks."""

    def test_cube_mesh(self) -> None:
        """Mesh can be created from a cubic mask."""
        mask = np.zeros((32, 32, 32), dtype=np.float32)
        mask[10:20, 10:20, 10:20] = 1.0
        mesh = create_mesh_from_mask(mask, voxel_spacing=(1.0, 1.0, 1.0))
        assert mesh is not None
        assert len(mesh.vertices) > 0
        assert len(mesh.faces) > 0

    def test_empty_mask_raises(self) -> None:
        """Empty mask raises ValueError."""
        mask = np.zeros((32, 32, 32), dtype=np.float32)
        with pytest.raises(ValueError, match="empty mask"):
            create_mesh_from_mask(mask)

    def test_spacing_affects_mesh(self) -> None:
        """Different spacings produce different mesh scales."""
        mask = np.zeros((32, 32, 32), dtype=np.float32)
        mask[10:20, 10:20, 10:20] = 1.0

        mesh1 = create_mesh_from_mask(mask, voxel_spacing=(1.0, 1.0, 1.0))
        mesh2 = create_mesh_from_mask(mask, voxel_spacing=(2.0, 2.0, 2.0))

        assert mesh_surface_area(mesh2) > mesh_surface_area(mesh1)


class TestVoxelVolume:
    """Tests for voxel-based volume computation."""

    def test_known_volume(self) -> None:
        """Volume of 10x10x10 cube with 1mm spacing = 1000 mm³."""
        mask = np.zeros((32, 32, 32), dtype=np.float32)
        mask[0:10, 0:10, 0:10] = 1.0
        vol = compute_voxel_volume(mask, voxel_spacing=(1.0, 1.0, 1.0))
        assert vol == pytest.approx(1000.0)

    def test_spacing_scales_volume(self) -> None:
        """Volume scales with voxel spacing."""
        mask = np.zeros((32, 32, 32), dtype=np.float32)
        mask[0:10, 0:10, 0:10] = 1.0
        vol = compute_voxel_volume(mask, voxel_spacing=(2.0, 2.0, 2.0))
        assert vol == pytest.approx(8000.0)


class TestLongAxis:
    """Tests for PCA-based long-axis computation."""

    def test_elongated_structure(self) -> None:
        """Long axis is longer than short axis for elongated structures."""
        mask = np.zeros((64, 16, 16), dtype=np.float32)
        mask[5:60, 5:12, 5:12] = 1.0
        long_axis = compute_long_axis_length(mask, voxel_spacing=(1.0, 1.0, 1.0))
        assert long_axis > 40.0  # Should be close to 55

    def test_empty_mask(self) -> None:
        """Empty mask returns 0."""
        mask = np.zeros((32, 32, 32), dtype=np.float32)
        assert compute_long_axis_length(mask, (1.0, 1.0, 1.0)) == 0.0


class TestSphericityIndex:
    """Tests for sphericity index computation."""

    def test_sphere_high_sphericity(self) -> None:
        """Sphere-like shape has sphericity close to 1."""
        # Create a sphere
        x, y, z = np.ogrid[-15:16, -15:16, -15:16]
        mask = ((x**2 + y**2 + z**2) <= 12**2).astype(np.float32)

        vol = compute_voxel_volume(mask, (1.0, 1.0, 1.0))
        mesh = create_mesh_from_mask(mask, (1.0, 1.0, 1.0))
        area = mesh_surface_area(mesh)

        psi = compute_sphericity_index(vol, area)
        assert 0.8 < psi <= 1.05  # Close to 1 for a sphere

    def test_zero_inputs(self) -> None:
        """Zero volume or area returns 0."""
        assert compute_sphericity_index(0.0, 100.0) == 0.0
        assert compute_sphericity_index(100.0, 0.0) == 0.0
