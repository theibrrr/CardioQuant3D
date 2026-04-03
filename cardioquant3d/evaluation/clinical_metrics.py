"""Clinical metric computation and error analysis vs ground truth."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cardioquant3d.geometry.measurements import GeometricMeasurements, compute_geometric_measurements
from cardioquant3d.geometry.mesh import create_mesh_from_mask


@dataclass
class ClinicalMetrics:
    """Clinical metrics extracted from a cardiac segmentation mask.

    Attributes:
        lv_volume_ml: Left ventricle volume in milliliters.
        surface_area_mm2: LV surface area in square millimeters.
        long_axis_mm: LV long-axis length in millimeters.
        sphericity_index: LV sphericity index (dimensionless).
    """

    lv_volume_ml: float
    surface_area_mm2: float
    long_axis_mm: float
    sphericity_index: float


@dataclass
class ClinicalMetricError:
    """Absolute and relative errors between predicted and GT clinical metrics.

    Attributes:
        volume_error_ml: Absolute error in LV volume (ml).
        volume_error_pct: Relative error in LV volume (%).
        surface_area_error_mm2: Absolute error in surface area (mm²).
        surface_area_error_pct: Relative error in surface area (%).
        long_axis_error_mm: Absolute error in long-axis length (mm).
        long_axis_error_pct: Relative error in long-axis length (%).
        sphericity_error: Absolute error in sphericity index.
        sphericity_error_pct: Relative error in sphericity index (%).
    """

    volume_error_ml: float
    volume_error_pct: float
    surface_area_error_mm2: float
    surface_area_error_pct: float
    long_axis_error_mm: float
    long_axis_error_pct: float
    sphericity_error: float
    sphericity_error_pct: float


def compute_clinical_metrics(
    mask: np.ndarray,
    voxel_spacing: tuple[float, ...],
) -> ClinicalMetrics:
    """Compute clinical metrics from a binary segmentation mask.

    Args:
        mask: Binary 3D mask of the LV.
        voxel_spacing: Physical voxel dimensions (mm) as (sx, sy, sz).

    Returns:
        ClinicalMetrics dataclass.
    """
    mesh = create_mesh_from_mask(mask, voxel_spacing)
    measurements = compute_geometric_measurements(mesh, mask, voxel_spacing)

    return ClinicalMetrics(
        lv_volume_ml=measurements.volume_ml,
        surface_area_mm2=measurements.surface_area_mm2,
        long_axis_mm=measurements.long_axis_mm,
        sphericity_index=measurements.sphericity_index,
    )


def compute_clinical_metric_errors(
    predicted: ClinicalMetrics,
    ground_truth: ClinicalMetrics,
) -> ClinicalMetricError:
    """Compute absolute and relative errors between predicted and ground truth.

    Args:
        predicted: Clinical metrics from predicted mask.
        ground_truth: Clinical metrics from ground truth mask.

    Returns:
        ClinicalMetricError dataclass.
    """

    def _rel_error(pred: float, gt: float) -> float:
        if abs(gt) < 1e-8:
            return 0.0 if abs(pred) < 1e-8 else 100.0
        return abs(pred - gt) / abs(gt) * 100.0

    return ClinicalMetricError(
        volume_error_ml=abs(predicted.lv_volume_ml - ground_truth.lv_volume_ml),
        volume_error_pct=_rel_error(predicted.lv_volume_ml, ground_truth.lv_volume_ml),
        surface_area_error_mm2=abs(
            predicted.surface_area_mm2 - ground_truth.surface_area_mm2
        ),
        surface_area_error_pct=_rel_error(
            predicted.surface_area_mm2, ground_truth.surface_area_mm2
        ),
        long_axis_error_mm=abs(predicted.long_axis_mm - ground_truth.long_axis_mm),
        long_axis_error_pct=_rel_error(
            predicted.long_axis_mm, ground_truth.long_axis_mm
        ),
        sphericity_error=abs(
            predicted.sphericity_index - ground_truth.sphericity_index
        ),
        sphericity_error_pct=_rel_error(
            predicted.sphericity_index, ground_truth.sphericity_index
        ),
    )
