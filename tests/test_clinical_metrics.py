"""Unit tests for clinical metrics."""

from __future__ import annotations

import numpy as np
import pytest

from cardioquant3d.evaluation.clinical_metrics import (
    ClinicalMetrics,
    compute_clinical_metric_errors,
)


class TestClinicalMetricErrors:
    """Tests for clinical metric error computation."""

    def test_zero_error_for_identical_metrics(self) -> None:
        """Identical metrics produce zero error."""
        m = ClinicalMetrics(
            lv_volume_ml=100.0,
            surface_area_mm2=5000.0,
            long_axis_mm=80.0,
            sphericity_index=0.75,
        )
        errors = compute_clinical_metric_errors(m, m)
        assert errors.volume_error_ml == pytest.approx(0.0)
        assert errors.volume_error_pct == pytest.approx(0.0)
        assert errors.surface_area_error_mm2 == pytest.approx(0.0)
        assert errors.long_axis_error_mm == pytest.approx(0.0)
        assert errors.sphericity_error == pytest.approx(0.0)

    def test_known_errors(self) -> None:
        """Known error values are correctly computed."""
        pred = ClinicalMetrics(
            lv_volume_ml=110.0,
            surface_area_mm2=5500.0,
            long_axis_mm=85.0,
            sphericity_index=0.80,
        )
        gt = ClinicalMetrics(
            lv_volume_ml=100.0,
            surface_area_mm2=5000.0,
            long_axis_mm=80.0,
            sphericity_index=0.75,
        )
        errors = compute_clinical_metric_errors(pred, gt)
        assert errors.volume_error_ml == pytest.approx(10.0)
        assert errors.volume_error_pct == pytest.approx(10.0)
        assert errors.surface_area_error_mm2 == pytest.approx(500.0)
        assert errors.long_axis_error_mm == pytest.approx(5.0)
        assert errors.sphericity_error == pytest.approx(0.05)
