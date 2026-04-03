"""Unit tests for preprocessing utilities."""

from __future__ import annotations

import numpy as np
import pytest

from cardioquant3d.data.preprocessing import (
    binarize_label,
    normalize_intensity,
    resample_volume,
)


class TestNormalizeIntensity:
    """Tests for intensity normalization."""

    def test_zero_mean(self) -> None:
        """Normalized volume has approximately zero mean."""
        vol = np.random.rand(32, 32, 16).astype(np.float32) * 1000
        norm = normalize_intensity(vol)
        assert abs(norm.mean()) < 0.1

    def test_unit_std(self) -> None:
        """Normalized volume has approximately unit std."""
        vol = np.random.rand(32, 32, 16).astype(np.float32) * 1000
        norm = normalize_intensity(vol)
        assert abs(norm.std() - 1.0) < 0.1

    def test_constant_volume(self) -> None:
        """Constant volume returns zero (no div-by-zero)."""
        vol = np.ones((32, 32, 16), dtype=np.float32) * 42
        norm = normalize_intensity(vol)
        assert np.allclose(norm, 0.0)


class TestBinarizeLabel:
    """Tests for label binarization."""

    def test_binarize_lv(self) -> None:
        """Only target label (3=LV) becomes 1."""
        label = np.array([0, 1, 2, 3, 0, 3], dtype=np.float32)
        binary = binarize_label(label, target_label=3)
        expected = np.array([0, 0, 0, 1, 0, 1], dtype=np.float32)
        np.testing.assert_array_equal(binary, expected)


class TestResampleVolume:
    """Tests for volume resampling."""

    def test_resample_shape(self) -> None:
        """Resampled volume has correct target shape."""
        vol = np.random.rand(64, 64, 32).astype(np.float32)
        resampled = resample_volume(vol, target_shape=(128, 128, 64))
        assert resampled.shape == (128, 128, 64)

    def test_resample_preserves_dtype(self) -> None:
        """Resampled volume is float32."""
        vol = np.random.rand(32, 32, 16).astype(np.float32)
        resampled = resample_volume(vol, target_shape=(16, 16, 8))
        assert resampled.dtype == np.float32
