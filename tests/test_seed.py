"""Unit tests for seed utility."""

from __future__ import annotations

import numpy as np
import torch

from cardioquant3d.utils.seed import set_seed


class TestSetSeed:
    """Tests for reproducibility seed setting."""

    def test_numpy_reproducibility(self) -> None:
        """NumPy produces same random numbers after set_seed."""
        set_seed(42, deterministic=False)
        a = np.random.rand(10)
        set_seed(42, deterministic=False)
        b = np.random.rand(10)
        np.testing.assert_array_equal(a, b)

    def test_torch_reproducibility(self) -> None:
        """PyTorch produces same random numbers after set_seed."""
        set_seed(42, deterministic=False)
        a = torch.randn(10)
        set_seed(42, deterministic=False)
        b = torch.randn(10)
        assert torch.allclose(a, b)
