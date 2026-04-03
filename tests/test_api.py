"""Unit tests for the FastAPI application."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from cardioquant3d.api.main import app


@pytest.fixture
def client() -> TestClient:
    """Create a test client for the FastAPI app."""
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_check(self, client: TestClient) -> None:
        """Health endpoint returns 200."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"


class TestAnalyzeEndpoint:
    """Tests for the /analyze endpoint."""

    def test_reject_non_nifti(self, client: TestClient) -> None:
        """Non-NIfTI files are rejected with 400."""
        response = client.post(
            "/analyze",
            files={"file": ("test.png", b"fake data", "image/png")},
        )
        assert response.status_code == 400

    def test_reject_empty_file(self, client: TestClient) -> None:
        """Empty files are rejected."""
        response = client.post(
            "/analyze",
            files={"file": ("test.nii", b"", "application/octet-stream")},
        )
        # Should get 400 (empty) or 503 (no model)
        assert response.status_code in (400, 503)
