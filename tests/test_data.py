"""Unit tests for data module."""

from __future__ import annotations

from pathlib import Path

import pytest

from cardioquant3d.data.dataset import parse_info_cfg, build_file_list, PatientInfo


class TestParseInfoCfg:
    """Tests for Info.cfg parser."""

    def test_parse_valid_cfg(self, tmp_path: Path) -> None:
        """Successfully parse a valid Info.cfg file."""
        cfg_content = "ED: 1\nES: 12\nGroup: DCM\nHeight: 184.0\nNbFrame: 30\nWeight: 95.0\n"
        cfg_path = tmp_path / "patient001" / "Info.cfg"
        cfg_path.parent.mkdir(parents=True)
        cfg_path.write_text(cfg_content)

        info = parse_info_cfg(cfg_path)
        assert info.ed_frame == 1
        assert info.es_frame == 12
        assert info.group == "DCM"
        assert info.height == 184.0
        assert info.weight == 95.0
        assert info.nb_frame == 30
        assert info.patient_id == "patient001"


class TestBuildFileList:
    """Tests for file list builder."""

    def test_empty_dir(self, tmp_path: Path) -> None:
        """Empty directory returns empty list."""
        samples = build_file_list(str(tmp_path))
        assert samples == []

    def test_finds_patient_files(self, tmp_path: Path) -> None:
        """Correctly finds image/label pairs for a patient."""
        patient_dir = tmp_path / "patient001"
        patient_dir.mkdir()

        # Create Info.cfg
        (patient_dir / "Info.cfg").write_text(
            "ED: 1\nES: 12\nGroup: DCM\nHeight: 184.0\nNbFrame: 30\nWeight: 95.0\n"
        )

        # Create dummy NIfTI files
        (patient_dir / "patient001_frame01.nii").write_bytes(b"dummy")
        (patient_dir / "patient001_frame01_gt.nii").write_bytes(b"dummy")
        (patient_dir / "patient001_frame12.nii").write_bytes(b"dummy")
        (patient_dir / "patient001_frame12_gt.nii").write_bytes(b"dummy")

        samples = build_file_list(str(tmp_path))
        assert len(samples) == 2
        assert samples[0]["patient_id"] == "patient001"
        assert "frame01" in samples[0]["frame"]
