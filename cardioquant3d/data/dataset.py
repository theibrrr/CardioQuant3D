"""ACDC cardiac MRI dataset loader for 3D segmentation."""

from __future__ import annotations

import configparser
import glob
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
from monai.data import CacheDataset, DataLoader
from monai.transforms import Compose

from cardioquant3d.data.transforms import get_train_transforms, get_val_transforms


@dataclass
class PatientInfo:
    """Parsed patient metadata from Info.cfg."""

    patient_id: str
    ed_frame: int
    es_frame: int
    group: str
    height: float
    weight: float
    nb_frame: int


def parse_info_cfg(info_path: Path) -> PatientInfo:
    """Parse an ACDC Info.cfg file into structured metadata.

    Args:
        info_path: Path to the Info.cfg file.

    Returns:
        PatientInfo dataclass with parsed fields.
    """
    config: dict[str, str] = {}
    with open(info_path, "r") as f:
        for line in f:
            line = line.strip()
            if ":" in line:
                key, value = line.split(":", 1)
                config[key.strip()] = value.strip()

    patient_id = info_path.parent.name
    return PatientInfo(
        patient_id=patient_id,
        ed_frame=int(config["ED"]),
        es_frame=int(config["ES"]),
        group=config.get("Group", "Unknown"),
        height=float(config.get("Height", 0.0)),
        weight=float(config.get("Weight", 0.0)),
        nb_frame=int(config.get("NbFrame", 0)),
    )


def build_file_list(
    data_dir: str,
    target_label: int = 3,
) -> list[dict[str, Any]]:
    """Scan ACDC directory and build a list of image/label pairs.

    Each patient folder is expected to contain:
      - patientXXX_frameYY.nii(.gz)      → image
      - patientXXX_frameYY_gt.nii(.gz)   → ground-truth mask

    Only ED and ES frames (as specified in Info.cfg) are included.

    Args:
        data_dir: Path to the ACDC split directory (training or testing).
        target_label: Label index for the target structure (3 = LV).

    Returns:
        List of dicts with keys: image, label, patient_id, frame, info.
    """
    data_dir = Path(data_dir)
    samples: list[dict[str, Any]] = []

    patient_dirs = sorted(
        [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("patient")]
    )

    for patient_dir in patient_dirs:
        info_path = patient_dir / "Info.cfg"
        if not info_path.exists():
            continue

        info = parse_info_cfg(info_path)
        patient_id = patient_dir.name

        for frame_idx in [info.ed_frame, info.es_frame]:
            frame_str = f"frame{frame_idx:02d}"
            image_pattern = str(patient_dir / f"{patient_id}_{frame_str}.nii*")
            label_pattern = str(patient_dir / f"{patient_id}_{frame_str}_gt.nii*")

            image_files = glob.glob(image_pattern)
            label_files = glob.glob(label_pattern)

            if image_files and label_files:
                samples.append(
                    {
                        "image": image_files[0],
                        "label": label_files[0],
                        "patient_id": patient_id,
                        "frame": frame_str,
                    }
                )

    return samples


def get_dataloaders(
    train_dir: str,
    spatial_size: list[int],
    target_label: int = 3,
    batch_size: int = 2,
    val_split: float = 0.2,
    cache_rate: float = 0.5,
    num_workers: int = 4,
    seed: int = 42,
    train_transform_kwargs: dict | None = None,
    val_transform_kwargs: dict | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Create training and validation DataLoaders from the ACDC training set.

    Args:
        train_dir: Path to the ACDC training directory.
        spatial_size: Target spatial dimensions [H, W, D].
        target_label: Label index for LV (default 3).
        batch_size: Batch size for both loaders.
        val_split: Fraction of data used for validation.
        cache_rate: Fraction of data to cache in memory.
        num_workers: Number of data loading workers.
        seed: Random seed for reproducible splits.
        train_transform_kwargs: Extra kwargs for get_train_transforms.
        val_transform_kwargs: Extra kwargs for get_val_transforms.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    all_samples = build_file_list(train_dir, target_label)

    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(all_samples))
    val_size = int(len(all_samples) * val_split)

    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_samples = [all_samples[i] for i in train_indices]
    val_samples = [all_samples[i] for i in val_indices]

    train_transforms = get_train_transforms(spatial_size, target_label, **(train_transform_kwargs or {}))
    val_transforms = get_val_transforms(spatial_size, target_label, **(val_transform_kwargs or {}))

    train_ds = CacheDataset(
        data=train_samples,
        transform=train_transforms,
        cache_rate=cache_rate,
        num_workers=num_workers,
    )
    val_ds = CacheDataset(
        data=val_samples,
        transform=val_transforms,
        cache_rate=cache_rate,
        num_workers=num_workers,
    )

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader


def get_test_dataloader(
    test_dir: str,
    spatial_size: list[int],
    target_label: int = 3,
    batch_size: int = 1,
    cache_rate: float = 1.0,
    num_workers: int = 4,
    val_transform_kwargs: dict | None = None,
) -> DataLoader:
    """Create a test DataLoader from the ACDC testing set.

    Args:
        test_dir: Path to the ACDC testing directory.
        spatial_size: Target spatial dimensions [H, W, D].
        target_label: Label index for LV (default 3).
        batch_size: Batch size.
        cache_rate: Fraction of data to cache in memory.
        num_workers: Number of data loading workers.
        val_transform_kwargs: Extra kwargs for get_val_transforms.

    Returns:
        Test DataLoader.
    """
    test_samples = build_file_list(test_dir, target_label)
    test_transforms = get_val_transforms(spatial_size, target_label, **(val_transform_kwargs or {}))

    test_ds = CacheDataset(
        data=test_samples,
        transform=test_transforms,
        cache_rate=cache_rate,
        num_workers=num_workers,
    )

    pin_memory = torch.cuda.is_available()

    return DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
