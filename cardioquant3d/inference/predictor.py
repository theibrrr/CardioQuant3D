"""Inference predictor for 3D cardiac segmentation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from monai.inferers import sliding_window_inference
from monai.transforms import Compose, EnsureType, AsDiscrete

from cardioquant3d.data.preprocessing import load_nifti
from cardioquant3d.data.transforms import get_inference_transforms
from cardioquant3d.evaluation.clinical_metrics import ClinicalMetrics, compute_clinical_metrics
from cardioquant3d.models.unet3d import UNet3D, build_model


class Predictor:
    """Inference engine for 3D cardiac segmentation and clinical analysis.

    Args:
        model: Trained 3D U-Net model.
        device: Compute device.
        spatial_size: Spatial size for sliding window inference.
        sw_batch_size: Sliding window batch size.
        overlap: Sliding window overlap fraction.
        inference_spacing: Voxel spacing used for transforms and clinical metrics.
        transform_kwargs: Extra kwargs forwarded to get_inference_transforms.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device | None = None,
        spatial_size: list[int] | None = None,
        sw_batch_size: int = 4,
        overlap: float = 0.5,
        inference_spacing: tuple[float, ...] = (1.5, 1.5, 3.0),
        transform_kwargs: dict | None = None,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.spatial_size = spatial_size or [128, 128, 32]
        self.sw_batch_size = sw_batch_size
        self.overlap = overlap
        self.inference_spacing = inference_spacing
        self.transform_kwargs = transform_kwargs or {}
        self.post_transform = Compose([EnsureType(), AsDiscrete(argmax=True)])

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        model_config: dict[str, Any] | None = None,
        device: torch.device | None = None,
        spatial_size: list[int] | None = None,
        sw_batch_size: int = 4,
        overlap: float = 0.5,
        inference_spacing: tuple[float, ...] = (1.5, 1.5, 3.0),
        transform_kwargs: dict | None = None,
    ) -> "Predictor":
        """Create a Predictor from a saved checkpoint.

        Args:
            checkpoint_path: Path to the model checkpoint (.pth).
            model_config: Model configuration dict.
            device: Compute device.
            spatial_size: Spatial size for sliding window inference.
            sw_batch_size: Sliding window batch size.
            overlap: Sliding window overlap fraction.

        Returns:
            Configured Predictor instance.
        """
        config = model_config or {}
        model = build_model(
            in_channels=config.get("in_channels", 1),
            out_channels=config.get("out_channels", 2),
            channels=config.get("channels", [32, 64, 128, 256]),
            strides=config.get("strides", [2, 2, 2]),
            num_res_units=config.get("num_res_units", 2),
            dropout=config.get("dropout", 0.0),  # No dropout at inference
            norm=config.get("norm", "batch"),
        )

        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)

        # Handle both full checkpoint and state-dict-only saves
        if "model_state_dict" in state_dict:
            model.load_state_dict(state_dict["model_state_dict"])
        else:
            model.load_state_dict(state_dict)

        return cls(
            model=model,
            device=device,
            spatial_size=spatial_size,
            sw_batch_size=sw_batch_size,
            overlap=overlap,
            inference_spacing=inference_spacing,
            transform_kwargs=transform_kwargs,
        )

    @torch.no_grad()
    def predict_volume(self, image_tensor: torch.Tensor) -> np.ndarray:
        """Run segmentation on a preprocessed image tensor.

        Args:
            image_tensor: Input tensor of shape (1, 1, H, W, D).

        Returns:
            Binary segmentation mask as numpy array (H, W, D).
        """
        image_tensor = image_tensor.to(self.device)

        output = sliding_window_inference(
            image_tensor,
            roi_size=self.spatial_size,
            sw_batch_size=self.sw_batch_size,
            predictor=self.model,
            overlap=self.overlap,
        )

        # Convert to discrete prediction
        prediction = self.post_transform(output[0])  # Remove batch dim
        # Channel 1 is LV (argmax selects the class channel)
        mask = (prediction == 1).cpu().numpy().squeeze()

        return mask.astype(np.float32)

    def predict_nifti(
        self,
        nifti_path: str,
    ) -> tuple[np.ndarray, tuple[float, ...], np.ndarray]:
        """Run segmentation on a NIfTI file.

        Args:
            nifti_path: Path to input .nii / .nii.gz file.

        Returns:
            Tuple of (binary mask, voxel spacing, affine matrix).
        """
        transforms = get_inference_transforms(self.spatial_size, **self.transform_kwargs)
        data = {"image": nifti_path}
        transformed = transforms(data)
        image_tensor = transformed["image"].unsqueeze(0)  # Add batch dim

        mask = self.predict_volume(image_tensor)

        # Keep preprocessed image for visualization (same spatial space as mask)
        processed_image = transformed["image"].numpy().squeeze()

        # Get original spacing
        nii = nib.load(nifti_path)
        spacing = tuple(float(s) for s in nii.header.get_zooms()[:3])
        affine = nii.affine

        return mask, spacing, affine, processed_image

    def analyze(
        self,
        nifti_path: str,
    ) -> tuple[ClinicalMetrics, np.ndarray, np.ndarray]:
        """Full pipeline: segment + compute clinical metrics.

        Args:
            nifti_path: Path to input NIfTI file.

        Returns:
            Tuple of (ClinicalMetrics, binary segmentation mask, preprocessed image).
        """
        mask, spacing, _, processed_image = self.predict_nifti(nifti_path)

        # Use the configured inference spacing
        metrics = compute_clinical_metrics(mask, self.inference_spacing)

        return metrics, mask, processed_image
