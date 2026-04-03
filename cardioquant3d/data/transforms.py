"""MONAI transforms for training and validation pipelines."""

from __future__ import annotations

from typing import Any

from monai.transforms import (
    Compose,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    RandAffined,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    Resized,
    ScaleIntensityRangePercentilesd,
    Spacingd,
    SpatialPadd,
)


class _BinarizeLabelTransform:
    """Picklable callable that binarizes a label tensor for a target class."""

    def __init__(self, target_label: int) -> None:
        self.target_label = target_label

    def __call__(self, x: Any) -> Any:
        return (x == self.target_label).float()


def _binarize_lv_label(target_label: int) -> "Lambdad":
    """Create a transform that binarizes a multi-class label into LV-only.

    Uses a picklable callable class to support multiprocessing on Windows.

    Args:
        target_label: Integer label for the LV class.

    Returns:
        Lambdad transform.
    """
    from monai.transforms import Lambdad

    return Lambdad(
        keys=["label"],
        func=_BinarizeLabelTransform(target_label),
    )


def get_train_transforms(
    spatial_size: list[int],
    target_label: int = 3,
    pixel_spacing: tuple[float, ...] = (1.5, 1.5, 3.0),
    orientation: str = "RAS",
    intensity_lower: float = 0.5,
    intensity_upper: float = 99.5,
    intensity_min: float = 0.0,
    intensity_max: float = 1.0,
    flip_prob: float = 0.5,
    rotate90_prob: float = 0.5,
    affine_prob: float = 0.3,
    affine_rotate_range: tuple[float, ...] = (0.1, 0.1, 0.1),
    affine_scale_range: tuple[float, ...] = (0.1, 0.1, 0.1),
    gaussian_noise_prob: float = 0.2,
    gaussian_noise_std: float = 0.05,
    gaussian_smooth_prob: float = 0.2,
    gaussian_smooth_sigma_range: tuple[float, float] = (0.5, 1.0),
    intensity_scale_prob: float = 0.3,
    intensity_scale_factors: float = 0.1,
    intensity_shift_prob: float = 0.3,
    intensity_shift_offsets: float = 0.1,
) -> Compose:
    """Build training transform pipeline with data augmentation.

    Args:
        spatial_size: Target [H, W, D] dimensions.
        target_label: Label index for LV binarization.
        pixel_spacing: Resampling target spacing (mm).
        orientation: Anatomical orientation code.
        intensity_lower: Lower percentile for intensity clipping.
        intensity_upper: Upper percentile for intensity clipping.
        intensity_min: Output minimum after scaling.
        intensity_max: Output maximum after scaling.
        flip_prob: Probability for random flips.
        rotate90_prob: Probability for random 90-degree rotations.
        affine_prob: Probability for random affine transforms.
        affine_rotate_range: Rotation range per axis (radians).
        affine_scale_range: Scale range per axis.
        gaussian_noise_prob: Probability for Gaussian noise.
        gaussian_noise_std: Standard deviation of Gaussian noise.
        gaussian_smooth_prob: Probability for Gaussian smoothing.
        gaussian_smooth_sigma_range: Sigma range for Gaussian smoothing.
        intensity_scale_prob: Probability for intensity scaling.
        intensity_scale_factors: Scale factor for intensity augmentation.
        intensity_shift_prob: Probability for intensity shift.
        intensity_shift_offsets: Offset for intensity shift augmentation.

    Returns:
        MONAI Compose transform.
    """
    sigma_lo, sigma_hi = gaussian_smooth_sigma_range
    return Compose(
        [
            LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            Orientationd(keys=["image", "label"], axcodes=orientation),
            Spacingd(
                keys=["image", "label"],
                pixdim=pixel_spacing,
                mode=("bilinear", "nearest"),
            ),
            _binarize_lv_label(target_label),
            ScaleIntensityRangePercentilesd(
                keys=["image"],
                lower=intensity_lower,
                upper=intensity_upper,
                b_min=intensity_min,
                b_max=intensity_max,
                clip=True,
            ),
            SpatialPadd(keys=["image", "label"], spatial_size=spatial_size),
            Resized(
                keys=["image", "label"],
                spatial_size=spatial_size,
                mode=("trilinear", "nearest"),
            ),
            RandFlipd(keys=["image", "label"], prob=flip_prob, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=flip_prob, spatial_axis=1),
            RandRotate90d(keys=["image", "label"], prob=rotate90_prob, spatial_axes=(0, 1)),
            RandAffined(
                keys=["image", "label"],
                prob=affine_prob,
                rotate_range=tuple(affine_rotate_range),
                scale_range=tuple(affine_scale_range),
                mode=("bilinear", "nearest"),
                padding_mode="zeros",
            ),
            RandGaussianNoised(keys=["image"], prob=gaussian_noise_prob, std=gaussian_noise_std),
            RandGaussianSmoothd(
                keys=["image"],
                prob=gaussian_smooth_prob,
                sigma_x=(sigma_lo, sigma_hi),
                sigma_y=(sigma_lo, sigma_hi),
                sigma_z=(sigma_lo, sigma_hi),
            ),
            RandScaleIntensityd(keys=["image"], factors=intensity_scale_factors, prob=intensity_scale_prob),
            RandShiftIntensityd(keys=["image"], offsets=intensity_shift_offsets, prob=intensity_shift_prob),
            EnsureTyped(keys=["image", "label"]),
        ]
    )


def get_val_transforms(
    spatial_size: list[int],
    target_label: int = 3,
    pixel_spacing: tuple[float, ...] = (1.5, 1.5, 3.0),
    orientation: str = "RAS",
    intensity_lower: float = 0.5,
    intensity_upper: float = 99.5,
    intensity_min: float = 0.0,
    intensity_max: float = 1.0,
) -> Compose:
    """Build validation/test transform pipeline (no augmentation).

    Args:
        spatial_size: Target [H, W, D] dimensions.
        target_label: Label index for LV binarization.
        pixel_spacing: Resampling target spacing (mm).
        orientation: Anatomical orientation code.
        intensity_lower: Lower percentile for intensity clipping.
        intensity_upper: Upper percentile for intensity clipping.
        intensity_min: Output minimum after scaling.
        intensity_max: Output maximum after scaling.

    Returns:
        MONAI Compose transform.
    """
    return Compose(
        [
            LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            Orientationd(keys=["image", "label"], axcodes=orientation),
            Spacingd(
                keys=["image", "label"],
                pixdim=pixel_spacing,
                mode=("bilinear", "nearest"),
            ),
            _binarize_lv_label(target_label),
            ScaleIntensityRangePercentilesd(
                keys=["image"],
                lower=intensity_lower,
                upper=intensity_upper,
                b_min=intensity_min,
                b_max=intensity_max,
                clip=True,
            ),
            SpatialPadd(keys=["image", "label"], spatial_size=spatial_size),
            Resized(
                keys=["image", "label"],
                spatial_size=spatial_size,
                mode=("trilinear", "nearest"),
            ),
            EnsureTyped(keys=["image", "label"]),
        ]
    )


def get_inference_transforms(
    spatial_size: list[int],
    pixel_spacing: tuple[float, ...] = (1.5, 1.5, 3.0),
    orientation: str = "RAS",
    intensity_lower: float = 0.5,
    intensity_upper: float = 99.5,
    intensity_min: float = 0.0,
    intensity_max: float = 1.0,
) -> Compose:
    """Build inference transform pipeline (image-only, no label).

    Args:
        spatial_size: Target [H, W, D] dimensions.
        pixel_spacing: Resampling target spacing (mm).
        orientation: Anatomical orientation code.
        intensity_lower: Lower percentile for intensity clipping.
        intensity_upper: Upper percentile for intensity clipping.
        intensity_min: Output minimum after scaling.
        intensity_max: Output maximum after scaling.

    Returns:
        MONAI Compose transform.
    """
    return Compose(
        [
            LoadImaged(keys=["image"], ensure_channel_first=True),
            Orientationd(keys=["image"], axcodes=orientation),
            Spacingd(
                keys=["image"],
                pixdim=pixel_spacing,
                mode="bilinear",
            ),
            ScaleIntensityRangePercentilesd(
                keys=["image"],
                lower=intensity_lower,
                upper=intensity_upper,
                b_min=intensity_min,
                b_max=intensity_max,
                clip=True,
            ),
            SpatialPadd(keys=["image"], spatial_size=spatial_size),
            Resized(
                keys=["image"],
                spatial_size=spatial_size,
                mode="trilinear",
            ),
            EnsureTyped(keys=["image"]),
        ]
    )
