"""Evaluation script for CardioQuant3D."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
import mlflow
import numpy as np
import torch
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete, Compose, EnsureType
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.table import Table

from cardioquant3d.data.dataset import get_test_dataloader, build_file_list
from cardioquant3d.data.preprocessing import load_nifti, binarize_label
from cardioquant3d.evaluation.clinical_metrics import (
    ClinicalMetricError,
    ClinicalMetrics,
    compute_clinical_metric_errors,
    compute_clinical_metrics,
)
from cardioquant3d.evaluation.metrics import compute_dice_score, compute_hausdorff_distance
from cardioquant3d.models.unet3d import build_model
from cardioquant3d.utils.logging import setup_logging
from cardioquant3d.utils.seed import set_seed

console = Console()
logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run CardioQuant3D evaluation pipeline.

    Computes segmentation metrics (Dice, Hausdorff) and clinical metric errors
    on the test set.

    Args:
        cfg: Hydra configuration object.
    """
    setup_logging(level="INFO")
    logger.info("CardioQuant3D Evaluation Pipeline")

    set_seed(cfg.experiment.seed, cfg.experiment.deterministic)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load model
    model = build_model(
        in_channels=cfg.model.in_channels,
        out_channels=cfg.model.out_channels,
        channels=list(cfg.model.channels),
        strides=list(cfg.model.strides),
        num_res_units=cfg.model.num_res_units,
        dropout=0.0,  # No dropout at eval
        norm=cfg.model.norm,
    )

    checkpoint_path = cfg.inference.checkpoint_path
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if "model_state_dict" in state_dict:
        model.load_state_dict(state_dict["model_state_dict"])
    else:
        model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()
    logger.info(f"Loaded checkpoint from {checkpoint_path}")

    # Data
    spatial_size = list(cfg.data.spatial_size)
    inference_spacing = tuple(cfg.data.pixel_spacing)

    _preprocess_kwargs = dict(
        pixel_spacing=inference_spacing,
        orientation=cfg.data.orientation,
        intensity_lower=cfg.preprocessing.intensity_lower_percentile,
        intensity_upper=cfg.preprocessing.intensity_upper_percentile,
        intensity_min=cfg.preprocessing.intensity_output_min,
        intensity_max=cfg.preprocessing.intensity_output_max,
    )

    test_loader = get_test_dataloader(
        test_dir=cfg.data.test_dir,
        spatial_size=spatial_size,
        target_label=cfg.data.target_label,
        batch_size=1,
        cache_rate=cfg.data.cache_rate,
        num_workers=cfg.data.num_workers,
        val_transform_kwargs=_preprocess_kwargs,
    )

    post_pred = Compose([EnsureType(), AsDiscrete(argmax=True)])

    # Evaluation
    all_dice: list[float] = []
    all_hausdorff: list[float] = []
    all_volume_errors: list[float] = []
    all_results: list[dict] = []

    console.print("[bold green]Running evaluation...[/bold green]")

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            images = batch["image"].to(device)
            labels = batch["label"]

            # Predict
            outputs = sliding_window_inference(
                images,
                roi_size=spatial_size,
                sw_batch_size=cfg.inference.sw_batch_size,
                predictor=model,
                overlap=cfg.inference.overlap,
            )

            pred = post_pred(outputs[0]).cpu().numpy().squeeze()
            pred_mask = (pred == 1).astype(np.float32)
            gt_mask = labels[0, 0].cpu().numpy().astype(np.float32)

            # Segmentation metrics
            dice = compute_dice_score(pred_mask, gt_mask)
            hd = compute_hausdorff_distance(
                pred_mask, gt_mask,
                voxel_spacing=inference_spacing,
            )

            all_dice.append(dice)
            all_hausdorff.append(hd if np.isfinite(hd) else 0.0)

            # Clinical metrics
            try:
                pred_metrics = compute_clinical_metrics(pred_mask, inference_spacing)
                gt_metrics = compute_clinical_metrics(gt_mask, inference_spacing)
                errors = compute_clinical_metric_errors(pred_metrics, gt_metrics)
                all_volume_errors.append(errors.volume_error_ml)

                result = {
                    "sample": i,
                    "dice": dice,
                    "hausdorff_95": hd if np.isfinite(hd) else None,
                    "pred_volume_ml": pred_metrics.lv_volume_ml,
                    "gt_volume_ml": gt_metrics.lv_volume_ml,
                    "volume_error_ml": errors.volume_error_ml,
                    "volume_error_pct": errors.volume_error_pct,
                    "surface_area_error_mm2": errors.surface_area_error_mm2,
                    "long_axis_error_mm": errors.long_axis_error_mm,
                    "sphericity_error": errors.sphericity_error,
                }
            except ValueError:
                result = {
                    "sample": i,
                    "dice": dice,
                    "hausdorff_95": hd if np.isfinite(hd) else None,
                    "pred_volume_ml": None,
                    "gt_volume_ml": None,
                    "volume_error_ml": None,
                }

            all_results.append(result)

            if (i + 1) % 10 == 0:
                console.print(f"  Processed {i + 1}/{len(test_loader)} samples")

    # Summary statistics
    mean_dice = float(np.mean(all_dice))
    std_dice = float(np.std(all_dice))
    finite_hd = [h for h in all_hausdorff if np.isfinite(h)]
    mean_hd = float(np.mean(finite_hd)) if finite_hd else float("inf")
    std_hd = float(np.std(finite_hd)) if finite_hd else 0.0
    mean_vol_err = float(np.mean(all_volume_errors)) if all_volume_errors else 0.0

    # Display results
    table = Table(title="CardioQuant3D Evaluation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Mean", style="green")
    table.add_column("Std", style="yellow")
    table.add_row("Dice Score", f"{mean_dice:.4f}", f"{std_dice:.4f}")
    table.add_row("Hausdorff Distance (mm)", f"{mean_hd:.2f}", f"{std_hd:.2f}")
    table.add_row("Volume Error (ml)", f"{mean_vol_err:.2f}", "")
    console.print(table)

    # MLflow logging
    mlflow.set_tracking_uri(cfg.experiment.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.experiment.name)

    with mlflow.start_run(run_name="evaluation"):
        mlflow.log_metrics(
            {
                "test_mean_dice": mean_dice,
                "test_std_dice": std_dice,
                "test_mean_hausdorff": mean_hd,
                "test_std_hausdorff": std_hd,
                "test_mean_volume_error_ml": mean_vol_err,
            }
        )

    # Save results to JSON
    output_dir = Path(cfg.experiment.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(
            {
                "summary": {
                    "mean_dice": mean_dice,
                    "std_dice": std_dice,
                    "mean_hausdorff_95": mean_hd,
                    "std_hausdorff_95": std_hd,
                    "mean_volume_error_ml": mean_vol_err,
                },
                "per_sample": all_results,
            },
            f,
            indent=2,
            default=str,
        )

    logger.info(f"Results saved to {results_path}")
    console.print(f"\n[bold green]Evaluation complete! Results saved to {results_path}[/bold green]")


if __name__ == "__main__":
    main()
