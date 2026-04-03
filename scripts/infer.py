"""Single-file inference script for CardioQuant3D."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig
from rich.console import Console
from rich.panel import Panel

from cardioquant3d.inference.predictor import Predictor
from cardioquant3d.utils.io import save_nifti
from cardioquant3d.utils.logging import setup_logging

console = Console()
logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run inference on a single NIfTI file.

    Args:
        cfg: Hydra configuration object.
    """
    setup_logging(level="INFO")
    logger.info("CardioQuant3D Inference")

    # Parse additional CLI args
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=True, help="Input NIfTI file path")
    parser.add_argument("--output-dir", "-o", type=str, default=None, help="Output directory")

    # Filter out Hydra args
    remaining_args = []
    skip_next = False
    for arg in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if arg.startswith("--config") or arg.startswith("-c"):
            skip_next = True
            continue
        if not arg.startswith("+") and not "=" in arg:
            remaining_args.append(arg)
        elif arg.startswith("--input") or arg.startswith("-i") or arg.startswith("--output"):
            remaining_args.append(arg)

    args, _ = parser.parse_known_args(remaining_args)

    input_path = args.input
    output_dir = Path(args.output_dir or cfg.inference.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build predictor from checkpoint
    model_config = {
        "in_channels": cfg.model.in_channels,
        "out_channels": cfg.model.out_channels,
        "channels": list(cfg.model.channels),
        "strides": list(cfg.model.strides),
        "num_res_units": cfg.model.num_res_units,
        "norm": cfg.model.norm,
    }

    inference_spacing = tuple(cfg.data.pixel_spacing)
    _transform_kwargs = dict(
        pixel_spacing=inference_spacing,
        orientation=cfg.data.orientation,
        intensity_lower=cfg.preprocessing.intensity_lower_percentile,
        intensity_upper=cfg.preprocessing.intensity_upper_percentile,
        intensity_min=cfg.preprocessing.intensity_output_min,
        intensity_max=cfg.preprocessing.intensity_output_max,
    )

    predictor = Predictor.from_checkpoint(
        checkpoint_path=cfg.inference.checkpoint_path,
        model_config=model_config,
        spatial_size=list(cfg.data.spatial_size),
        sw_batch_size=cfg.inference.sw_batch_size,
        overlap=cfg.inference.overlap,
        inference_spacing=inference_spacing,
        transform_kwargs=_transform_kwargs,
    )

    console.print(f"[bold green]Analyzing: {input_path}[/bold green]")

    metrics, mask, _ = predictor.analyze(input_path)

    # Save results
    input_name = Path(input_path).stem.replace(".nii", "")

    # Save predicted mask
    import nibabel as nib

    nii = nib.load(input_path)
    save_nifti(mask, nii.affine, output_dir / f"{input_name}_pred_mask.nii.gz")

    # Save metrics
    metrics_dict = {
        "lv_volume_ml": round(metrics.lv_volume_ml, 2),
        "surface_area_mm2": round(metrics.surface_area_mm2, 2),
        "long_axis_mm": round(metrics.long_axis_mm, 2),
        "sphericity_index": round(metrics.sphericity_index, 4),
    }

    metrics_path = output_dir / f"{input_name}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)

    # Display results
    panel = Panel(
        f"[cyan]LV Volume:[/cyan]        {metrics.lv_volume_ml:.2f} ml\n"
        f"[cyan]Surface Area:[/cyan]     {metrics.surface_area_mm2:.2f} mm²\n"
        f"[cyan]Long Axis:[/cyan]        {metrics.long_axis_mm:.2f} mm\n"
        f"[cyan]Sphericity Index:[/cyan] {metrics.sphericity_index:.4f}",
        title="CardioQuant3D Analysis Results",
        border_style="green",
    )
    console.print(panel)
    console.print(f"\nMask saved to: {output_dir / f'{input_name}_pred_mask.nii.gz'}")
    console.print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
