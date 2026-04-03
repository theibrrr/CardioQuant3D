"""Training script for CardioQuant3D 3D U-Net."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from rich.console import Console

from cardioquant3d.data.dataset import get_dataloaders
from cardioquant3d.models.unet3d import build_model
from cardioquant3d.training.losses import CombinedSegmentationLoss
from cardioquant3d.training.trainer import CheckpointManager, EarlyStopping, Trainer
from cardioquant3d.utils.logging import setup_logging
from cardioquant3d.utils.seed import set_seed

console = Console()
logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run CardioQuant3D training pipeline.

    Args:
        cfg: Hydra configuration object.
    """
    setup_logging(level="INFO")
    logger.info("CardioQuant3D Training Pipeline")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Reproducibility
    set_seed(cfg.experiment.seed, cfg.experiment.deterministic)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Data
    console.print("[bold green]Loading ACDC dataset...[/bold green]")

    # Build kwargs dicts that get forwarded to transform functions
    _preprocess_kwargs = dict(
        pixel_spacing=tuple(cfg.data.pixel_spacing),
        orientation=cfg.data.orientation,
        intensity_lower=cfg.preprocessing.intensity_lower_percentile,
        intensity_upper=cfg.preprocessing.intensity_upper_percentile,
        intensity_min=cfg.preprocessing.intensity_output_min,
        intensity_max=cfg.preprocessing.intensity_output_max,
    )
    _augment_kwargs = dict(
        flip_prob=cfg.augmentation.flip_prob,
        rotate90_prob=cfg.augmentation.rotate90_prob,
        affine_prob=cfg.augmentation.affine_prob,
        affine_rotate_range=tuple(cfg.augmentation.affine_rotate_range),
        affine_scale_range=tuple(cfg.augmentation.affine_scale_range),
        gaussian_noise_prob=cfg.augmentation.gaussian_noise_prob,
        gaussian_noise_std=cfg.augmentation.gaussian_noise_std,
        gaussian_smooth_prob=cfg.augmentation.gaussian_smooth_prob,
        gaussian_smooth_sigma_range=tuple(cfg.augmentation.gaussian_smooth_sigma_range),
        intensity_scale_prob=cfg.augmentation.intensity_scale_prob,
        intensity_scale_factors=cfg.augmentation.intensity_scale_factors,
        intensity_shift_prob=cfg.augmentation.intensity_shift_prob,
        intensity_shift_offsets=cfg.augmentation.intensity_shift_offsets,
    )

    train_loader, val_loader = get_dataloaders(
        train_dir=cfg.data.train_dir,
        spatial_size=list(cfg.data.spatial_size),
        target_label=cfg.data.target_label,
        batch_size=cfg.training.batch_size,
        val_split=cfg.training.val_split,
        cache_rate=cfg.data.cache_rate,
        num_workers=cfg.data.num_workers,
        seed=cfg.experiment.seed,
        train_transform_kwargs={**_preprocess_kwargs, **_augment_kwargs},
        val_transform_kwargs=_preprocess_kwargs,
    )
    logger.info(
        f"Train: {len(train_loader.dataset)} samples, "
        f"Val: {len(val_loader.dataset)} samples"
    )

    # Model
    console.print("[bold green]Building 3D U-Net model...[/bold green]")
    model = build_model(
        in_channels=cfg.model.in_channels,
        out_channels=cfg.model.out_channels,
        channels=list(cfg.model.channels),
        strides=list(cfg.model.strides),
        num_res_units=cfg.model.num_res_units,
        dropout=cfg.model.dropout,
        norm=cfg.model.norm,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Loss
    loss_fn = CombinedSegmentationLoss(
        include_background=cfg.training.loss.include_background,
        softmax=cfg.training.loss.softmax,
        lambda_dice=cfg.training.loss.lambda_dice,
        lambda_ce=cfg.training.loss.lambda_ce,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.training.epochs,
        eta_min=cfg.training.scheduler_eta_min,
    )

    # Early stopping
    early_stopping = None
    if cfg.training.early_stopping.enabled:
        early_stopping = EarlyStopping(
            patience=cfg.training.early_stopping.patience,
            min_delta=cfg.training.early_stopping.min_delta,
            mode="max",
        )

    # Checkpoint manager
    output_dir = Path(cfg.experiment.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_manager = None
    if cfg.training.checkpointing.enabled:
        checkpoint_manager = CheckpointManager(
            save_dir=str(output_dir / "checkpoints"),
            save_top_k=cfg.training.checkpointing.save_top_k,
            monitor=cfg.training.checkpointing.monitor,
            mode=cfg.training.checkpointing.mode,
        )

    # Trainer
    console.print("[bold green]Starting training...[/bold green]")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        device=device,
        epochs=cfg.training.epochs,
        amp=cfg.training.amp,
        gradient_clip_max_norm=cfg.training.gradient_clip_max_norm,
        val_interval=cfg.training.val_interval,
        early_stopping=early_stopping,
        checkpoint_manager=checkpoint_manager,
        mlflow_tracking_uri=cfg.experiment.mlflow_tracking_uri,
        experiment_name=cfg.experiment.name,
        spatial_size=list(cfg.data.spatial_size),
    )

    results = trainer.train()

    console.print(f"\n[bold green]Training complete![/bold green]")
    console.print(f"  Best validation Dice: {results['best_dice']:.4f}")
    console.print(f"  Best model saved to: {results['best_model_path']}")


if __name__ == "__main__":
    main()
