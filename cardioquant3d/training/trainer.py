"""Training engine for 3D cardiac segmentation."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch
import torch.nn as nn
from monai.data import DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete, Compose, EnsureType
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from torch.amp import GradScaler, autocast

from cardioquant3d.training.losses import CombinedSegmentationLoss

logger = logging.getLogger(__name__)
console = Console()


class EarlyStopping:
    """Early stopping tracker.

    Args:
        patience: Number of epochs to wait for improvement.
        min_delta: Minimum change to qualify as an improvement.
        mode: 'max' for metrics like Dice, 'min' for loss.
    """

    def __init__(
        self,
        patience: int = 30,
        min_delta: float = 0.001,
        mode: str = "max",
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: float | None = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        """Check if training should stop.

        Args:
            score: Current metric value.

        Returns:
            True if training should stop.
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True

        return False


class CheckpointManager:
    """Manages model checkpoint saving.

    Args:
        save_dir: Directory to save checkpoints.
        save_top_k: Number of best checkpoints to keep.
        monitor: Metric name to monitor.
        mode: 'max' or 'min'.
    """

    def __init__(
        self,
        save_dir: str,
        save_top_k: int = 3,
        monitor: str = "val_dice",
        mode: str = "max",
    ) -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_top_k = save_top_k
        self.monitor = monitor
        self.mode = mode
        self.best_scores: list[tuple[float, Path]] = []

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        score: float,
    ) -> Path | None:
        """Save model if score is among top-k.

        Args:
            model: Model to save.
            optimizer: Optimizer state.
            epoch: Current epoch.
            score: Current metric value.

        Returns:
            Path to saved checkpoint, or None if not saved.
        """
        should_save = len(self.best_scores) < self.save_top_k

        if not should_save and self.best_scores:
            if self.mode == "max":
                worst = min(self.best_scores, key=lambda x: x[0])
                should_save = score > worst[0]
            else:
                worst = max(self.best_scores, key=lambda x: x[0])
                should_save = score < worst[0]

        if should_save:
            ckpt_path = self.save_dir / f"checkpoint_epoch{epoch:03d}_{self.monitor}_{score:.4f}.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    f"{self.monitor}": score,
                },
                ckpt_path,
            )
            self.best_scores.append((score, ckpt_path))

            # Remove worst if exceeding top-k
            if len(self.best_scores) > self.save_top_k:
                if self.mode == "max":
                    worst = min(self.best_scores, key=lambda x: x[0])
                else:
                    worst = max(self.best_scores, key=lambda x: x[0])
                self.best_scores.remove(worst)
                if worst[1].exists():
                    worst[1].unlink()

            return ckpt_path

        return None

    @property
    def best_checkpoint(self) -> Path | None:
        """Return path to the best checkpoint."""
        if not self.best_scores:
            return None
        if self.mode == "max":
            return max(self.best_scores, key=lambda x: x[0])[1]
        return min(self.best_scores, key=lambda x: x[0])[1]


class Trainer:
    """Training engine for 3D U-Net segmentation.

    Args:
        model: 3D U-Net model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        optimizer: Optimizer instance.
        scheduler: Learning rate scheduler.
        loss_fn: Loss function.
        device: Compute device.
        epochs: Number of training epochs.
        amp: Enable automatic mixed precision.
        gradient_clip_max_norm: Maximum gradient norm for clipping.
        val_interval: Run validation every N epochs.
        early_stopping: EarlyStopping instance or None.
        checkpoint_manager: CheckpointManager instance or None.
        mlflow_tracking_uri: MLflow tracking URI.
        experiment_name: MLflow experiment name.
        spatial_size: Spatial size for sliding window inference.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Any | None = None,
        loss_fn: nn.Module | None = None,
        device: torch.device | None = None,
        epochs: int = 200,
        amp: bool = True,
        gradient_clip_max_norm: float = 1.0,
        val_interval: int = 2,
        early_stopping: EarlyStopping | None = None,
        checkpoint_manager: CheckpointManager | None = None,
        mlflow_tracking_uri: str = "file:./mlruns",
        experiment_name: str = "cardioquant3d",
        spatial_size: list[int] | None = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn or CombinedSegmentationLoss()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = epochs
        self.amp = amp
        self.gradient_clip_max_norm = gradient_clip_max_norm
        self.val_interval = val_interval
        self.early_stopping = early_stopping
        self.checkpoint_manager = checkpoint_manager
        self.spatial_size = spatial_size or [128, 128, 32]

        self.model = self.model.to(self.device)
        self.loss_fn = self.loss_fn.to(self.device)
        self.scaler = GradScaler("cpu" if self.device.type == "cpu" else "cuda", enabled=self.amp)

        self.dice_metric = DiceMetric(include_background=False, reduction="mean")
        self.post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=2)])
        self.post_label = Compose([EnsureType(), AsDiscrete(to_onehot=2)])

        # MLflow setup
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(experiment_name)

    def _train_epoch(self, epoch: int) -> float:
        """Run one training epoch.

        Args:
            epoch: Current epoch index.

        Returns:
            Average training loss for the epoch.
        """
        self.model.train()
        epoch_loss = 0.0
        step = 0

        for batch in self.train_loader:
            step += 1
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()

            with autocast(device_type=self.device.type, enabled=self.amp):
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)

            self.scaler.scale(loss).backward()

            if self.gradient_clip_max_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip_max_norm,
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()
            epoch_loss += loss.item()

        return epoch_loss / max(step, 1)

    @torch.no_grad()
    def _validate(self) -> float:
        """Run validation and compute Dice score.

        Returns:
            Mean Dice score across validation set.
        """
        self.model.eval()
        self.dice_metric.reset()

        for batch in self.val_loader:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            outputs = sliding_window_inference(
                images,
                roi_size=self.spatial_size,
                sw_batch_size=4,
                predictor=self.model,
                overlap=0.5,
            )

            outputs_list = decollate_batch(outputs)
            labels_list = decollate_batch(labels)

            outputs_post = [self.post_pred(o) for o in outputs_list]
            labels_post = [self.post_label(l) for l in labels_list]

            self.dice_metric(y_pred=outputs_post, y=labels_post)

        mean_dice = self.dice_metric.aggregate().item()
        self.dice_metric.reset()
        return mean_dice

    def train(self) -> dict[str, Any]:
        """Execute full training loop with MLflow tracking.

        Returns:
            Dictionary with training history.
        """
        history: dict[str, list[float]] = {
            "train_loss": [],
            "val_dice": [],
            "lr": [],
        }

        best_dice = 0.0
        best_model_path: str | None = None

        with mlflow.start_run():
            # Log hyperparameters
            mlflow.log_params(
                {
                    "epochs": self.epochs,
                    "batch_size": self.train_loader.batch_size,
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
                    "amp": self.amp,
                    "spatial_size": str(self.spatial_size),
                    "optimizer": self.optimizer.__class__.__name__,
                    "loss_fn": self.loss_fn.__class__.__name__,
                    "device": str(self.device),
                    "num_train_samples": len(self.train_loader.dataset),
                    "num_val_samples": len(self.val_loader.dataset),
                }
            )

            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("Training", total=self.epochs)

                for epoch in range(1, self.epochs + 1):
                    t0 = time.time()
                    train_loss = self._train_epoch(epoch)
                    history["train_loss"].append(train_loss)

                    current_lr = self.optimizer.param_groups[0]["lr"]
                    history["lr"].append(current_lr)

                    mlflow.log_metrics(
                        {"train_loss": train_loss, "learning_rate": current_lr},
                        step=epoch,
                    )

                    val_dice = 0.0
                    if epoch % self.val_interval == 0:
                        val_dice = self._validate()
                        history["val_dice"].append(val_dice)

                        mlflow.log_metric("val_dice", val_dice, step=epoch)

                        if val_dice > best_dice:
                            best_dice = val_dice
                            # Save best model
                            best_path = Path(self.checkpoint_manager.save_dir if self.checkpoint_manager else "./outputs") / "best_model.pth"
                            best_path.parent.mkdir(parents=True, exist_ok=True)
                            torch.save(self.model.state_dict(), best_path)
                            best_model_path = str(best_path)

                        if self.checkpoint_manager:
                            self.checkpoint_manager.save(
                                self.model, self.optimizer, epoch, val_dice
                            )

                        if self.early_stopping and self.early_stopping(val_dice):
                            logger.info(
                                f"Early stopping at epoch {epoch} with best Dice={best_dice:.4f}"
                            )
                            console.print(
                                f"[bold red]Early stopping at epoch {epoch}[/bold red]"
                            )
                            break

                    if self.scheduler is not None:
                        self.scheduler.step()

                    elapsed = time.time() - t0
                    progress.update(task, advance=1)

                    if epoch % self.val_interval == 0:
                        console.print(
                            f"  Epoch {epoch:3d}/{self.epochs} | "
                            f"Loss: {train_loss:.4f} | "
                            f"Val Dice: {val_dice:.4f} | "
                            f"Best: {best_dice:.4f} | "
                            f"LR: {current_lr:.2e} | "
                            f"Time: {elapsed:.1f}s"
                        )

            # Log final metrics and model artifact
            mlflow.log_metric("best_val_dice", best_dice)

            if best_model_path:
                mlflow.log_artifact(best_model_path)

        return {
            "history": history,
            "best_dice": best_dice,
            "best_model_path": best_model_path,
        }
