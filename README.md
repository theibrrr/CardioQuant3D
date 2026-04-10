# CardioQuant3D

**3D Cardiac Segmentation and Geometric Quantification Pipeline**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7-EE4C2C.svg)](https://pytorch.org/)
[![MONAI](https://img.shields.io/badge/MONAI-1.4-green.svg)](https://monai.io/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<p align="center">
  <img src="assets/git.gif" alt="CardioQuant3D Demo" width="720" />
</p>

---

## About This Project

CardioQuant3D is a 3D cardiac MRI segmentation and geometric quantification project designed with production practices in mind. The pipeline combines model training, geometric analysis, and deployment workflows with MLflow, Hydra, Docker, testing, and linting. It emphasizes reproducibility, modularity, and clear evaluation using both segmentation metrics (Dice, Hausdorff) and clinically meaningful geometric measurements (volume, surface area, long-axis, sphericity).

---


## 1. Project Overview

CardioQuant3D is an end-to-end pipeline for **3D cardiac MRI segmentation** and **geometric quantification** of the left ventricle (LV). It combines deep learning–based segmentation using a 3D U-Net (MONAI) with physically grounded geometric analysis to extract clinically meaningful metrics from cardiac MRI volumes.

The system processes NIfTI-format 3D cardiac MRI volumes, produces binary LV segmentation masks, converts them to 3D triangle meshes via Marching Cubes, and computes:

- **LV Volume** (ml)
- **Surface Area** (mm²)
- **Long-Axis Length** (PCA-based, mm)
- **Sphericity Index** (dimensionless)

All pipeline stages—from data loading through training, evaluation, inference, and deployment—are modular, config-driven, and reproducible.

---

## 2. Clinical Motivation

Accurate quantification of left ventricular geometry is critical for:

- Diagnosing dilated cardiomyopathy (DCM), hypertrophic cardiomyopathy (HCM), and other structural heart diseases
- Assessing cardiac remodeling post-myocardial infarction
- Monitoring treatment response and surgical planning
- Research in cardiac biomechanics and computational cardiology

Manual segmentation is time-consuming and operator-dependent. CardioQuant3D automates this process with a reproducible, clinically validated approach.

---

## 3. Architecture

```
                         ┌──────────────────────────────┐
                         │       NIfTI Input Volume      │
                         └──────────────┬───────────────┘
                                        │
                                        ▼
                         ┌──────────────────────────────┐
                         │    Preprocessing & Augment.   │
                         │  (MONAI Transforms Pipeline)  │
                         └──────────────┬───────────────┘
                                        │
                                        ▼
                         ┌──────────────────────────────┐
                         │       3D U-Net (MONAI)        │
                         │   Encoder → Decoder + Skip    │
                         │   [32, 64, 128, 256] channels │
                         └──────────────┬───────────────┘
                                        │
                              ┌─────────┴─────────┐
                              ▼                   ▼
                    ┌─────────────────┐ ┌─────────────────┐
                    │  Segmentation   │ │   Evaluation     │
                    │  Mask (Binary)  │ │ Dice / Hausdorff │
                    └────────┬────────┘ └─────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Marching Cubes  │
                    │  → 3D Mesh      │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  Geometric      │
                    │  Measurements   │
                    │ Vol / SA / LA   │
                    │ / Sphericity    │
                    └────────┬────────┘
                             │
                    ┌────────┴────────┐
                    ▼                 ▼
           ┌──────────────┐  ┌──────────────┐
           │   MLflow      │  │   FastAPI     │
           │   Tracking    │  │   Endpoint    │
           └──────────────┘  └──────────────┘
```

---

## 4. Dataset — ACDC (Automated Cardiac Diagnosis Challenge)

This project uses the **ACDC** dataset, a publicly available benchmark for cardiac MRI segmentation.

| Info | Detail |
|------|--------|
| **Source** | [ACDC Challenge — CREATIS Lab](https://www.creatis.insa-lyon.fr/Challenge/acdc/) |
| **Download** | Register on the ACDC challenge website to obtain the data |
| **Size** | 150 patients, ~7.4 GB (100 training + 50 testing) |
| **Format** | NIfTI (.nii / .nii.gz), short-axis cine MRI |
| **Labels** | 0 = Background, 1 = RV, 2 = Myocardium, **3 = LV (target)** |
| **Frames** | End-diastolic (ED) + End-systolic (ES) per patient |

> **Mandatory Citation:**
> O. Bernard, A. Lalande, C. Zotti, F. Cervenansky, et al.
> "Deep Learning Techniques for Automatic MRI Cardiac Multi-structures Segmentation and Diagnosis: Is the Problem Solved?"
> *IEEE Transactions on Medical Imaging*, vol. 37, no. 11, pp. 2514–2525, Nov. 2018.
> DOI: [10.1109/TMI.2018.2837502](https://doi.org/10.1109/TMI.2018.2837502)

Expected layout:

```
database/
├── training/
│   ├── patient001/
│   │   ├── Info.cfg
│   │   ├── patient001_4d.nii
│   │   ├── patient001_frame01.nii        # ED frame
│   │   ├── patient001_frame01_gt.nii     # ED ground truth
│   │   ├── patient001_frame12.nii        # ES frame
│   │   └── patient001_frame12_gt.nii     # ES ground truth
│   ├── patient002/
│   │   └── ...
│   └── patient100/
│
└── testing/
    ├── patient101/
    │   └── ...
    └── patient150/
```

**Label map:** 0 = Background, 1 = RV, 2 = Myocardium, 3 = LV (target)

---

## 5. Installation

### Option A: Conda (Recommended)

```bash
conda env create -f environment.yaml
conda activate cardioquant3d
pip install -e ".[dev]"
```

### Option B: pip

```bash
python -m venv cardioquant3d
source cardioquant3d/bin/activate  # Windows: cardioquant3d\Scripts\activate
pip install -e ".[dev]"
```

### PyTorch with GPU Support (Recommended)

By default, `pip install torch` installs the **CPU-only** build. For GPU-accelerated training, install PyTorch with CUDA support **after** the steps above:

```bash
# CUDA 11.8 (e.g., RTX 2060, RTX 3090, etc.)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 12.6
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

Verify GPU is detected:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: True NVIDIA GeForce RTX 2060
```

> **Note:** Choose the CUDA version that matches your locally installed NVIDIA driver.
> Check compatibility at https://pytorch.org/get-started/locally/

### Pre-commit hooks

```bash
pre-commit install
```

---

## 6. Training

```bash
# Default configuration
python scripts/train.py

# Override parameters via Hydra
python scripts/train.py training.epochs=100 training.batch_size=4 training.learning_rate=5e-4

# Use different data directory
python scripts/train.py data.root_dir=/path/to/acdc
```

Training features:
- Automatic mixed precision (AMP)
- Cosine annealing learning rate schedule
- Early stopping (patience=30)
- Top-k checkpoint saving
- MLflow experiment tracking
- Rich progress bars

---

## 7. Evaluation

```bash
# Evaluate on test set
python scripts/evaluate.py

# Custom checkpoint
python scripts/evaluate.py inference.checkpoint_path=./outputs/checkpoints/best.pth
```

Outputs:
- Per-sample Dice and Hausdorff distances
- Clinical metric errors (volume, surface area, long axis, sphericity)
- Summary statistics printed as Rich table
- Results saved to `outputs/evaluation_results.json`
- Metrics logged to MLflow

### Evaluation Results

The following results were obtained on the **ACDC test set** (patients 101–150, 100 samples: 50 ED + 50 ES frames) using the default configuration:

| Setting | Value |
|---------|-------|
| Epochs | 100 |
| Batch Size | 4 |
| Learning Rate | 1e-3 (AdamW, cosine annealing) |
| Model | 3D U-Net [32, 64, 128, 256] channels |
| Spatial Size | 128 × 128 × 32 |
| Pixel Spacing | 1.5 × 1.5 × 3.0 mm |
| Loss | Dice + CE (equal weight) |
| AMP | Enabled |

#### Segmentation Metrics

| Metric | Mean | Std | Median | Min | Max |
|--------|------|-----|--------|-----|-----|
| **Dice Score** | 0.876 | 0.072 | 0.904 | 0.605 | 0.954 |
| **Hausdorff-95 (mm)** | 5.51 | 9.94 | 3.00 | 1.50 | 76.89 |

#### Clinical Metric Errors

| Metric | Mean | Median |
|--------|------|--------|
| **Volume Error (ml)** | 3.22 | 2.42 |
| **Volume Error (%)** | 13.0% | 6.8% |

#### Distribution Analysis

| Threshold | Count | Percentage |
|-----------|-------|------------|
| Dice ≥ 0.90 | 54 / 100 | 54.0% |
| Dice ≥ 0.85 | 72 / 100 | 72.0% |
| Dice < 0.75 | 7 / 100 | 7.0% |
| HD95 ≤ 3 mm | 60 / 100 | 60.0% |

#### Observations

- **Overall performance is strong for 100 epochs of training.** The median Dice of 0.904 indicates that more than half of predictions have clinically acceptable segmentation quality.
- **Small LV volumes are the primary failure mode.** The worst-performing samples (Dice < 0.75) all have ground-truth volumes below 15 ml — these are typically end-systolic (ES) frames where the LV cavity is at its smallest. A few voxels of error on such small structures causes large relative degradation.
- **Outlier Hausdorff values** (76.9 mm, 51.3 mm) inflate the mean HD95. The median of 3.0 mm is a more representative figure — 60% of samples have surface errors within a single voxel.
- **Volume error median of 6.8%** is within the clinically acceptable range (typically < 10–15%). The mean is pulled higher by outliers on small-volume ES frames.
- **Compared to ACDC challenge state-of-the-art** (LV Dice ~0.93–0.95), these results are reasonable for a baseline 100-epoch run. Further training (200–300 epochs), post-processing, and test-time augmentation could close this gap.

---

## 8. MLflow Experiment Tracking

CardioQuant3D uses [MLflow](https://mlflow.org/) to log every aspect of each run — metrics, hyperparameters, and model artifacts — so that experiments are reproducible and comparable.

```bash
# Launch the MLflow dashboard
mlflow ui --backend-store-uri file:./mlruns

# Open in browser → http://localhost:5000
```

**What gets logged automatically:**

| Stage | Logged Items |
|-------|-------------|
| **Training** | `train_loss`, `val_dice`, `learning_rate` per epoch; all hyperparameters from Hydra config; best model checkpoint as artifact |
| **Evaluation** | `mean_dice`, `mean_hd95`, `mean_volume_error` on the test set; per-sample results JSON |

Each `python scripts/train.py` invocation creates a new MLflow run under the experiment name defined in `experiment.name` (default: `cardioquant3d`). You can compare runs side by side in the MLflow UI, filter by metric thresholds, and download artifacts.

> **Tip:** All tracking data lives in `mlruns/`  (local file store). No remote server required — simply point `mlflow ui` at it.

---

## 9. Inference

### CLI

```bash
# Single file inference
python scripts/infer.py --input /path/to/cardiac_mri.nii.gz

# Custom output directory
python scripts/infer.py --input scan.nii.gz --output-dir ./results
```

### FastAPI

```bash
# Start server
make api
# or
uvicorn cardioquant3d.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Open **http://localhost:8000** for the interactive upload page.

#### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Upload page — drag & drop a NIfTI file for visual analysis |
| `GET` | `/health` | Health check (returns service status and whether the model is loaded) |
| `POST` | `/analyze` | JSON API — returns clinical metrics as JSON |
| `POST` | `/visualize` | HTML report — MRI slices with LV overlay + metric cards |
| `GET` | `/compare` | Two-file upload page (MRI + ground truth mask) |
| `POST` | `/compare` | HTML report — prediction vs ground truth overlay with Dice score |
| `GET` | `/docs` | Swagger UI (auto-generated) |

**Health check response fields:**

- `status`: service health
- `model_loaded`: whether the model has been loaded by the API

**Health check example:**

```json
{
  "status": "ok",
  "model_loaded": true
}
```

**JSON API example:**

```bash
curl -X POST "http://localhost:8000/analyze" \
     -H "accept: application/json" \
     -F "file=@cardiac_mri.nii.gz"
```

```json
{
    "lv_volume_ml": 142.35,
    "surface_area_mm2": 12450.80,
    "long_axis_mm": 87.42,
    "sphericity_index": 0.6821
}
```

---

## 10. Docker Usage

```bash
# Build image
make docker-build
# or
docker build -t cardioquant3d:latest .

# Run container
docker run -p 8000:8000 \
    -v ./outputs:/app/outputs \
    -e CARDIOQUANT3D_CHECKPOINT=/app/outputs/checkpoints/best_model.pth \
    cardioquant3d:latest

# Test
curl http://localhost:8000/health
```

---

## 11. Development

```bash
# Run all quality checks
make lint

# Run tests
make test

# Run tests with coverage
make test-cov

# Format code
black .
isort .
```

---

## 12. Configuration Reference

All settings are managed via [Hydra](https://hydra.cc/) YAML files under `configs/`. Override any value from the CLI:

```bash
python scripts/train.py training.epochs=200 data.cache_rate=0.5 augmentation.flip_prob=0.3
```

### Data (`configs/config.yaml` → `data`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data.root_dir` | str | `"./database"` | Root directory of the ACDC dataset |
| `data.train_dir` | str | `"${data.root_dir}/training"` | Training data directory |
| `data.test_dir` | str | `"${data.root_dir}/testing"` | Testing data directory |
| `data.target_label` | int | `3` | Label index for segmentation target (3 = LV) |
| `data.spatial_size` | list[int] | `[128, 128, 32]` | Target spatial dimensions [H, W, D] for resampling |
| `data.pixel_spacing` | list[float] | `[1.5, 1.5, 3.0]` | Resampling target voxel spacing in mm |
| `data.orientation` | str | `"RAS"` | Anatomical orientation code (Right-Anterior-Superior) |
| `data.cache_rate` | float | `1.0` | Fraction of dataset to cache in RAM (0.0–1.0). Higher = faster training, more memory |
| `data.num_workers` | int | `4` | Number of parallel data-loading workers |

### Preprocessing (`configs/config.yaml` → `preprocessing`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `preprocessing.intensity_lower_percentile` | float | `0.5` | Lower percentile for intensity clipping |
| `preprocessing.intensity_upper_percentile` | float | `99.5` | Upper percentile for intensity clipping |
| `preprocessing.intensity_output_min` | float | `0.0` | Minimum output value after intensity scaling |
| `preprocessing.intensity_output_max` | float | `1.0` | Maximum output value after intensity scaling |

### Augmentation (`configs/config.yaml` → `augmentation`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `augmentation.flip_prob` | float | `0.5` | Probability for random axis flips |
| `augmentation.rotate90_prob` | float | `0.5` | Probability for random 90° rotations |
| `augmentation.affine_prob` | float | `0.3` | Probability for random affine transforms |
| `augmentation.affine_rotate_range` | list[float] | `[0.1, 0.1, 0.1]` | Rotation range per axis (radians) |
| `augmentation.affine_scale_range` | list[float] | `[0.1, 0.1, 0.1]` | Scale range per axis |
| `augmentation.gaussian_noise_prob` | float | `0.2` | Probability for additive Gaussian noise |
| `augmentation.gaussian_noise_std` | float | `0.05` | Standard deviation of Gaussian noise |
| `augmentation.gaussian_smooth_prob` | float | `0.2` | Probability for Gaussian smoothing |
| `augmentation.gaussian_smooth_sigma_range` | list[float] | `[0.5, 1.0]` | Sigma range for Gaussian smoothing kernel |
| `augmentation.intensity_scale_prob` | float | `0.3` | Probability for random intensity scaling |
| `augmentation.intensity_scale_factors` | float | `0.1` | Scale factor for intensity augmentation |
| `augmentation.intensity_shift_prob` | float | `0.3` | Probability for random intensity shift |
| `augmentation.intensity_shift_offsets` | float | `0.1` | Offset for intensity shift augmentation |

### Model (`configs/model/model.yaml`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model.architecture` | str | `"unet3d"` | Model architecture identifier |
| `model.in_channels` | int | `1` | Number of input channels (1 for grayscale MRI) |
| `model.out_channels` | int | `2` | Number of output classes (background + LV) |
| `model.channels` | list[int] | `[32, 64, 128, 256]` | Feature channels per encoder level |
| `model.strides` | list[int] | `[2, 2, 2]` | Down-sampling strides between levels |
| `model.num_res_units` | int | `2` | Residual units per level |
| `model.dropout` | float | `0.2` | Dropout rate (set to 0 at inference) |
| `model.norm` | str | `"batch"` | Normalization type (`"batch"`, `"instance"`, `"group"`) |

### Training (`configs/training/training.yaml`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `training.epochs` | int | `100` | Maximum number of training epochs |
| `training.batch_size` | int | `4` | Mini-batch size |
| `training.learning_rate` | float | `1e-3` | Initial learning rate for AdamW |
| `training.weight_decay` | float | `1e-5` | L2 regularization weight |
| `training.optimizer` | str | `"adamw"` | Optimizer name |
| `training.scheduler` | str | `"cosine"` | LR scheduler type |
| `training.scheduler_eta_min` | float | `1e-7` | Minimum LR for cosine annealing |
| `training.warmup_epochs` | int | `10` | Number of warmup epochs |
| `training.amp` | bool | `true` | Enable automatic mixed precision |
| `training.gradient_clip_max_norm` | float | `1.0` | Max norm for gradient clipping |
| `training.val_split` | float | `0.2` | Fraction of training data used for validation |
| `training.val_interval` | int | `2` | Validate every N epochs |

### Loss (`configs/training/training.yaml` → `training.loss`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `training.loss.lambda_dice` | float | `1.0` | Weight of Dice loss component |
| `training.loss.lambda_ce` | float | `1.0` | Weight of Cross-Entropy loss component |
| `training.loss.include_background` | bool | `false` | Include background class in loss computation |
| `training.loss.softmax` | bool | `true` | Apply softmax to model outputs |

### Early Stopping (`configs/training/training.yaml` → `training.early_stopping`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `training.early_stopping.enabled` | bool | `true` | Enable early stopping |
| `training.early_stopping.patience` | int | `30` | Epochs without improvement before stopping |
| `training.early_stopping.min_delta` | float | `0.001` | Minimum improvement threshold |
| `training.early_stopping.monitor` | str | `"val_dice"` | Metric to monitor |

### Checkpointing (`configs/training/training.yaml` → `training.checkpointing`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `training.checkpointing.enabled` | bool | `true` | Enable checkpoint saving |
| `training.checkpointing.save_top_k` | int | `3` | Number of best checkpoints to keep |
| `training.checkpointing.monitor` | str | `"val_dice"` | Metric to monitor for best selection |
| `training.checkpointing.mode` | str | `"max"` | `"max"` or `"min"` for monitor metric |

### Inference (`configs/inference/inference.yaml`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `inference.checkpoint_path` | str | `"./outputs/checkpoints/best_model.pth"` | Path to trained model checkpoint |
| `inference.device` | str | `"auto"` | Device selection (`"auto"`, `"cuda"`, `"cpu"`) |
| `inference.sw_batch_size` | int | `4` | Sliding window inference batch size |
| `inference.overlap` | float | `0.5` | Sliding window overlap fraction (0.0–1.0) |
| `inference.output_dir` | str | `"./outputs/predictions"` | Directory for inference outputs |

### Experiment (`configs/config.yaml` → `experiment`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `experiment.name` | str | `"cardioquant3d"` | MLflow experiment name |
| `experiment.seed` | int | `42` | Global random seed for reproducibility |
| `experiment.deterministic` | bool | `true` | Enable deterministic CUDA operations |
| `experiment.output_dir` | str | `"./outputs"` | Root output directory |
| `experiment.mlflow_tracking_uri` | str | `"file:./mlruns"` | MLflow tracking URI |

---

## 13. Repository Structure

```
CardioQuant3D/
├── assets/                   # Demo GIF and visual assets
├── configs/                  # Hydra YAML configurations
│   ├── config.yaml           # Main config
│   ├── model/model.yaml      # 3D U-Net architecture
│   ├── training/training.yaml# Training hyperparameters
│   └── inference/inference.yaml
├── cardioquant3d/            # Python package
│   ├── data/                 # Dataset, transforms, preprocessing
│   ├── models/               # 3D U-Net implementation
│   ├── training/             # Trainer, losses
│   ├── evaluation/           # Dice, Hausdorff, clinical metrics
│   ├── geometry/             # Mesh generation, measurements
│   ├── inference/            # Predictor engine
│   ├── api/                  # FastAPI application
│   └── utils/                # I/O, logging, seed
├── scripts/                  # Entry-point scripts
│   ├── train.py
│   ├── evaluate.py
│   └── infer.py
├── tests/                    # Unit tests (pytest)
├── CHANGELOG.md
├── CONTRIBUTING.md
├── Dockerfile
├── Makefile
├── .pre-commit-config.yaml
├── environment.yaml
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## 14. Citation

If you use the ACDC dataset, **you must cite the original paper:**

```bibtex
@article{bernard2018deep,
  title={Deep Learning Techniques for Automatic MRI Cardiac Multi-structures Segmentation and Diagnosis: Is the Problem Solved?},
  author={Bernard, Olivier and Lalande, Alain and Zotti, Cl{\'e}ment and Cervenansky, Fr{\'e}d{\'e}ric and others},
  journal={IEEE Transactions on Medical Imaging},
  volume={37},
  number={11},
  pages={2514--2525},
  year={2018},
  publisher={IEEE},
  doi={10.1109/TMI.2018.2837502}
}
```

Dataset download: [https://www.creatis.insa-lyon.fr/Challenge/acdc/](https://www.creatis.insa-lyon.fr/Challenge/acdc/)

---

## 15. Future Improvements

| Area | Improvement | Expected Impact |
|------|------------|-----------------|
| **Training** | Extend to 200–300 epochs with test-time augmentation (TTA) | Close the gap to SOTA Dice ~0.93–0.95 |
| **Architecture** | Experiment with nnU-Net, Swin UNETR, or attention-gated U-Net | Better capture of small LV at ES frames |
| **Multi-structure** | Extend segmentation to RV + Myocardium (all 3 labels) | Broader clinical utility (EF, myocardial mass) |
| **Post-processing** | Connected-component filtering + morphological refinement | Remove spurious small predictions, smoother contours |
| **Data** | Add M&Ms, CAMUS, or UK Biobank datasets for cross-domain generalization | Improve robustness on unseen scanners/protocols |
| **Metrics** | Compute Ejection Fraction (EF) from ED/ES volume pairs | Most common clinical metric in cardiology |
| **Deployment** | Add ONNX export + TensorRT optimization | 5–10× faster inference on GPU |
| **API** | WebSocket-based progress for large files; batch upload support | Better UX for clinical workflows |
| **Quality** | Add model regression tests — Dice on a held-out set must not drop | Prevent silent model degradation |
| **Visualization** | 3D mesh rendering (VTK / PyVista) in the web UI | Interactive 3D exploration of the segmentation |

---


**License:** MIT
