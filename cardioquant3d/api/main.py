"""FastAPI application for CardioQuant3D inference."""

from __future__ import annotations

import base64
import io
import logging
import os
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel, Field

from monai.transforms import (
    Compose as MonaiCompose,
    EnsureTyped,
    Lambdad,
    LoadImaged,
    Orientationd,
    Resized,
    ScaleIntensityRangePercentilesd,
    Spacingd,
    SpatialPadd,
)

from cardioquant3d.data.transforms import _BinarizeLabelTransform
from cardioquant3d.inference.predictor import Predictor

logger = logging.getLogger(__name__)

app = FastAPI(
    title="CardioQuant3D",
    description="3D Cardiac Segmentation and Geometric Quantification API",
    version="1.0.0",
)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def upload_page():
    """Serve the upload form for visual analysis."""
    return HTMLResponse(content="""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>CardioQuant3D — Upload</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { background: #0f0f23; color: #e0e0e0; font-family: 'Segoe UI', Arial, sans-serif;
               display: flex; justify-content: center; align-items: center; min-height: 100vh; }
        .card { background: linear-gradient(135deg, #1a1a3e, #16213e); border: 1px solid #2a2a5a;
                border-radius: 16px; padding: 50px; text-align: center; max-width: 500px; width: 90%; }
        h1 { color: #6EC8A0; font-size: 28px; margin-bottom: 8px; }
        .sub { color: #888; font-size: 14px; margin-bottom: 30px; }
        .drop-zone { border: 2px dashed #2a2a5a; border-radius: 12px; padding: 40px 20px;
                     cursor: pointer; transition: all 0.3s; margin-bottom: 20px; }
        .drop-zone:hover, .drop-zone.drag-over { border-color: #6EC8A0; background: rgba(110,200,160,0.06); }
        .drop-zone p { color: #aaa; font-size: 15px; }
        .drop-zone .icon { font-size: 40px; margin-bottom: 10px; }
        .file-name { color: #6EC8A0; font-size: 13px; margin-top: 8px; display: none; }
        button { background: #6EC8A0; color: #0f0f23; border: none; border-radius: 8px;
                 padding: 14px 40px; font-size: 16px; font-weight: bold; cursor: pointer;
                 transition: all 0.3s; width: 100%; }
        button:hover { background: #5CB28D; }
        button:disabled { background: #333; color: #666; cursor: not-allowed; }
        .spinner { display: none; margin: 20px auto; }
        .spinner.active { display: block; }
        .spinner::after { content: ''; display: block; width: 40px; height: 40px; margin: 0 auto;
                          border: 4px solid #2a2a5a; border-top-color: #6EC8A0; border-radius: 50%;
                          animation: spin 0.8s linear infinite; }
        @keyframes spin { to { transform: rotate(360deg); } }
        .links { margin-top: 20px; font-size: 12px; }
        .links a { color: #6EC8A0; text-decoration: none; }
    </style>
</head>
<body>
    <div class="card">
        <h1>CardioQuant3D</h1>
        <p class="sub">3D Cardiac Segmentation &amp; Geometric Quantification</p>
        <form id="form" action="/visualize" method="post" enctype="multipart/form-data">
            <div class="drop-zone" id="dropZone" onclick="document.getElementById('fileInput').click()">
                <div class="icon">&#128156;</div>
                <p>Drop a NIfTI file here or click to browse</p>
                <p style="font-size:12px; color:#666; margin-top:6px">.nii or .nii.gz</p>
                <div class="file-name" id="fileName"></div>
            </div>
            <input type="file" name="file" id="fileInput" accept=".nii,.nii.gz,.gz" style="display:none">
            <button type="submit" id="btn" disabled>Analyze</button>
            <div class="spinner" id="spinner"></div>
        </form>
        <div class="links">
            <a href="/docs">Swagger UI</a> &nbsp;|&nbsp; <a href="/health">Health Check</a> &nbsp;|&nbsp; <a href="/compare">Compare with GT</a>
        </div>
    </div>
    <script>
        const dropZone = document.getElementById('dropZone');
        const fileInput = document.getElementById('fileInput');
        const fileName = document.getElementById('fileName');
        const btn = document.getElementById('btn');
        const spinner = document.getElementById('spinner');
        const form = document.getElementById('form');

        fileInput.addEventListener('change', () => {
            if (fileInput.files.length) {
                fileName.textContent = fileInput.files[0].name;
                fileName.style.display = 'block';
                btn.disabled = false;
            }
        });

        ['dragenter','dragover'].forEach(e => {
            dropZone.addEventListener(e, ev => { ev.preventDefault(); dropZone.classList.add('drag-over'); });
        });
        ['dragleave','drop'].forEach(e => {
            dropZone.addEventListener(e, ev => { ev.preventDefault(); dropZone.classList.remove('drag-over'); });
        });
        dropZone.addEventListener('drop', ev => {
            fileInput.files = ev.dataTransfer.files;
            fileInput.dispatchEvent(new Event('change'));
        });

        form.addEventListener('submit', () => {
            btn.disabled = true;
            btn.textContent = 'Analyzing...';
            spinner.classList.add('active');
        });
    </script>
</body>
</html>""")


class AnalysisResponse(BaseModel):
    """Response schema for cardiac analysis endpoint."""

    lv_volume_ml: float = Field(..., description="Left ventricle volume in milliliters")
    surface_area_mm2: float = Field(..., description="LV surface area in square millimeters")
    long_axis_mm: float = Field(..., description="LV long-axis length in millimeters")
    sphericity_index: float = Field(..., description="LV sphericity index (dimensionless)")


class HealthResponse(BaseModel):
    """Response schema for health check."""

    status: str
    model_loaded: bool


# Global predictor — loaded once at startup
_predictor: Predictor | None = None


def get_predictor() -> Predictor:
    """Get or initialize the global predictor.

    Returns:
        Initialized Predictor instance.

    Raises:
        HTTPException: If model checkpoint is not found.
    """
    global _predictor

    if _predictor is not None:
        return _predictor

    checkpoint_path = os.environ.get(
        "CARDIOQUANT3D_CHECKPOINT",
        "./outputs/best_model.pth",
    )

    if not Path(checkpoint_path).exists():
        raise HTTPException(
            status_code=503,
            detail=f"Model checkpoint not found at {checkpoint_path}. "
            "Set CARDIOQUANT3D_CHECKPOINT environment variable.",
        )

    _predictor = Predictor.from_checkpoint(checkpoint_path)
    logger.info(f"Model loaded from {checkpoint_path}")

    return _predictor


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        model_loaded=_predictor is not None,
    )


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(file: UploadFile = File(...)) -> AnalysisResponse:
    """Analyze a cardiac MRI NIfTI file.

    Accepts a .nii or .nii.gz file and returns geometric quantification
    of the left ventricle.

    Args:
        file: Uploaded NIfTI file.

    Returns:
        AnalysisResponse with LV measurements.

    Raises:
        HTTPException: On invalid file or processing error.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    if not (file.filename.endswith(".nii") or file.filename.endswith(".nii.gz")):
        raise HTTPException(
            status_code=400,
            detail="Only .nii and .nii.gz files are accepted.",
        )

    predictor = get_predictor()

    # Save uploaded file to temp location
    suffix = ".nii.gz" if file.filename.endswith(".nii.gz") else ".nii"
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            content = await file.read()
            if len(content) == 0:
                raise HTTPException(status_code=400, detail="Uploaded file is empty.")
            tmp.write(content)
            tmp_path = tmp.name

        metrics, _, _ = predictor.analyze(tmp_path)

        return AnalysisResponse(
            lv_volume_ml=round(metrics.lv_volume_ml, 2),
            surface_area_mm2=round(metrics.surface_area_mm2, 2),
            long_axis_mm=round(metrics.long_axis_mm, 2),
            sphericity_index=round(metrics.sphericity_index, 4),
        )

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("Analysis failed")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except (OSError, UnboundLocalError):
            pass


def _build_overlay_figure(image_data: np.ndarray, mask: np.ndarray) -> str:
    """Create a matplotlib figure with MRI slices and segmentation overlay.

    Both image_data and mask must be in the same spatial space (inference space).

    Returns base64-encoded PNG string.
    """
    n_slices = image_data.shape[2]

    # Pick 5 representative axial slices through the volume
    slice_indices = np.linspace(n_slices * 0.2, n_slices * 0.8, 5, dtype=int)
    # Prefer slices with most LV content
    lv_per_slice = [mask[:, :, s].sum() for s in range(n_slices)]
    if max(lv_per_slice) > 0:
        top_slices = sorted(range(n_slices), key=lambda s: lv_per_slice[s], reverse=True)[:5]
        slice_indices = sorted(top_slices)

    fig, axes = plt.subplots(2, 5, figsize=(20, 8), facecolor="#1a1a2e")

    for col, s_idx in enumerate(slice_indices):
        img_slice = image_data[:, :, s_idx]
        mask_slice = mask[:, :, s_idx]

        # Top row: original MRI
        axes[0, col].imshow(img_slice.T, cmap="gray", origin="lower")
        axes[0, col].set_title(f"Slice {s_idx}", color="white", fontsize=11)
        axes[0, col].axis("off")

        # Bottom row: MRI + overlay
        axes[1, col].imshow(img_slice.T, cmap="gray", origin="lower")
        if mask_slice.sum() > 0:
            overlay = np.ma.masked_where(mask_slice.T < 0.5, mask_slice.T)
            axes[1, col].imshow(overlay, cmap="autumn", alpha=0.5, origin="lower")
            # Draw contour
            axes[1, col].contour(mask_slice.T, levels=[0.5], colors=["#6EC8A0"], linewidths=1.5)
        axes[1, col].set_title("+ LV Mask", color="#6EC8A0", fontsize=11)
        axes[1, col].axis("off")

    fig.suptitle("CardioQuant3D — Segmentation Result", color="white", fontsize=16, fontweight="bold")
    axes[0, 0].text(-0.1, 0.5, "Original", transform=axes[0, 0].transAxes,
                     color="white", fontsize=12, va="center", ha="right", rotation=90)
    axes[1, 0].text(-0.1, 0.5, "Segmented", transform=axes[1, 0].transAxes,
                     color="#6EC8A0", fontsize=12, va="center", ha="right", rotation=90)
    plt.tight_layout(rect=[0.02, 0, 1, 0.95])

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


@app.post("/visualize", response_class=HTMLResponse)
async def visualize(file: UploadFile = File(...)) -> HTMLResponse:
    """Analyze a cardiac MRI and return an interactive HTML report.

    Shows segmentation overlay on MRI slices alongside clinical metrics.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")
    if not (file.filename.endswith(".nii") or file.filename.endswith(".nii.gz")):
        raise HTTPException(status_code=400, detail="Only .nii and .nii.gz files are accepted.")

    predictor = get_predictor()
    suffix = ".nii.gz" if file.filename.endswith(".nii.gz") else ".nii"

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            content = await file.read()
            if len(content) == 0:
                raise HTTPException(status_code=400, detail="Uploaded file is empty.")
            tmp.write(content)
            tmp_path = tmp.name

        metrics, mask, processed_image = predictor.analyze(tmp_path)
        img_b64 = _build_overlay_figure(processed_image, mask)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>CardioQuant3D — Analysis Result</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ background: #0f0f23; color: #e0e0e0; font-family: 'Segoe UI', Arial, sans-serif; padding: 30px; }}
        h1 {{ color: #6EC8A0; text-align: center; margin-bottom: 8px; font-size: 28px; }}
        .subtitle {{ text-align: center; color: #888; margin-bottom: 30px; font-size: 14px; }}
        .container {{ max-width: 1100px; margin: 0 auto; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 30px; }}
        .metric-card {{
            background: linear-gradient(135deg, #1a1a3e, #16213e);
            border: 1px solid #2a2a5a; border-radius: 12px; padding: 20px; text-align: center;
        }}
        .metric-card .value {{ font-size: 28px; font-weight: bold; color: #6EC8A0; margin-bottom: 4px; }}
        .metric-card .label {{ font-size: 13px; color: #aaa; }}
        .metric-card .unit {{ font-size: 12px; color: #666; }}
        .visual {{ text-align: center; }}
        .visual img {{ max-width: 100%; border-radius: 12px; border: 1px solid #2a2a5a; }}
        .filename {{ text-align: center; color: #666; margin-top: 15px; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>CardioQuant3D</h1>
        <p class="subtitle">3D Cardiac Segmentation &amp; Geometric Quantification — {file.filename}</p>

        <div class="metrics-grid">
            <div class="metric-card">
                <div class="value">{metrics.lv_volume_ml:.2f}</div>
                <div class="label">LV Volume</div>
                <div class="unit">ml</div>
            </div>
            <div class="metric-card">
                <div class="value">{metrics.surface_area_mm2:.1f}</div>
                <div class="label">Surface Area</div>
                <div class="unit">mm²</div>
            </div>
            <div class="metric-card">
                <div class="value">{metrics.long_axis_mm:.2f}</div>
                <div class="label">Long Axis</div>
                <div class="unit">mm</div>
            </div>
            <div class="metric-card">
                <div class="value">{metrics.sphericity_index:.4f}</div>
                <div class="label">Sphericity Index</div>
                <div class="unit">dimensionless</div>
            </div>
        </div>

        <div class="visual">
            <img src="data:image/png;base64,{img_b64}" alt="Segmentation Overlay" />
        </div>

        <p class="filename">Input: {file.filename}</p>
    </div>
</body>
</html>"""

        return HTMLResponse(content=html)

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("Visualization failed")
        raise HTTPException(status_code=500, detail=f"Visualization failed: {str(e)}")
    finally:
        try:
            os.unlink(tmp_path)
        except (OSError, UnboundLocalError):
            pass


# ─── Compare with Ground Truth ──────────────────────────────────


@app.get("/compare", response_class=HTMLResponse, include_in_schema=False)
async def compare_page():
    """Serve the compare upload form."""
    return HTMLResponse(content="""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>CardioQuant3D — Compare with GT</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { background: #0f0f23; color: #e0e0e0; font-family: 'Segoe UI', Arial, sans-serif;
               display: flex; justify-content: center; align-items: center; min-height: 100vh; }
        .card { background: linear-gradient(135deg, #1a1a3e, #16213e); border: 1px solid #2a2a5a;
                border-radius: 16px; padding: 50px; text-align: center; max-width: 550px; width: 90%; }
        h1 { color: #6EC8A0; font-size: 28px; margin-bottom: 8px; }
        .sub { color: #888; font-size: 14px; margin-bottom: 30px; }
        .upload-row { display: flex; gap: 16px; margin-bottom: 20px; }
        .drop-zone { flex: 1; border: 2px dashed #2a2a5a; border-radius: 12px; padding: 24px 12px;
                     cursor: pointer; transition: all 0.3s; }
        .drop-zone:hover, .drop-zone.drag-over { border-color: #6EC8A0; background: rgba(110,200,160,0.06); }
        .drop-zone .icon { font-size: 30px; margin-bottom: 6px; }
        .drop-zone p { color: #aaa; font-size: 13px; }
        .file-name { color: #6EC8A0; font-size: 11px; margin-top: 6px; word-break: break-all; display: none; }
        button { background: #6EC8A0; color: #0f0f23; border: none; border-radius: 8px;
                 padding: 14px 40px; font-size: 16px; font-weight: bold; cursor: pointer;
                 transition: all 0.3s; width: 100%; }
        button:hover { background: #5CB28D; }
        button:disabled { background: #333; color: #666; cursor: not-allowed; }
        .spinner { display: none; margin: 20px auto; }
        .spinner.active { display: block; }
        .spinner::after { content: ''; display: block; width: 40px; height: 40px; margin: 0 auto;
                          border: 4px solid #2a2a5a; border-top-color: #6EC8A0; border-radius: 50%;
                          animation: spin 0.8s linear infinite; }
        @keyframes spin { to { transform: rotate(360deg); } }
        .links { margin-top: 20px; font-size: 12px; }
        .links a { color: #6EC8A0; text-decoration: none; }
        .label-tag { display: inline-block; background: #6EC8A022; color: #6EC8A0; border-radius: 4px;
                     padding: 2px 8px; font-size: 11px; margin-bottom: 8px; }
        .label-tag.gt { background: #D4956B22; color: #D4956B; }
    </style>
</head>
<body>
    <div class="card">
        <h1>CardioQuant3D</h1>
        <p class="sub">Compare Prediction vs Ground Truth</p>
        <form id="form" action="/compare" method="post" enctype="multipart/form-data">
            <div class="upload-row">
                <div class="drop-zone" id="dz1" onclick="document.getElementById('f1').click()">
                    <span class="label-tag">MRI Image</span>
                    <div class="icon">&#129657;</div>
                    <p>patient_frameXX.nii</p>
                    <div class="file-name" id="fn1"></div>
                </div>
                <div class="drop-zone" id="dz2" onclick="document.getElementById('f2').click()">
                    <span class="label-tag gt">Ground Truth</span>
                    <div class="icon">&#127919;</div>
                    <p>patient_frameXX_gt.nii</p>
                    <div class="file-name" id="fn2"></div>
                </div>
            </div>
            <input type="file" name="image" id="f1" accept=".nii,.nii.gz,.gz" style="display:none">
            <input type="file" name="ground_truth" id="f2" accept=".nii,.nii.gz,.gz" style="display:none">
            <button type="submit" id="btn" disabled>Compare</button>
            <div class="spinner" id="spinner"></div>
        </form>
        <div class="links">
            <a href="/">&#8592; Back to Analyze</a> &nbsp;|&nbsp; <a href="/docs">Swagger UI</a>
        </div>
    </div>
    <script>
        const f1=document.getElementById('f1'), f2=document.getElementById('f2');
        const fn1=document.getElementById('fn1'), fn2=document.getElementById('fn2');
        const btn=document.getElementById('btn'), form=document.getElementById('form');
        const spinner=document.getElementById('spinner');
        function check() { btn.disabled = !(f1.files.length && f2.files.length); }
        f1.addEventListener('change', () => { if(f1.files.length){fn1.textContent=f1.files[0].name;fn1.style.display='block';} check(); });
        f2.addEventListener('change', () => { if(f2.files.length){fn2.textContent=f2.files[0].name;fn2.style.display='block';} check(); });

        ['dz1','dz2'].forEach((id,i) => {
            const dz=document.getElementById(id), fi=[f1,f2][i];
            ['dragenter','dragover'].forEach(e=>dz.addEventListener(e,ev=>{ev.preventDefault();dz.classList.add('drag-over');}));
            ['dragleave','drop'].forEach(e=>dz.addEventListener(e,ev=>{ev.preventDefault();dz.classList.remove('drag-over');}));
            dz.addEventListener('drop',ev=>{fi.files=ev.dataTransfer.files;fi.dispatchEvent(new Event('change'));});
        });

        form.addEventListener('submit', () => { btn.disabled=true; btn.textContent='Comparing...'; spinner.classList.add('active'); });
    </script>
</body>
</html>""")


def _build_comparison_figure(
    image_data: np.ndarray, pred_mask: np.ndarray, gt_mask: np.ndarray,
) -> str:
    """Create a 3-row comparison figure: Original / Prediction / Ground Truth.

    All three arrays must already be in the same spatial space (inference space).

    Returns base64-encoded PNG string.
    """
    n_slices = image_data.shape[2]

    # Pick slices with most combined LV content
    combined = pred_mask + gt_mask
    per_slice = [combined[:, :, s].sum() for s in range(n_slices)]
    if max(per_slice) > 0:
        top = sorted(range(n_slices), key=lambda s: per_slice[s], reverse=True)[:5]
        slice_indices = sorted(top)
    else:
        slice_indices = np.linspace(n_slices * 0.2, n_slices * 0.8, 5, dtype=int).tolist()

    fig, axes = plt.subplots(3, 5, figsize=(20, 12), facecolor="#1a1a2e")

    for col, s_idx in enumerate(slice_indices):
        img_s = image_data[:, :, s_idx]
        pred_s = pred_mask[:, :, s_idx]
        gt_s = gt_mask[:, :, s_idx]

        # Row 0: Original MRI
        axes[0, col].imshow(img_s.T, cmap="gray", origin="lower")
        axes[0, col].set_title(f"Slice {s_idx}", color="white", fontsize=11)
        axes[0, col].axis("off")

        # Row 1: Prediction overlay (green)
        axes[1, col].imshow(img_s.T, cmap="gray", origin="lower")
        if pred_s.sum() > 0:
            ov = np.ma.masked_where(pred_s.T < 0.5, pred_s.T)
            axes[1, col].imshow(ov, cmap="Greens", alpha=0.45, origin="lower")
            axes[1, col].contour(pred_s.T, levels=[0.5], colors=["#6EC8A0"], linewidths=1.5)
        axes[1, col].set_title("Prediction", color="#6EC8A0", fontsize=11)
        axes[1, col].axis("off")

        # Row 2: Ground truth overlay (orange)
        axes[2, col].imshow(img_s.T, cmap="gray", origin="lower")
        if gt_s.sum() > 0:
            ov = np.ma.masked_where(gt_s.T < 0.5, gt_s.T)
            axes[2, col].imshow(ov, cmap="Oranges", alpha=0.45, origin="lower")
            axes[2, col].contour(gt_s.T, levels=[0.5], colors=["#D4956B"], linewidths=1.5)
        axes[2, col].set_title("Ground Truth", color="#D4956B", fontsize=11)
        axes[2, col].axis("off")

    fig.suptitle("CardioQuant3D — Prediction vs Ground Truth",
                 color="white", fontsize=16, fontweight="bold")
    for row, (label, color) in enumerate([
        ("Original", "white"), ("Prediction", "#6EC8A0"), ("Ground Truth", "#D4956B"),
    ]):
        axes[row, 0].text(-0.1, 0.5, label, transform=axes[row, 0].transAxes,
                          color=color, fontsize=12, va="center", ha="right", rotation=90)

    plt.tight_layout(rect=[0.02, 0, 1, 0.95])
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


@app.post("/compare", response_class=HTMLResponse)
async def compare(
    image: UploadFile = File(...),
    ground_truth: UploadFile = File(...),
) -> HTMLResponse:
    """Compare model prediction against ground truth segmentation."""
    for f in (image, ground_truth):
        if not f.filename:
            raise HTTPException(status_code=400, detail="No filename provided.")
        if not (f.filename.endswith(".nii") or f.filename.endswith(".nii.gz")):
            raise HTTPException(status_code=400, detail="Only .nii and .nii.gz files are accepted.")

    predictor = get_predictor()
    tmp_paths = []

    try:
        # Save uploaded files
        for f in (image, ground_truth):
            suffix = ".nii.gz" if f.filename.endswith(".nii.gz") else ".nii"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                content = await f.read()
                if len(content) == 0:
                    raise HTTPException(status_code=400, detail=f"{f.filename} is empty.")
                tmp.write(content)
                tmp_paths.append(tmp.name)

        img_path, gt_path = tmp_paths

        # Run prediction (result is in inference space: 128x128x32)
        metrics_pred, pred_mask, _ = predictor.analyze(img_path)

        # Process GT through the SAME spatial transforms as the image
        # so that gt_mask is in the same coordinate space as pred_mask.
        spacing = predictor.inference_spacing
        tkw = predictor.transform_kwargs
        gt_transforms = MonaiCompose([
            LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            Orientationd(keys=["image", "label"],
                         axcodes=tkw.get("orientation", "RAS")),
            Spacingd(keys=["image", "label"],
                     pixdim=tkw.get("pixel_spacing", spacing),
                     mode=("bilinear", "nearest")),
            Lambdad(keys=["label"], func=_BinarizeLabelTransform(3)),
            ScaleIntensityRangePercentilesd(
                keys=["image"],
                lower=tkw.get("intensity_lower", 0.5),
                upper=tkw.get("intensity_upper", 99.5),
                b_min=tkw.get("intensity_min", 0.0),
                b_max=tkw.get("intensity_max", 1.0),
                clip=True,
            ),
            SpatialPadd(keys=["image", "label"],
                        spatial_size=predictor.spatial_size),
            Resized(keys=["image", "label"],
                    spatial_size=predictor.spatial_size,
                    mode=("trilinear", "nearest")),
            EnsureTyped(keys=["image", "label"]),
        ])

        data = gt_transforms({"image": img_path, "label": gt_path})
        processed_image = data["image"].numpy().squeeze()
        gt_mask = data["label"].numpy().squeeze()
        gt_mask = (gt_mask > 0.5).astype(np.float32)

        # If GT has no label-3 voxels, fall back to any label > 0
        if gt_mask.sum() == 0:
            gt_transforms_any = MonaiCompose([
                LoadImaged(keys=["label"], ensure_channel_first=True),
                Orientationd(keys=["label"],
                             axcodes=tkw.get("orientation", "RAS")),
                Spacingd(keys=["label"],
                         pixdim=tkw.get("pixel_spacing", spacing),
                         mode="nearest"),
                Lambdad(keys=["label"],
                        func=lambda x: (x > 0).float()),
                SpatialPadd(keys=["label"],
                            spatial_size=predictor.spatial_size),
                Resized(keys=["label"],
                        spatial_size=predictor.spatial_size,
                        mode="nearest"),
                EnsureTyped(keys=["label"]),
            ])
            gt_mask = gt_transforms_any({"label": gt_path})["label"].numpy().squeeze()
            gt_mask = (gt_mask > 0.5).astype(np.float32)

        # Compute GT metrics in inference spacing
        from cardioquant3d.evaluation.clinical_metrics import compute_clinical_metrics
        metrics_gt = compute_clinical_metrics(gt_mask, spacing)

        # Compute Dice — both masks are now in the same space
        p = pred_mask.flatten()
        g = gt_mask.flatten()
        intersection = (p * g).sum()
        dice = (2.0 * intersection) / (p.sum() + g.sum() + 1e-8)

        # Build comparison figure using preprocessed image
        img_b64 = _build_comparison_figure(processed_image, pred_mask, gt_mask)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>CardioQuant3D — Comparison Result</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ background: #0f0f23; color: #e0e0e0; font-family: 'Segoe UI', Arial, sans-serif; padding: 30px; }}
        h1 {{ color: #6EC8A0; text-align: center; margin-bottom: 8px; font-size: 28px; }}
        .subtitle {{ text-align: center; color: #888; margin-bottom: 24px; font-size: 14px; }}
        .container {{ max-width: 1100px; margin: 0 auto; }}

        .dice-banner {{
            text-align: center; margin-bottom: 24px; padding: 18px;
            background: linear-gradient(135deg, #1a1a3e, #16213e);
            border: 1px solid #2a2a5a; border-radius: 12px;
        }}
        .dice-banner .score {{ font-size: 48px; font-weight: bold; color: {"#6EC8A0" if dice >= 0.85 else "#D4956B" if dice >= 0.70 else "#D47272"}; }}
        .dice-banner .label {{ font-size: 14px; color: #aaa; margin-top: 2px; }}

        .metrics-compare {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 24px; }}
        .metrics-panel {{
            background: linear-gradient(135deg, #1a1a3e, #16213e);
            border: 1px solid #2a2a5a; border-radius: 12px; padding: 20px;
        }}
        .metrics-panel h3 {{ font-size: 14px; margin-bottom: 12px; }}
        .metrics-panel h3.pred {{ color: #6EC8A0; }}
        .metrics-panel h3.gt {{ color: #D4956B; }}
        .metric-row {{ display: flex; justify-content: space-between; padding: 6px 0;
                       border-bottom: 1px solid #2a2a5a22; font-size: 14px; }}
        .metric-row .val {{ color: #fff; font-weight: bold; }}

        .visual {{ text-align: center; }}
        .visual img {{ max-width: 100%; border-radius: 12px; border: 1px solid #2a2a5a; }}
        .back {{ text-align: center; margin-top: 20px; }}
        .back a {{ color: #6EC8A0; text-decoration: none; font-size: 13px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>CardioQuant3D</h1>
        <p class="subtitle">Prediction vs Ground Truth — {image.filename}</p>

        <div class="dice-banner">
            <div class="score">{dice:.4f}</div>
            <div class="label">Dice Score</div>
        </div>

        <div class="metrics-compare">
            <div class="metrics-panel">
                <h3 class="pred">Prediction</h3>
                <div class="metric-row"><span>LV Volume</span><span class="val">{metrics_pred.lv_volume_ml:.2f} ml</span></div>
                <div class="metric-row"><span>Surface Area</span><span class="val">{metrics_pred.surface_area_mm2:.1f} mm²</span></div>
                <div class="metric-row"><span>Long Axis</span><span class="val">{metrics_pred.long_axis_mm:.2f} mm</span></div>
                <div class="metric-row"><span>Sphericity</span><span class="val">{metrics_pred.sphericity_index:.4f}</span></div>
            </div>
            <div class="metrics-panel">
                <h3 class="gt">Ground Truth</h3>
                <div class="metric-row"><span>LV Volume</span><span class="val">{metrics_gt.lv_volume_ml:.2f} ml</span></div>
                <div class="metric-row"><span>Surface Area</span><span class="val">{metrics_gt.surface_area_mm2:.1f} mm²</span></div>
                <div class="metric-row"><span>Long Axis</span><span class="val">{metrics_gt.long_axis_mm:.2f} mm</span></div>
                <div class="metric-row"><span>Sphericity</span><span class="val">{metrics_gt.sphericity_index:.4f}</span></div>
            </div>
        </div>

        <div class="visual">
            <img src="data:image/png;base64,{img_b64}" alt="Comparison" />
        </div>

        <div class="back"><a href="/compare">&#8592; Compare another</a> &nbsp;|&nbsp; <a href="/">Analyze</a></div>
    </div>
</body>
</html>"""

        return HTMLResponse(content=html)

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("Comparison failed")
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")
    finally:
        for p in tmp_paths:
            try:
                os.unlink(p)
            except (OSError, UnboundLocalError):
                pass
