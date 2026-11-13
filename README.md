# 🧠 3D U-Net Multimodal Brain Tumor Segmentation & Clinical Web Visualizer

**Semester 5 On-the-Job Training (OJT) Capstone Project**  
*Multimodal Deep Learning & Interactive WebGL Volumetric Suite for Clinical Oncology*

---

Live Demo - https://tumor-detection-sq03.onrender.com/


## 🌟 System Architecture & Core Capabilities

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Interactive Clinical Web App                    │
│   (Three.js 3D WebGL Volumetric Studio + 2D DICOM Slice Navigator)    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Upload (FLAIR, T1, T1ce, T2 / SEG)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   Unified FastAPI Microservice Engine                  │
│       • Bounding Box Extraction & Non-Zero Z-Score Standardization     │
│       • PyTorch 3D U-Net Inference Engine (CPU / MPS / CUDA)           │
│       • Pathological Lesion Anchoring & Morphological Cleaning        │
│       • Marching Cubes 3D Surface Reconstruction (VTK-grade meshes)   │
│       • Quantitative Volumetric Estimation (mm³ / cm³)                 │
└────────────────────────────────────────────────────────────────────────┘
```

### Key Technical Innovations
1. **Multi-Channel 3D Deep Learning**: Takes all 4 core clinical MRI sequences (FLAIR, T1ce, T2, T1) and resamples to a standardized $64 \times 64 \times 64$ volumetric grid.
2. **Subregion Segmentation Hierarchy (BraTS Standard)**:
   * **Whole Tumor (WT)**: Peritumoral edema + infiltrating tumor tissue.
   * **Tumor Core (TC)**: Necrotic center + active enhancing mass.
   * **Enhancing Tumor (ET)**: Actively proliferating angiogenic tissue.
3. **Hardware-Accelerated 3D WebGL Viewer**:
   * Interactive camera rotation, zoom, and preset anatomical projections (3D Iso, Axial, Coronal, Sagittal).
   * Layer visibility toggles (Brain Cortex, WT, TC, ET, Orthogonal MPR Slices, 3D Cage).
   * 3D Cross-Sectional clipping planes with dynamic Z and Y plane cuts.
4. **2D DICOM Multi-Slice Axial Explorer**:
   * Real base64 grayscale canvas rendering across all 64 axial depths.
   * Dynamic window leveling/contrast adjustment.
   * Modality switching (FLAIR, T1ce, T2, T1).
   * Concentric boundary rendering to inspect tumor core margins against active enhancing regions.
   * Automated Cine Loop playback and "Jump to Tumor Slice" auto-navigation.
5. **Clinical PDF Diagnostic Report Generator**:
   * One-click PDF generation featuring patient metadata, quantitative tumor burden, subregion volumes, and radiologist sign-off section.

---

## 📂 Repository Directory Layout

```
.
├── backend/
│   └── main.py                     # Unified FastAPI application (API + React SPA serving)
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   └── ThreeDViewer.jsx   # Three.js WebGL 3D volumetric renderer
│   │   ├── App.jsx                 # Main clinical diagnostic suite & 2D DICOM viewer
│   │   ├── App.css                 # Dark-mode glassmorphic clinical styling
│   │   └── main.jsx                # React root mount
│   ├── package.json
│   └── vite.config.js
├── ml/
│   ├── models/
│   │   └── unet3d.py               # 3D U-Net neural network architecture (1.4M params)
│   ├── losses/
│   │   └── dice_loss.py            # Soft Dice Loss & Hausdorff distance metrics
│   ├── preprocessing/
│   │   └── loader.py               # NIfTI loader, bbox crop, and Z-score normalization
│   ├── postprocessing/
│   │   └── volume.py               # Marching Cubes mesh generator & volume formulas
│   ├── weights/
│   │   └── best_model.pth          # PyTorch trained weights checkpoint
│   └── train.py                    # Standalone training script
├── sample_data/                    # Lightweight benchmark patient scans for 1-click test
├── tests/
│   ├── test_backend.py             # FastAPI endpoints, file upload & SPA verification
│   └── test_ml_pipeline.py         # PyTorch shapes, Dice formulas, and math verification
├── build.sh                        # Render automated deployment build script
├── render.yaml                     # Render Infrastructure-as-Code Blueprint
├── requirements.txt                # CPU-optimized Python dependencies for cloud deployment
└── README.md
```

---

## ⚡ Quick Start (Local Development)

### 1. Prerequisites
* Python 3.9+ with `pip`
* Node.js 18+ and `npm`

### 2. Setup Backend & Python Environment
```bash
# Install dependencies (CPU-optimized PyTorch)
pip install -r requirements.txt

# Run automated tests to verify model & endpoints
python3 -m pytest tests/ -v
```

### 3. Run Unified Server (Recommended)
You can run the full fullstack application with a single command:
```bash
# Build the frontend assets
cd frontend && npm install && npm run build && cd ..

# Launch FastAPI on port 8000
uvicorn backend.main:app --reload --port 8000
```
Open **`http://localhost:8000`** in your browser. Both the React interface and the Python ML backend will be live!

### 4. Separate Dev Mode (Frontend Hot-Reloading)
```bash
# Terminal 1: Backend
uvicorn backend.main:app --reload --port 8000

# Terminal 2: Frontend
cd frontend
npm run dev
```

---

## ☁️ Deploying to Render (Free Cloud Hosting)

This repository is pre-configured with a unified architecture so the entire app (FastAPI backend + React frontend) runs as a single web service on Render's free tier.

### Option A: 1-Click Render Blueprint (Easiest)
1. Fork or push this repository to your GitHub account:
   ```bash
   git add .
   git commit -m "Configure production deployment on Render"
   git push origin main
   ```
2. Log in to [Render Dashboard](https://dashboard.render.com).
3. Click **"New +"** → **"Blueprint"**.
4. Select your `Medical_Image_Segmentation-` repository.
5. Render reads `render.yaml` automatically and creates the service. Click **"Apply"**.

### Option B: Manual Web Service Setup
If creating a Web Service manually in Render:
1. Click **"New +"** → **"Web Service"**.
2. Connect your GitHub repository.
3. Configure the settings:
   * **Name:** `brain-tumor-segmentation-3d`
   * **Language:** `Python`
   * **Branch:** `main`
   * **Build Command:** `./build.sh`
   * **Start Command:** `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
   * **Plan:** `Free`
4. Under **Environment Variables**, add:
   * `PYTHON_VERSION` = `3.10.12`
   * `PYTORCH_ENABLE_MPS_FALLBACK` = `1`
5. Click **"Deploy Web Service"**.

Render will run `./build.sh`, install the lightweight CPU PyTorch wheel, compile the React assets into `frontend/dist`, and launch the web server. Within 2-3 minutes, you will receive a public HTTPS URL (e.g. `https://brain-tumor-segmentation-3d.onrender.com`) ready to present for your college evaluation!

---

## 🧪 Verification & Test Results
```bash
$ python3 -m pytest tests/ -v
============================== test session starts ==============================
tests/test_backend.py::test_read_root PASSED                             [  9%]
tests/test_backend.py::test_health_check PASSED                          [ 18%]
tests/test_backend.py::test_segment_demo PASSED                          [ 27%]
tests/test_backend.py::test_segment_sample PASSED                        [ 36%]
tests/test_backend.py::test_segment_upload_single PASSED                 [ 45%]
tests/test_backend.py::test_spa_index_serving PASSED                     [ 54%]
tests/test_ml_pipeline.py::test_unet3d_forward_pass PASSED               [ 63%]
tests/test_ml_pipeline.py::test_dice_coefficient_metric PASSED           [ 72%]
tests/test_ml_pipeline.py::test_z_score_normalization PASSED             [ 81%]
tests/test_ml_pipeline.py::test_volume_math_calculation PASSED           [ 90%]
tests/test_ml_pipeline.py::test_hausdorff_distance_metric PASSED         [100%]
============================== 11 passed in 2.66s ==============================
```

---

## 📜 Academic Integrity & License
This project was developed by **Shivam Chaudhary** (Reg: 240410700100) for the Semester 5 On-Job-Training (OJT) requirement. Licensed under the MIT License.
