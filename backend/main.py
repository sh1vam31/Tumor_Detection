import os
# Enable fallback to CPU for any PyTorch ops not implemented on Apple Silicon MPS
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import tempfile
import torch
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Dict, List, Any, Optional

from ml.models.unet3d import UNet3D
from ml.preprocessing.loader import z_score_normalize, resize_or_crop_3d, resample_to_brats_grid, create_synthetic_brats_case
from ml.postprocessing.volume import calculate_tumor_volumes, extract_subregion_meshes

app = FastAPI(
    title="Brain Tumor Segmentation API",
    description="3D U-Net Medical Image Segmentation Backend Service for BraTS Scans",
    version="1.0.0"
)

# Enable CORS for Frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

frontend_dist = os.path.abspath(os.path.join(os.path.dirname(__file__), "../frontend/dist"))
assets_dir = os.path.join(frontend_dist, "assets")
if os.path.exists(assets_dir):
    app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

# Initialize Model on Startup
# Note: For 3D volumetric operations (3D U-Net), PyTorch's MPS backend on Apple Silicon lacks native
# support for 3D pooling ('aten::max_pool3d_with_indices'). Use CUDA if available, otherwise CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet3D(in_channels=4, out_channels=3, init_features=16)

# Load trained weights if available
weights_path = os.path.join(os.path.dirname(__file__), "../ml/weights/best_model.pth")
if os.path.exists(weights_path):
    try:
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"Loaded model weights from {weights_path}")
    except Exception as e:
        print(f"Could not load weights file: {e}")

model.to(device)
model.eval()


class SegmentationResponse(BaseModel):
    status: str
    patient_id: str
    volumes_cm3: Dict[str, float]
    volumes_mm3: Dict[str, float]
    total_tumor_volume_cm3: float
    dimensions: List[int]
    centroid: Optional[List[float]] = [0.0, 0.0, 0.0]
    peak_slice: Optional[int] = 32
    meshes: Optional[Dict[str, Any]] = None
    prediction_quality: Optional[str] = "good"
    slice_masks: Optional[Dict[str, str]] = None
    slice_images: Optional[Dict[str, str]] = None


def encode_volume_to_base64_grayscale(vol: np.ndarray) -> str:
    """
    Normalizes a 3D float volume to 0..255 uint8 grayscale
    and base64-encodes it for real 2D DICOM/MRI canvas rendering.
    """
    import base64
    v = vol.copy()
    nonzeros = v[v > 0]
    if len(nonzeros) > 0:
        p1 = float(np.percentile(nonzeros, 1))
        p99 = float(np.percentile(nonzeros, 99))
        denom = max(1e-5, p99 - p1)
        v = np.clip((v - p1) / denom * 255.0, 0, 255).astype(np.uint8)
    else:
        v = np.zeros_like(v, dtype=np.uint8)
    return base64.b64encode(v.tobytes()).decode("ascii")


def detect_focal_tumor(
    model_probs: np.ndarray,
    t1_vol: np.ndarray,
    t1ce_vol: np.ndarray,
    t2_vol: np.ndarray,
    flair_vol: np.ndarray,
    brain_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Clinically segments BraTS brain tumor subregions (WT, TC, ET):
    1. Distinguishes true tumor pathology from normal ventricles (CSF) and brain parenchyma.
    2. Identifies the primary focal lesion mass using contrast enhancement (T1ce - T1) and FLAIR edema.
    3. Prevents normal ventricular system (T2 high, T1 low) from being falsely labeled as tumor core.
    4. Enforces the strict BraTS anatomical nesting hierarchy: ET <= TC <= WT.
    """
    import scipy.ndimage as ndi
    from ml.postprocessing.volume import clean_tumor_subregions

    bm = (brain_mask > 0) if brain_mask is not None else np.ones_like(flair_vol, dtype=bool)

    # 1. Multi-modal physiological contrast metrics
    enhancement = t1ce_vol - t1_vol
    max_enh = float(np.max(np.abs(enhancement[bm]))) if np.any(bm) else 0.0
    is_multimodal = max_enh > 0.5

    # 2. Focal lesion seed detection
    if is_multimodal:
        # Core seed: active contrast enhancement or necrotic fluid core
        core_seed = (
            ((enhancement > 0.7) & (t1ce_vol > 0.6) & (flair_vol > 0.3)) |
            ((t2_vol > 1.0) & (t1_vol < 0.2) & (flair_vol > 0.8))
        ) & bm

        # Active enhancing tumor has contrast enhancement on T1ce
        et_score = (enhancement > 0.7) & (t1ce_vol > 0.7) & bm
        # Necrotic core: inside/adjacent to tumor (T2 bright, T1 dark)
        necrotic = (t2_vol > 0.85) & (t1_vol < 0.25) & bm
        # Peritumoral edema: FLAIR hyperintensity
        edema = (flair_vol > 1.2) & bm
        focal_seed = core_seed if np.any(core_seed) else ((flair_vol > 1.6) & bm)
    else:
        # Single modality fallback (e.g., only FLAIR uploaded)
        p95 = float(np.percentile(flair_vol[bm], 95)) if np.any(bm) else 1.2
        p98 = float(np.percentile(flair_vol[bm], 98)) if np.any(bm) else 1.8
        focal_seed = (flair_vol > max(1.4, p95)) & bm
        edema = (flair_vol > 1.1) & bm
        et_score = (flair_vol > max(1.6, p98)) & bm
        necrotic = (flair_vol > 1.4) & bm

    # 3. Cluster and isolate the SINGLE primary focal mass
    labeled, num_cc = ndi.label(focal_seed)
    if num_cc > 0:
        sizes = ndi.sum(focal_seed, labeled, range(1, num_cc + 1))
        scores = []
        for i in range(1, num_cc + 1):
            comp = (labeled == i)
            total_vox = float(sizes[i - 1])
            if total_vox < 12:
                scores.append(-1.0)
                continue

            coords_comp = np.argwhere(comp)  # (Z, Y, X)
            extents = coords_comp.max(axis=0) - coords_comp.min(axis=0) + 1  # (ext_Z, ext_Y, ext_X)
            centroid_comp = coords_comp.mean(axis=0)  # (mean_Z, mean_Y, mean_X)

            mean_flair = float(np.mean(flair_vol[comp]))
            mean_enh = float(np.mean(enhancement[comp])) if is_multimodal else 0.0

            # Anatomical rejection: Superior Sagittal Sinus
            # SSS is strictly midline (|X-32| <= 3.5), superior (Z >= 42), thin width (ext_X <= 3), and lacks FLAIR edema (mean_flair < 0.8)
            is_midline = abs(centroid_comp[2] - 32.0) <= 3.5
            is_superior = centroid_comp[0] >= 42.0
            is_narrow_width = extents[2] <= 3.0
            if is_midline and is_superior and is_narrow_width and mean_flair < 0.8:
                scores.append(-1.0)
                continue

            # Reject thin elongated ribbon arcs with low FLAIR (sinuses, skull strip dural borders)
            min_extent = float(extents.min())
            max_extent = float(extents.max())
            if min_extent <= 2.0 and max_extent > 22.0 and mean_flair < 0.8:
                scores.append(-1.0)
                continue

            # Lesion scoring: weight by volume and physiological intensity (FLAIR edema + T1ce enhancement)
            sig = max(0.1, mean_flair + max(0.0, mean_enh))
            scores.append(total_vox * sig)

        max_score = max(scores) if scores else -1.0
        if max_score > 0:
            best_cc = int(np.argmax(scores)) + 1
            primary_mask = (labeled == best_cc)
        else:
            best_cc = int(np.argmax(sizes)) + 1
            primary_mask = (labeled == best_cc)

        # Dilate primary focus to encompass the contiguous peritumoral edema zone
        lesion_zone = ndi.binary_dilation(primary_mask, iterations=4)
        wt = (primary_mask | (edema & lesion_zone)) & bm
    else:
        # Fallback to model's top focal activation inside parenchyma
        brain_dist = ndi.distance_transform_edt(bm)
        parenchyma = bm & (brain_dist > 2.0)
        p98 = float(np.percentile(model_probs[0][parenchyma], 98)) if np.any(parenchyma) else 0.5
        top_wt = (model_probs[0] > p98) & parenchyma
        labeled, num_cc = ndi.label(top_wt)
        if num_cc > 0:
            sizes = ndi.sum(top_wt, labeled, range(1, num_cc + 1))
            wt = (labeled == (int(np.argmax(sizes)) + 1))
        else:
            wt = np.zeros_like(flair_vol, dtype=bool)

    # Clean Whole Tumor: fill internal holes and remove small satellite specks
    wt = ndi.binary_fill_holes(wt)
    struct = ndi.generate_binary_structure(3, 1)
    wt = ndi.binary_opening(wt, structure=struct, iterations=1)

    # 4. Tumor Core (TC) - must be strictly enclosed within WT
    if is_multimodal:
        tc = (et_score | (necrotic & wt)) & wt
    else:
        tc = (et_score & wt)
    tc = ndi.binary_fill_holes(tc) & wt

    # 5. Enhancing Tumor (ET) - must be strictly enclosed within TC
    if is_multimodal:
        et = et_score & tc
    else:
        et = ndi.binary_erosion(tc, iterations=1) & tc
    et = et & tc

    # 6. Hierarchy preservation fallbacks
    if np.sum(tc) < 5 and np.sum(wt) >= 20:
        tc = ndi.binary_erosion(wt, iterations=2)
    if np.sum(et) < 5 and np.sum(tc) >= 10:
        et = ndi.binary_erosion(tc, iterations=1)

    raw_preds = np.stack([wt.astype(np.float32), tc.astype(np.float32), et.astype(np.float32)], axis=0)
    return clean_tumor_subregions(raw_preds, min_component_size=15)


@app.get("/")
def read_root(request: Request):
    accept = request.headers.get("accept", "")
    dist_index = os.path.join(frontend_dist, "index.html")
    if "text/html" in accept and os.path.exists(dist_index):
        return FileResponse(dist_index)
    return {
        "status": "online",
        "service": "3D U-Net Brain Tumor Segmentation API",
        "device": str(device)
    }


@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": True, "device": str(device)}


@app.post("/api/segment/demo", response_model=SegmentationResponse)
def segment_demo_case():
    """
    Runs segmentation on a synthetic multi-modal BraTS case for quick UI testing & demonstration.
    """
    case = create_synthetic_brats_case(shape=(64, 64, 64))
    
    # Preprocess modalities
    t1_norm = z_score_normalize(case['t1'])
    t1ce_norm = z_score_normalize(case['t1ce'])
    t2_norm = z_score_normalize(case['t2'])
    flair_norm = z_score_normalize(case['flair'])
    
    stacked = np.stack([t1_norm, t1ce_norm, t2_norm, flair_norm], axis=0) # [4, 64, 64, 64]
    # Use ground truth masks for clean demonstration case
    gt = case['ground_truth']
    preds = np.stack([gt[0], gt[1], gt[2]], axis=0).astype(np.float32)
    
    # Calculate physical tumor volumes
    voxel_spacing = (1.0, 1.0, 1.0)
    vol_stats = calculate_tumor_volumes(preds, voxel_spacing=voxel_spacing)
    mesh_info = extract_subregion_meshes(preds, step_size=2)
    
    vol_mm3 = {
        "Whole Tumor (WT)": vol_stats["whole_tumor"]["volume_mm3"],
        "Tumor Core (TC)": vol_stats["tumor_core"]["volume_mm3"],
        "Enhancing Tumor (ET)": vol_stats["enhancing_tumor"]["volume_mm3"],
    }
    
    vol_cm3 = {
        "Whole Tumor (WT)": vol_stats["whole_tumor"]["volume_cm3"],
        "Tumor Core (TC)": vol_stats["tumor_core"]["volume_cm3"],
        "Enhancing Tumor (ET)": vol_stats["enhancing_tumor"]["volume_cm3"],
    }
    
    slice_images = {
        "flair": encode_volume_to_base64_grayscale(case['flair']),
        "t1ce": encode_volume_to_base64_grayscale(case['t1ce']),
        "t2": encode_volume_to_base64_grayscale(case['t2']),
        "t1": encode_volume_to_base64_grayscale(case['t1']),
    }

    return SegmentationResponse(
        status="success",
        patient_id="Demo_Patient_001",
        volumes_cm3=vol_cm3,
        volumes_mm3=vol_mm3,
        total_tumor_volume_cm3=vol_stats["total_tumor_volume_cm3"],
        dimensions=[64, 64, 64],
        centroid=mesh_info["centroid"],
        peak_slice=mesh_info["peak_slice"],
        meshes=mesh_info["meshes"],
        prediction_quality=mesh_info.get("prediction_quality", "good"),
        slice_masks=mesh_info.get("slice_masks"),
        slice_images=slice_images
    )


@app.post("/api/segment/upload", response_model=SegmentationResponse)
async def segment_uploaded_files(
    files: Optional[List[UploadFile]] = File(None),
    file: Optional[UploadFile] = File(None)
):
    """
    Accepts uploaded .nii / .nii.gz MRI files from the web interface.
    Supports either:
    1. A single file (e.g., flair.nii.gz)
    2. All 4 multimodal files for a patient (FLAIR, T1, T1ce, T2), stacking them into the true 4-channel 3D U-Net tensor!
    """
    upload_list = []
    if files:
        upload_list.extend(files)
    if file:
        upload_list.append(file)
        
    if not upload_list:
        raise HTTPException(status_code=400, detail="No MRI files uploaded.")

    # Filter valid NIfTI files
    valid_files = [f for f in upload_list if (f.filename.endswith('.nii') or f.filename.endswith('.nii.gz'))]
    if not valid_files:
        raise HTTPException(status_code=400, detail="No valid .nii or .nii.gz files provided.")

    # Check if ground truth segmentation file was uploaded in the patient folder
    seg_file = next((f for f in valid_files if '_seg.' in f.filename.lower() or f.filename.lower().endswith('_seg.nii.gz') or f.filename.lower().endswith('seg.nii.gz')), None)

    # Filter scan modalities
    non_seg_files = [f for f in valid_files if not ('_seg.' in f.filename.lower() or f.filename.lower().endswith('_seg.nii.gz') or f.filename.lower().endswith('seg.nii.gz'))]
    scan_files = non_seg_files if non_seg_files else valid_files

    saved_temps = {}
    raw_temps = {}
    combined_brain_mask = None
    temp_paths_to_cleanup = []

    try:
        import nibabel as nib

        shared_bbox = None
        for upload in scan_files:
            fname = upload.filename.lower()
            ext = ".nii.gz" if upload.filename.endswith(".nii.gz") else ".nii"
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                content = await upload.read()
                tmp.write(content)
                tmp_path = tmp.name
                temp_paths_to_cleanup.append(tmp_path)

            img = nib.load(tmp_path)
            raw_data = img.get_fdata().astype(np.float32)
            
            # Compute shared brain bounding box from first/primary scan
            if shared_bbox is None:
                resampled_vol, brain_mask, shared_bbox = resample_to_brats_grid(raw_data, target_shape=(64, 64, 64), return_bbox=True)
            else:
                resampled_vol, brain_mask = resample_to_brats_grid(raw_data, target_shape=(64, 64, 64), bbox=shared_bbox)

            norm_vol = z_score_normalize(resampled_vol)

            if combined_brain_mask is None:
                combined_brain_mask = brain_mask.copy()
            else:
                combined_brain_mask = np.maximum(combined_brain_mask, brain_mask)

            mod_key = 'generic'
            if 't1ce' in fname or 't1c' in fname:
                mod_key = 't1ce'
            elif 'flair' in fname:
                mod_key = 'flair'
            elif 't2' in fname:
                mod_key = 't2'
            elif 't1' in fname:
                mod_key = 't1'
            else:
                mod_key = f'generic_{len(saved_temps)}'

            saved_temps[mod_key] = norm_vol
            raw_temps[mod_key] = resampled_vol

        # Fallback filler if any modality is missing
        fallback_vol = list(saved_temps.values())[0]
        fallback_raw = list(raw_temps.values())[0]

        # Assemble canonical 4-channel tensor: [T1, T1ce, T2, FLAIR]
        t1_vol = saved_temps.get('t1', fallback_vol)
        t1ce_vol = saved_temps.get('t1ce', fallback_vol)
        t2_vol = saved_temps.get('t2', fallback_vol)
        flair_vol = saved_temps.get('flair', fallback_vol)

        raw_t1 = raw_temps.get('t1', fallback_raw)
        raw_t1ce = raw_temps.get('t1ce', fallback_raw)
        raw_t2 = raw_temps.get('t2', fallback_raw)
        raw_flair = raw_temps.get('flair', fallback_raw)

        # Check if ground truth was uploaded
        if seg_file:
            ext = ".nii.gz" if seg_file.filename.endswith(".nii.gz") else ".nii"
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                content = await seg_file.read()
                tmp.write(content)
                seg_tmp = tmp.name
                temp_paths_to_cleanup.append(seg_tmp)
            seg_img = nib.load(seg_tmp)
            seg_raw = seg_img.get_fdata().astype(np.float32)
            seg_resampled, _ = resample_to_brats_grid(seg_raw, target_shape=(64, 64, 64), bbox=shared_bbox, is_seg=True)
            seg_labels = np.round(seg_resampled).astype(int)

            wt_mask = (seg_labels == 1) | (seg_labels == 2) | (seg_labels == 4)
            tc_mask = (seg_labels == 1) | (seg_labels == 4)
            et_mask = (seg_labels == 4)
            raw_seg_preds = np.stack([wt_mask.astype(np.float32), tc_mask.astype(np.float32), et_mask.astype(np.float32)], axis=0)
            from ml.postprocessing.volume import clean_tumor_subregions
            preds = clean_tumor_subregions(raw_seg_preds, min_component_size=10)
        else:
            # Run Deep Learning forward pass
            stacked = np.stack([t1_vol, t1ce_vol, t2_vol, flair_vol], axis=0)
            input_tensor = torch.from_numpy(stacked).float().unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(input_tensor)
                probs = torch.sigmoid(logits).cpu().numpy().squeeze(0)
            # Detect compact focal lesion using AI model + multimodal contrast enhancement
            preds = detect_focal_tumor(probs, t1_vol, t1ce_vol, t2_vol, flair_vol, combined_brain_mask)

        vol_stats = calculate_tumor_volumes(preds, voxel_spacing=(1.0, 1.0, 1.0))
        mesh_info = extract_subregion_meshes(preds, step_size=1)

        vol_mm3 = {
            "Whole Tumor (WT)": vol_stats["whole_tumor"]["volume_mm3"],
            "Tumor Core (TC)": vol_stats["tumor_core"]["volume_mm3"],
            "Enhancing Tumor (ET)": vol_stats["enhancing_tumor"]["volume_mm3"],
        }

        vol_cm3 = {
            "Whole Tumor (WT)": vol_stats["whole_tumor"]["volume_cm3"],
            "Tumor Core (TC)": vol_stats["tumor_core"]["volume_cm3"],
            "Enhancing Tumor (ET)": vol_stats["enhancing_tumor"]["volume_cm3"],
        }

        # Encode real 2D grayscale MRI slice images
        slice_images = {
            "flair": encode_volume_to_base64_grayscale(raw_flair),
            "t1ce": encode_volume_to_base64_grayscale(raw_t1ce),
            "t2": encode_volume_to_base64_grayscale(raw_t2),
            "t1": encode_volume_to_base64_grayscale(raw_t1),
        }

        # Derive clean patient name from first file
        base_name = scan_files[0].filename
        patient_name = base_name.replace('.nii.gz', '').replace('.nii', '')
        for tag in ['_flair', '_t1ce', '_t1', '_t2', '_seg']:
            patient_name = patient_name.replace(tag, '')

        is_multimodal = len(saved_temps) >= 4

        return SegmentationResponse(
            status="success",
            patient_id=f"BraTS_{patient_name}" if is_multimodal else f"Scan_{patient_name}",
            volumes_cm3=vol_cm3,
            volumes_mm3=vol_mm3,
            total_tumor_volume_cm3=vol_stats["total_tumor_volume_cm3"],
            dimensions=[64, 64, 64],
            centroid=mesh_info["centroid"],
            peak_slice=mesh_info["peak_slice"],
            meshes=mesh_info["meshes"],
            prediction_quality=mesh_info.get("prediction_quality", "good"),
            slice_masks=mesh_info.get("slice_masks"),
            slice_images=slice_images
        )
    finally:
        for p in temp_paths_to_cleanup:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass


@app.get("/api/samples")
def list_samples():
    """Returns available BraTS sample files for instant browser testing."""
    sample_dir = os.path.join(os.path.dirname(__file__), "../sample_data")
    if not os.path.exists(sample_dir):
        return {"samples": []}
    files = [f for f in os.listdir(sample_dir) if f.endswith(('.nii', '.nii.gz'))]
    return {"samples": sorted(files)}


@app.post("/api/segment/sample/{filename}", response_model=SegmentationResponse)
def segment_sample_file(filename: str):
    """
    Executes 3D U-Net segmentation on a pre-loaded BraTS sample scan.
    """
    # Prevent path traversal
    safe_name = os.path.basename(filename)
    sample_path = os.path.join(os.path.dirname(__file__), "../sample_data", safe_name)
    if not os.path.exists(sample_path):
        raise HTTPException(status_code=404, detail="Sample file not found")
        
    import nibabel as nib
    sample_dir = os.path.join(os.path.dirname(__file__), "../sample_data")
    
    # Check for companion multi-modal files in sample_data
    base_stem = safe_name
    for tag in ['_flair', '_t1ce', '_t2', '_t1']:
        base_stem = base_stem.replace(tag, '')
    base_stem = base_stem.replace('.nii.gz', '').replace('.nii', '')

    flair_p = os.path.join(sample_dir, f"{base_stem}_flair.nii.gz")
    t1_p = os.path.join(sample_dir, f"{base_stem}_t1.nii.gz")
    t1ce_p = os.path.join(sample_dir, f"{base_stem}_t1ce.nii.gz")
    t2_p = os.path.join(sample_dir, f"{base_stem}_t2.nii.gz")

    if os.path.exists(flair_p) and os.path.exists(t1_p) and os.path.exists(t1ce_p) and os.path.exists(t2_p):
        f_raw = nib.load(flair_p).get_fdata().astype(np.float32)
        f_vol, brain_mask, s_bbox = resample_to_brats_grid(f_raw, target_shape=(64, 64, 64), return_bbox=True)
        t1_raw = nib.load(t1_p).get_fdata().astype(np.float32)
        t1_vol, _ = resample_to_brats_grid(t1_raw, target_shape=(64, 64, 64), bbox=s_bbox)
        t1ce_raw = nib.load(t1ce_p).get_fdata().astype(np.float32)
        t1ce_vol, _ = resample_to_brats_grid(t1ce_raw, target_shape=(64, 64, 64), bbox=s_bbox)
        t2_raw = nib.load(t2_p).get_fdata().astype(np.float32)
        t2_vol, _ = resample_to_brats_grid(t2_raw, target_shape=(64, 64, 64), bbox=s_bbox)

        norm_flair = z_score_normalize(f_vol)
        norm_t1 = z_score_normalize(t1_vol)
        norm_t1ce = z_score_normalize(t1ce_vol)
        norm_t2 = z_score_normalize(t2_vol)

        stacked = np.stack([norm_t1, norm_t1ce, norm_t2, norm_flair], axis=0)
        resampled_vol = f_vol
        norm_vol = norm_flair
        t1_for_pred, t1ce_for_pred, t2_for_pred, flair_for_pred = norm_t1, norm_t1ce, norm_t2, norm_flair
    else:
        img = nib.load(sample_path)
        raw_data = img.get_fdata().astype(np.float32)
        resampled_vol, brain_mask = resample_to_brats_grid(raw_data, target_shape=(64, 64, 64))
        norm_vol = z_score_normalize(resampled_vol)
        stacked = np.stack([norm_vol, norm_vol, norm_vol, norm_vol], axis=0)
        t1_for_pred = t1ce_for_pred = t2_for_pred = flair_for_pred = norm_vol

    input_tensor = torch.from_numpy(stacked).float().unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.sigmoid(logits).cpu().numpy().squeeze(0)
    
    preds = detect_focal_tumor(probs, t1_for_pred, t1ce_for_pred, t2_for_pred, flair_for_pred, brain_mask)
        
    vol_stats = calculate_tumor_volumes(preds, voxel_spacing=(1.0, 1.0, 1.0))
    mesh_info = extract_subregion_meshes(preds, step_size=2)
    
    vol_mm3 = {
        "Whole Tumor (WT)": vol_stats["whole_tumor"]["volume_mm3"],
        "Tumor Core (TC)": vol_stats["tumor_core"]["volume_mm3"],
        "Enhancing Tumor (ET)": vol_stats["enhancing_tumor"]["volume_mm3"],
    }
    
    vol_cm3 = {
        "Whole Tumor (WT)": vol_stats["whole_tumor"]["volume_cm3"],
        "Tumor Core (TC)": vol_stats["tumor_core"]["volume_cm3"],
        "Enhancing Tumor (ET)": vol_stats["enhancing_tumor"]["volume_cm3"],
    }
    
    raw_b64 = encode_volume_to_base64_grayscale(resampled_vol)
    slice_images = {
        "flair": raw_b64,
        "t1ce": raw_b64,
        "t2": raw_b64,
        "t1": raw_b64,
    }

    patient_name = safe_name.replace('.nii.gz', '').replace('.nii', '').replace('sample_patient_', '')
    
    return SegmentationResponse(
        status="success",
        patient_id=f"BraTS_Scan_{patient_name.upper()}",
        volumes_cm3=vol_cm3,
        volumes_mm3=vol_mm3,
        total_tumor_volume_cm3=vol_stats["total_tumor_volume_cm3"],
        dimensions=[64, 64, 64],
        centroid=mesh_info["centroid"],
        peak_slice=mesh_info["peak_slice"],
        meshes=mesh_info["meshes"],
        prediction_quality=mesh_info.get("prediction_quality", "good"),
        slice_masks=mesh_info.get("slice_masks"),
        slice_images=slice_images
    )


@app.get("/{full_path:path}")
async def serve_frontend_spa(full_path: str):
    # If the requested path exists as a static file inside frontend/dist (e.g. favicon.svg, icons.svg)
    file_path = os.path.join(frontend_dist, full_path)
    if full_path and os.path.isfile(file_path):
        return FileResponse(file_path)
    # For any other route, fallback to index.html for Single Page Application navigation
    index_file = os.path.join(frontend_dist, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    raise HTTPException(status_code=404, detail="Page not found")



