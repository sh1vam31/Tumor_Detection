import numpy as np
from typing import Dict, Any, Tuple, List
from skimage import measure

import scipy.ndimage as ndi

def clean_tumor_subregions(pred_masks: np.ndarray, min_component_size: int = 25) -> np.ndarray:
    """
    Clinically refines and denoises predicted 3D segmentation subregions:
    1. Morphological binary closing to seal internal voids/hollow rings.
    2. Connected component analysis to remove isolated debris and satellite false-positive clusters.
    3. Enforces the strict BraTS anatomical nesting hierarchy: ET <= TC <= WT.
    """
    cleaned = np.zeros_like(pred_masks, dtype=np.float32)
    
    # Process WT (Channel 0)
    wt_raw = pred_masks[0] > 0.5
    if np.sum(wt_raw) > 0:
        wt_closed = ndi.binary_fill_holes(wt_raw)
        
        labeled, num_features = ndi.label(wt_closed)
        if num_features > 0:
            sizes = ndi.sum(wt_closed, labeled, range(1, num_features + 1))
            largest_idx = int(np.argmax(sizes)) + 1
            primary_mask = (labeled == largest_idx)
            # Only keep primary lesion and contiguous peritumoral satellites
            dilated_primary = ndi.binary_dilation(primary_mask, iterations=3)
            wt_filtered = np.zeros_like(wt_closed, dtype=bool)
            for i, s in enumerate(sizes, 1):
                if i == largest_idx or (s >= min_component_size and np.any(np.logical_and(labeled == i, dilated_primary))):
                    wt_filtered[labeled == i] = True
            cleaned[0] = wt_filtered.astype(np.float32)
    
    # Process TC (Channel 1) - must be inside cleaned WT
    tc_raw = np.logical_and(pred_masks[1] > 0.5, cleaned[0] > 0.5)
    if np.sum(tc_raw) > 0:
        tc_closed = ndi.binary_fill_holes(tc_raw)
        labeled, num_features = ndi.label(tc_closed)
        if num_features > 0:
            sizes = ndi.sum(tc_closed, labeled, range(1, num_features + 1))
            largest_idx = int(np.argmax(sizes)) + 1
            primary_tc = (labeled == largest_idx)
            dilated_tc = ndi.binary_dilation(primary_tc, iterations=2)
            tc_filtered = np.zeros_like(tc_closed, dtype=bool)
            for i, s in enumerate(sizes, 1):
                if i == largest_idx or (s >= max(10, min_component_size // 2) and np.any(np.logical_and(labeled == i, dilated_tc))):
                    tc_filtered[labeled == i] = True
            cleaned[1] = np.logical_and(tc_filtered, cleaned[0] > 0.5).astype(np.float32)
    elif np.sum(cleaned[0]) >= 20:
        cleaned[1] = ndi.binary_erosion(cleaned[0] > 0.5, iterations=2).astype(np.float32)

    # Process ET (Channel 2) - must be inside cleaned TC
    et_raw = np.logical_and(pred_masks[2] > 0.5, cleaned[1] > 0.5)
    if np.sum(et_raw) > 0:
        et_closed = ndi.binary_fill_holes(et_raw)
        labeled, num_features = ndi.label(et_closed)
        if num_features > 0:
            sizes = ndi.sum(et_closed, labeled, range(1, num_features + 1))
            largest_idx = int(np.argmax(sizes)) + 1
            primary_et = (labeled == largest_idx)
            dilated_et = ndi.binary_dilation(primary_et, iterations=2)
            et_filtered = np.zeros_like(et_closed, dtype=bool)
            for i, s in enumerate(sizes, 1):
                if i == largest_idx or (s >= max(5, min_component_size // 3) and np.any(np.logical_and(labeled == i, dilated_et))):
                    et_filtered[labeled == i] = True
            cleaned[2] = np.logical_and(et_filtered, cleaned[1] > 0.5).astype(np.float32)
    elif np.sum(cleaned[1]) >= 15:
        cleaned[2] = ndi.binary_erosion(cleaned[1] > 0.5, iterations=1).astype(np.float32)

    return cleaned


def enforce_brats_hierarchy(pred_masks: np.ndarray) -> np.ndarray:
    """
    Enforces the clinical BraTS physiological nested hierarchy:
    1. Whole Tumor (WT) encompasses all pathologic tissue.
    2. Tumor Core (TC) must be strictly a subset of WT: TC = TC & WT.
    3. Enhancing Tumor (ET) must be strictly a subset of TC: ET = ET & TC.
    """
    return clean_tumor_subregions(pred_masks)


def calculate_tumor_volumes(
    pred_masks: np.ndarray, 
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> Dict[str, Any]:
    """
    Calculates mathematical tumor volumes for each subregion based on voxel count and spacing.
    Enforces BraTS hierarchy and removes noise artifacts.
    pred_masks shape: [3, Depth, Height, Width] -> (0: WT, 1: TC, 2: ET)
    """
    pred_masks = clean_tumor_subregions(pred_masks)
    voxel_vol_mm3 = float(voxel_spacing[0] * voxel_spacing[1] * voxel_spacing[2])
    
    wt_voxels = int(np.sum(pred_masks[0] > 0.5))
    tc_voxels = int(np.sum(pred_masks[1] > 0.5))
    et_voxels = int(np.sum(pred_masks[2] > 0.5))

    wt_vol_mm3 = round(wt_voxels * voxel_vol_mm3, 2)
    tc_vol_mm3 = round(tc_voxels * voxel_vol_mm3, 2)
    et_vol_mm3 = round(et_voxels * voxel_vol_mm3, 2)

    return {
        "voxel_spacing_mm": list(voxel_spacing),
        "whole_tumor": {
            "voxel_count": wt_voxels,
            "volume_mm3": wt_vol_mm3,
            "volume_cm3": round(wt_vol_mm3 / 1000.0, 3)
        },
        "tumor_core": {
            "voxel_count": tc_voxels,
            "volume_mm3": tc_vol_mm3,
            "volume_cm3": round(tc_vol_mm3 / 1000.0, 3)
        },
        "enhancing_tumor": {
            "voxel_count": et_voxels,
            "volume_mm3": et_vol_mm3,
            "volume_cm3": round(et_vol_mm3 / 1000.0, 3)
        },
        "total_tumor_volume_mm3": wt_vol_mm3,
        "total_tumor_volume_cm3": round(wt_vol_mm3 / 1000.0, 3)
    }

def generate_3d_mesh(mask_3d: np.ndarray, step_size: int = 1, level: float = 0.5) -> Dict[str, Any]:
    """
    Converts a binary 3D segmentation mask into 3D mesh vertices and triangle faces
    using Gaussian-smoothed Marching Cubes for smooth, organic Three.js rendering.
    Maps voxel axes (Depth, Height, Width) -> Three.js (X, Y, Z) centered at (0, 0, 0):
      Axis 2 (Width/Sagittal)   -> Three.js X (lateral left-right)
      Axis 0 (Depth/Axial)      -> Three.js Y (superior-inferior height)
      Axis 1 (Height/Coronal)   -> Three.js Z (antero-posterior depth)
    """
    if np.sum(mask_3d > 0.5) < 15:
        return {"vertices": [], "faces": [], "normals": [], "vertex_count": 0, "face_count": 0}

    try:
        # Gaussian smoothing before Marching Cubes produces smooth organic isosurfaces
        # instead of jagged polygonal staircases and polygon tearing
        smoothed = ndi.gaussian_filter(mask_3d.astype(np.float32), sigma=0.85)

        verts, faces, normals, _ = measure.marching_cubes(
            smoothed, 
            level=level, 
            step_size=step_size
        )
        
        # Center in 64x64x64 volume coordinate space (-32 to +32):
        x = (verts[:, 2] - 32.0).astype(float)
        y = (verts[:, 0] - 32.0).astype(float)
        z = (verts[:, 1] - 32.0).astype(float)
        three_verts = np.stack([x, y, z], axis=1)

        nx = normals[:, 2].astype(float)
        ny = normals[:, 0].astype(float)
        nz = normals[:, 1].astype(float)
        three_normals = np.stack([nx, ny, nz], axis=1)

        return {
            "vertices": three_verts.flatten().tolist(), # [x0, y0, z0, x1, y1, z1, ...]
            "faces": faces.flatten().tolist(),             # [v0, v1, v2, ...]
            "normals": three_normals.flatten().tolist(),
            "vertex_count": int(len(verts)),
            "face_count": int(len(faces))
        }
    except Exception as e:
        print(f"Marching cubes error: {e}")
        return {"vertices": [], "faces": [], "normals": [], "vertex_count": 0, "face_count": 0}


def extract_subregion_meshes(pred_masks: np.ndarray, step_size: int = 1) -> Dict[str, Any]:
    """
    Extracts 3D surface meshes for WT, TC, and ET using Marching Cubes,
    along with centroid, peak slice, per-slice masks for 2D rendering,
    and a prediction quality indicator.
    pred_masks shape: [3, 64, 64, 64]
    """
    import base64

    # Denoise, close hollow voids, and enforce BraTS nesting constraints
    pred_masks = clean_tumor_subregions(pred_masks)


    wt_voxels_count = int(np.sum(pred_masks[0] > 0.5))
    tc_voxels_count = int(np.sum(pred_masks[1] > 0.5))
    et_voxels_count = int(np.sum(pred_masks[2] > 0.5))
    total_voxels = 64 * 64 * 64  # 262144

    # Anatomical sanity check:
    # - TC <= WT (guaranteed by hierarchy)
    # - WT should not exceed 15% of brain volume (< 15% is clinically realistic)
    wt_fraction = wt_voxels_count / total_voxels
    tc_ok = tc_voxels_count <= wt_voxels_count * 1.05
    wt_realistic = (wt_voxels_count > 0) and (wt_fraction < 0.15)
    prediction_quality = "good" if (tc_ok and wt_realistic) else "low"

    # Extract Marching Cubes meshes without excessive ballooning dilation
    if prediction_quality == "good":
        # Enforce that the outer WT mesh smoothly encloses the inner TC and ET meshes,
        # preventing the core from protruding outside on shared anatomical boundaries.
        tc_mask = pred_masks[1] > 0.5
        et_mask = pred_masks[2] > 0.5
        wt_envelope = (pred_masks[0] > 0.5) | ndi.binary_dilation(tc_mask, iterations=1)
        tc_envelope = tc_mask | (ndi.binary_dilation(et_mask, iterations=1) & wt_envelope)

        wt_mesh = generate_3d_mesh(wt_envelope, step_size=step_size, level=0.50)
        tc_mesh = generate_3d_mesh(tc_envelope, step_size=step_size, level=0.50)
        et_mesh = generate_3d_mesh(et_mask, step_size=step_size, level=0.50)
    else:
        # Return empty meshes — frontend will use procedural fallback geometry
        empty = {"vertices": [], "faces": [], "normals": [], "vertex_count": 0, "face_count": 0}
        wt_mesh, tc_mesh, et_mesh = empty, empty, empty


    # Prioritize Tumor Core (TC / ET) for peak_slice and centroid so the 2D viewer immediately
    # showcases the active core (yellow/red) rather than peripheral edema.
    tc_voxels = np.argwhere(pred_masks[1] > 0.5)
    wt_voxels = np.argwhere(pred_masks[0] > 0.5)
    if len(tc_voxels) > 0:
        tc_per_slice = np.sum(pred_masks[1] > 0.5, axis=(1, 2))
        peak_slice = int(np.argmax(tc_per_slice)) + 1  # Slice with maximum tumor core mass
        mean_z, mean_y, mean_x = tc_voxels.mean(axis=0)
        centroid = [float(round(mean_x - 32.0, 2)), float(round(mean_z - 32.0, 2)), float(round(mean_y - 32.0, 2))]
    elif len(wt_voxels) > 0:
        wt_per_slice = np.sum(pred_masks[0] > 0.5, axis=(1, 2))
        peak_slice = int(np.argmax(wt_per_slice)) + 1
        mean_z, mean_y, mean_x = wt_voxels.mean(axis=0)
        centroid = [float(round(mean_x - 32.0, 2)), float(round(mean_z - 32.0, 2)), float(round(mean_y - 32.0, 2))]
    else:
        centroid = [0.0, 0.0, 0.0]
        peak_slice = 32

    # Per-slice 2D masks (base64-encoded uint8): shape [64, 64, 64], axis-0 = slice index
    slice_masks = {
        "wt": base64.b64encode(pred_masks[0].astype(np.uint8).tobytes()).decode("ascii"),
        "tc": base64.b64encode(pred_masks[1].astype(np.uint8).tobytes()).decode("ascii"),
        "et": base64.b64encode(pred_masks[2].astype(np.uint8).tobytes()).decode("ascii"),
    }

    return {
        "meshes": {
            "wt": wt_mesh,
            "tc": tc_mesh,
            "et": et_mesh
        },
        "centroid": centroid,
        "peak_slice": peak_slice,
        "prediction_quality": prediction_quality,
        "slice_masks": slice_masks,
    }


