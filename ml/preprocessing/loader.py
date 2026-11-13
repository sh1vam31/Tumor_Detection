import os
import numpy as np
import torch
from typing import Tuple, List, Optional, Dict, Any

# Safe import for nibabel
try:
    import nibabel as nib
except ImportError:
    nib = None

def z_score_normalize(volume: np.ndarray) -> np.ndarray:
    """
    Applies robust percentile clipping (1st to 99th percentile) and Z-score normalization
    specifically to non-zero brain regions. Standardizes scanner contrast across BraTS cases.
    """
    normalized = np.zeros_like(volume, dtype=np.float32)
    mask = volume > 0
    if np.any(mask):
        vals = volume[mask]
        p1 = float(np.percentile(vals, 1))
        p99 = float(np.percentile(vals, 99))
        clipped = np.clip(volume, p1, p99)
        
        mean = float(np.mean(clipped[mask]))
        std = float(np.std(clipped[mask]))
        if std > 1e-6:
            normalized[mask] = (clipped[mask] - mean) / std
        else:
            normalized[mask] = clipped[mask] - mean
    return normalized

def resample_to_brats_grid(
    volume: np.ndarray, 
    target_shape: Tuple[int, int, int] = (64, 64, 64),
    bbox: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    is_seg: bool = False,
    return_bbox: bool = False
) -> Any:
    """
    Intelligently resamples an MRI volume into the canonical 3D BraTS grid (64, 64, 64):
    1. Reorients dimensions so Depth/Axial is axis 0: (Z, Y, X).
       In standard BraTS NIfTI files, shape is (X, Y, Z) = (240, 240, 155),
       where axis 2 is the axial stack. We transpose to (Z, Y, X).
    2. Uses provided brain bounding box (bbox) or computes non-zero brain bounding box.
    3. Crops to the brain bounding box.
    4. Resamples using 3D trilinear interpolation (or nearest-neighbor for segmentation labels)
       so the ENTIRE brain and tumor fit into target_shape with 1:1 anatomical alignment.
    Returns:
      (resampled_volume, resampled_brain_mask) or (resampled_volume, resampled_brain_mask, bbox)
    """
    import torch.nn.functional as F
    
    vol = volume.astype(np.float32)
    
    # 1. Reorient if needed: if shape has 3rd dimension smaller than 1st & 2nd (e.g. 240, 240, 155)
    if vol.ndim == 3 and vol.shape[2] < vol.shape[0] and vol.shape[2] < vol.shape[1]:
        # Transpose (X, Y, Z) -> (Z, Y, X)
        vol = np.transpose(vol, (2, 1, 0))
        
    # 2. Determine bounding box
    if bbox is not None:
        min_c, max_c = bbox
        min_c = np.maximum(0, np.minimum(min_c, np.array(vol.shape)))
        max_c = np.maximum(min_c + 1, np.minimum(max_c, np.array(vol.shape)))
        cropped_vol = vol[min_c[0]:max_c[0], min_c[1]:max_c[1], min_c[2]:max_c[2]]
        computed_bbox = (min_c, max_c)
    elif vol.shape == target_shape:
        cropped_vol = vol
        computed_bbox = (np.zeros(3, dtype=int), np.array(vol.shape, dtype=int))
    else:
        # Find brain bounding box (voxels > 5th percentile of non-zero to eliminate background air)
        nonzero_vals = vol[vol > 0]
        thresh = max(0.0, float(np.percentile(nonzero_vals, 5))) if len(nonzero_vals) > 0 else 0.0
        brain_idx = np.argwhere(vol > thresh)
        
        if len(brain_idx) > 100:
            min_c = np.maximum(0, brain_idx.min(axis=0) - 2)
            max_c = np.minimum(np.array(vol.shape), brain_idx.max(axis=0) + 3)
            cropped_vol = vol[min_c[0]:max_c[0], min_c[1]:max_c[1], min_c[2]:max_c[2]]
            computed_bbox = (min_c, max_c)
        else:
            cropped_vol = vol
            computed_bbox = (np.zeros(3, dtype=int), np.array(vol.shape, dtype=int))
        
    # 3. Interpolation to target shape
    tensor = torch.from_numpy(cropped_vol).unsqueeze(0).unsqueeze(0).float()
    mode = 'nearest' if is_seg else 'trilinear'
    kwargs = {} if is_seg else {'align_corners': False}
    resampled = F.interpolate(
        tensor, 
        size=target_shape, 
        mode=mode, 
        **kwargs
    ).squeeze().cpu().numpy()
    
    # Brain mask on the resampled volume
    if is_seg:
        brain_mask = (resampled > 0).astype(np.float32)
    else:
        resampled_nonzero = resampled[resampled > 0]
        mask_thresh = max(0.0, float(np.percentile(resampled_nonzero, 2))) if len(resampled_nonzero) > 0 else 0.0
        brain_mask = (resampled > mask_thresh).astype(np.float32)
    
    if return_bbox:
        return resampled, brain_mask, computed_bbox
    return resampled, brain_mask


def load_nifti_volume(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads a .nii or .nii.gz volume using nibabel.
    Returns: (voxel_data_3d, affine_matrix)
    """
    if nib is None:
        raise ImportError("nibabel is required for reading raw .nii files. Run: pip install nibabel")
    img = nib.load(file_path)
    data = img.get_fdata().astype(np.float32)
    affine = img.affine
    return data, affine

def preprocess_multimodal_mri(
    modality_paths: Dict[str, str], 
    target_shape: Tuple[int, int, int] = (64, 64, 64)
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Loads 4 MRI modalities (T1, T1ce, T2, FLAIR), normalizes them,
    and stacks into a 4D PyTorch tensor [4, Depth, Height, Width].
    """
    modalities_order = ['t1', 't1ce', 't2', 'flair']
    processed_channels = []
    affine = np.eye(4)

    for mod in modalities_order:
        path = modality_paths.get(mod)
        if path and os.path.exists(path):
            vol, aff = load_nifti_volume(path)
            affine = aff
        else:
            # Fallback or synthetic representation if file is single or missing
            vol = np.zeros(target_shape, dtype=np.float32)
        
        # Normalize
        norm_vol = z_score_normalize(vol)
        
        # Center crop or pad to target shape
        cropped = resize_or_crop_3d(norm_vol, target_shape)
        processed_channels.append(cropped)

    # Stack to shape [4, D, H, W]
    stacked = np.stack(processed_channels, axis=0)
    tensor = torch.from_numpy(stacked).float().unsqueeze(0) # [1, 4, D, H, W]
    return tensor, affine

def resize_or_crop_3d(volume: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    """Center crops or pads a 3D numpy array to match target_shape."""
    curr_shape = volume.shape
    if curr_shape == target_shape:
        return volume

    output = np.zeros(target_shape, dtype=volume.dtype)
    
    # Calculate crop/pad ranges
    src_slices = []
    dst_slices = []
    
    for c_dim, t_dim in zip(curr_shape, target_shape):
        if c_dim >= t_dim:
            start = (c_dim - t_dim) // 2
            src_slices.append(slice(start, start + t_dim))
            dst_slices.append(slice(0, t_dim))
        else:
            start = (t_dim - c_dim) // 2
            src_slices.append(slice(0, c_dim))
            dst_slices.append(slice(start, start + c_dim))
            
    output[tuple(dst_slices)] = volume[tuple(src_slices)]
    return output

def create_synthetic_brats_case(shape: Tuple[int, int, int] = (64, 64, 64)) -> Dict[str, np.ndarray]:
    """
    Generates an anatomically structured synthetic 3D brain scan with multi-region tumor
    (Whole Tumor, Tumor Core, Enhancing Tumor) for offline testing and demos.
    """
    d, h, w = shape
    z, y, x = np.mgrid[:d, :h, :w]
    
    # Brain boundary (ellipsoid)
    center = (d // 2, h // 2, w // 2)
    radius = (d // 2.3, h // 2.2, w // 2.2)
    brain_mask = ((z - center[0])**2 / radius[0]**2 + 
                  (y - center[1])**2 / radius[1]**2 + 
                  (x - center[2])**2 / radius[2]**2) <= 1.0

    # Base brain tissue intensity
    t1 = np.zeros(shape, dtype=np.float32)
    t1[brain_mask] = 80.0 + 15.0 * np.sin(x[brain_mask] / 5.0) + 10.0 * np.cos(y[brain_mask] / 5.0)
    
    t1ce = t1.copy()
    t2 = t1.copy() * 1.2
    flair = t1.copy() * 0.9

    # Tumor location (offset from center)
    t_center = (center[0] + 4, center[1] - 5, center[2] + 6)
    
    # Whole Tumor (WT) - Edema sphere (radius 10)
    dist_sq = (z - t_center[0])**2 + (y - t_center[1])**2 + (x - t_center[2])**2
    wt_mask = (dist_sq <= 10**2) & brain_mask
    flair[wt_mask] += 120.0
    t2[wt_mask] += 140.0
    
    # Tumor Core (TC) - radius 6
    tc_mask = (dist_sq <= 6**2) & brain_mask
    t1[tc_mask] -= 30.0
    
    # Enhancing Tumor (ET) - rim of active tumor
    et_mask = (dist_sq <= 6**2) & (dist_sq >= 3**2) & brain_mask
    t1ce[et_mask] += 160.0

    ground_truth = np.zeros((3, d, h, w), dtype=np.float32)
    ground_truth[0] = wt_mask.astype(np.float32) # Whole Tumor
    ground_truth[1] = tc_mask.astype(np.float32) # Tumor Core
    ground_truth[2] = et_mask.astype(np.float32) # Enhancing Tumor

    return {
        "t1": t1,
        "t1ce": t1ce,
        "t2": t2,
        "flair": flair,
        "brain_mask": brain_mask,
        "ground_truth": ground_truth
    }
