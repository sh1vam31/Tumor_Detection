import pytest
import torch
import numpy as np

from ml.models.unet3d import UNet3D
from ml.losses.dice_loss import SoftDiceLoss, compute_dice_coefficient, compute_hausdorff_distance_95
from ml.preprocessing.loader import z_score_normalize, create_synthetic_brats_case
from ml.postprocessing.volume import calculate_tumor_volumes

def test_unet3d_forward_pass():
    """Verify 3D U-Net tensor input and output dimensions"""
    model = UNet3D(in_channels=4, out_channels=3, init_features=8)
    dummy_tensor = torch.randn(1, 4, 16, 32, 32)
    output = model(dummy_tensor)
    assert output.shape == (1, 3, 16, 32, 32), f"Expected shape (1, 3, 16, 32, 32), got {output.shape}"

def test_dice_coefficient_metric():
    """Verify mathematical Dice Score calculations"""
    gt = np.ones((10, 10, 10))
    pred_exact = np.ones((10, 10, 10))
    pred_empty = np.zeros((10, 10, 10))

    score_perfect = compute_dice_coefficient(pred_exact, gt)
    score_zero = compute_dice_coefficient(pred_empty, gt)

    assert round(score_perfect, 2) == 1.0, f"Expected 1.0, got {score_perfect}"
    assert round(score_zero, 2) == 0.0, f"Expected 0.0, got {score_zero}"

def test_z_score_normalization():
    """Verify non-zero voxel standardization"""
    arr = np.array([0.0, 0.0, 10.0, 20.0, 30.0], dtype=np.float32)
    norm = z_score_normalize(arr)
    assert norm[0] == 0.0
    assert norm[1] == 0.0
    non_zero = norm[2:]
    assert np.isclose(np.mean(non_zero), 0.0, atol=1e-5)
    assert np.isclose(np.std(non_zero), 1.0, atol=1e-5)

def test_volume_math_calculation():
    """Verify voxel-to-mm3 mathematical scaling"""
    dummy_masks = np.zeros((3, 10, 10, 10), dtype=np.float32)
    dummy_masks[0, :5, :5, :5] = 1.0 # 5*5*5 = 125 voxels for WT
    
    volumes = calculate_tumor_volumes(dummy_masks, voxel_spacing=(2.0, 2.0, 2.0))
    # 125 voxels * (2*2*2 = 8 mm3/voxel) = 1000 mm3 = 1 cm3
    assert volumes["whole_tumor"]["voxel_count"] == 125
    assert volumes["whole_tumor"]["volume_mm3"] == 1000.0
    assert volumes["whole_tumor"]["volume_cm3"] == 1.0

def test_hausdorff_distance_metric():
    """Verify Hausdorff 95th percentile contour boundary metric"""
    mask1 = np.zeros((20, 20, 20), dtype=np.float32)
    mask2 = np.zeros((20, 20, 20), dtype=np.float32)
    mask1[5:10, 5:10, 5:10] = 1.0
    mask2[5:10, 5:10, 5:10] = 1.0
    hd = compute_hausdorff_distance_95(mask1, mask2)
    assert hd == 0.0
