import torch
import torch.nn as nn
import numpy as np
from scipy.spatial.distance import directed_hausdorff

class SoftDiceLoss(nn.Module):
    """
    Computes Soft Dice Loss for multi-class/multi-label 3D medical segmentation.
    Dice = 2 * |X ∩ Y| / (|X| + |Y| + smooth)
    """
    def __init__(self, smooth=1e-5):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # Apply sigmoid to convert logits to probabilities
        probs = torch.sigmoid(logits)
        
        # Flatten tensors across spatial dimensions: [Batch, Channels, Spatial]
        batch_size = targets.size(0)
        num_classes = targets.size(1)
        
        probs = probs.view(batch_size, num_classes, -1)
        targets = targets.view(batch_size, num_classes, -1)
        
        intersection = torch.sum(probs * targets, dim=-1)
        cardinality = torch.sum(probs + targets, dim=-1)
        
        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        loss = 1.0 - dice_score.mean()
        return loss
    
class CombinedDiceCrossEntropyLoss(nn.Module):
    """
    Combined Loss = BCEWithLogitsLoss + SoftDiceLoss
    Mitigates severe class imbalance in 3D brain tumor MRI scans.
    """
    def __init__(self, dice_weight=0.5, bce_weight=0.5, smooth=1e-5):
        super(CombinedDiceCrossEntropyLoss, self).__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = SoftDiceLoss(smooth=smooth)
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        dice = self.dice_loss(logits, targets)
        bce = self.bce_loss(logits, targets.float())
        return (self.dice_weight * dice) + (self.bce_weight * bce)

def compute_dice_coefficient(prediction: np.ndarray, ground_truth: np.ndarray, smooth: float = 1e-5) -> float:
    """
    Computes standard Dice Similarity Coefficient (DSC):
    DSC = (2 * |Pred ∩ GT|) / (|Pred| + |GT|)
    """
    pred_flat = (prediction > 0.5).astype(np.float32).flatten()
    gt_flat = (ground_truth > 0.5).astype(np.float32).flatten()
    
    intersection = np.sum(pred_flat * gt_flat)
    total = np.sum(pred_flat) + np.sum(gt_flat)
    
    if total == 0:
        return 1.0 # Both empty (true negative)
    
    return float((2.0 * intersection + smooth) / (total + smooth))

def compute_hausdorff_distance_95(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Computes approximate 95th percentile Hausdorff Distance between binary contours.
    """
    pred_coords = np.argwhere(pred_mask > 0.5)
    gt_coords = np.argwhere(gt_mask > 0.5)
    
    if len(pred_coords) == 0 and len(gt_coords) == 0:
        return 0.0
    if len(pred_coords) == 0 or len(gt_coords) == 0:
        return 373.0 # Maximum penalty distance
        
    # Sample subset of points for fast computation
    max_pts = 1000
    if len(pred_coords) > max_pts:
        idx = np.random.choice(len(pred_coords), max_pts, replace=False)
        pred_coords = pred_coords[idx]
    if len(gt_coords) > max_pts:
        idx = np.random.choice(len(gt_coords), max_pts, replace=False)
        gt_coords = gt_coords[idx]

    d_forward = directed_hausdorff(pred_coords, gt_coords)[0]
    d_backward = directed_hausdorff(gt_coords, pred_coords)[0]
    hd = max(d_forward, d_backward)
    return float(round(hd * 0.95, 2))
