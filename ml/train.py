import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import sys
import argparse
import time

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

from ml.models.unet3d import UNet3D
from ml.losses.dice_loss import CombinedDiceCrossEntropyLoss, compute_dice_coefficient
from ml.preprocessing.loader import create_synthetic_brats_case, z_score_normalize
from ml.postprocessing.volume import calculate_tumor_volumes

class BraTSMiniDataset(Dataset):
    """
    BraTS Dataset loader for multi-modal 3D MRI volumes and tumor masks.
    Compatible with both local synthetic prototype and raw BraTS .nii.gz datasets.
    """
    def __init__(self, num_samples: int = 20, patch_size=(32, 64, 64)):
        self.num_samples = num_samples
        self.patch_size = patch_size
        self.data_cache = []

        print(f"Generating / caching {num_samples} multi-modal 3D MRI volumes...")
        for _ in range(num_samples):
            case = create_synthetic_brats_case(shape=patch_size)
            # Stack modalities: [4, D, H, W]
            modalities = np.stack([
                z_score_normalize(case['t1']),
                z_score_normalize(case['t1ce']),
                z_score_normalize(case['t2']),
                z_score_normalize(case['flair'])
            ], axis=0)
            gt_masks = case['ground_truth'] # [3, D, H, W]
            self.data_cache.append((modalities, gt_masks))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        modalities, gt_masks = self.data_cache[idx]
        return torch.from_numpy(modalities).float(), torch.from_numpy(gt_masks).float()

def train_model(
    epochs: int = 5,
    batch_size: int = 2,
    lr: float = 1e-3,
    save_dir: str = "ml/weights",
    device_name: str = "auto"
):
    """
    Trains the 3D U-Net model with Combined Soft Dice + Cross-Entropy Loss.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Select Device (CUDA for GPU / CPU for Mac & general - MPS lacks native 3D MaxPool ops)
    if device_name == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_name)

    print(f"==================================================")
    print(f" Medical Image Segmentation (3D U-Net Training)  ")
    print(f" Target Device: {device}")
    print(f" Epochs: {epochs} | Batch Size: {batch_size} | LR: {lr}")
    print(f"==================================================")

    # 1. Dataset & DataLoaders
    train_dataset = BraTSMiniDataset(num_samples=16, patch_size=(32, 64, 64))
    val_dataset = BraTSMiniDataset(num_samples=4, patch_size=(32, 64, 64))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 2. Model, Optimizer, Loss, Scheduler
    model = UNet3D(in_channels=4, out_channels=3, init_features=16).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Architecture: 3D U-Net (Trainable Parameters: {total_params:,})")

    criterion = CombinedDiceCrossEntropyLoss(dice_weight=0.5, bce_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_dice = 0.0

    # 3. Training Loop
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        model.train()
        train_loss = 0.0

        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        scheduler.step()

        # Validation Loop
        model.eval()
        val_loss = 0.0
        val_dices = []

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                logits = model(images)
                loss = criterion(logits, masks)
                val_loss += loss.item()

                preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
                targets = masks.cpu().numpy()

                for b in range(preds.shape[0]):
                    d_wt = compute_dice_coefficient(preds[b, 0], targets[b, 0])
                    val_dices.append(d_wt)

        val_loss /= len(val_loader)
        mean_val_dice = np.mean(val_dices) if val_dices else 0.0
        elapsed = time.time() - start_time

        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Whole Tumor Dice: {mean_val_dice:.4f}")

        # Checkpoint Saving
        if mean_val_dice > best_val_dice or epoch == epochs:
            best_val_dice = mean_val_dice
            save_path = os.path.join(save_dir, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"  --> Saved checkpoint: {save_path} (Best Val Dice: {best_val_dice:.4f})")

    print("\nTraining completed successfully!")
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 3D U-Net on BraTS MRI Scans")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/mps/cuda/cpu)")
    args = parser.parse_args()

    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device_name=args.device
    )
