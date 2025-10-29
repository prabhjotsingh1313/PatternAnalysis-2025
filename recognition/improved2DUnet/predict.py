"""
predict.py - Inference / visualization script for trained Improved U-Net

This script:
  - Loads the best saved checkpoint from training (best_model.pth)
  - Runs inference on a batch from the test set
  - Computes per-channel Dice on that batch
  - Saves visualizations:
        MRI input
        Ground truth mask
        Predicted mask
        Overlay of predicted prostate on MRI
"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from modules import ImprovedUNet2D, dice_per_channel
from dataset import get_data_loaders

def make_label_colormap(num_classes):
    """
    Build a simple, fixed colormap for visualizing multi-class segmentation masks.
    We cap at 7 colors (background + 6 classes). Add more if needed.
    """
    base_colors = [
        (0, 0, 0),       # 0 - black
        (1, 0, 0),       # 1 - red
        (0, 1, 0),       # 2 - green
        (0, 0, 1),       # 3 - blue
        (1, 1, 0),       # 4 - yellow
        (0, 1, 1),       # 5 - cyan
        (1, 0, 1),       # 6 - magenta
    ]
    base_colors = base_colors[:num_classes]
    return ListedColormap(base_colors)

def visualize_batch(images,
                    gts_onehot,
                    preds_probs,
                    prostate_ch,
                    out_dir,
                    num_samples=4):
    """
    Save two visualizations:

    1. predictions.png:
        For each sample:
            - MRI input (grayscale)
            - GT argmax mask (color)
            - Pred argmax mask (color)

    2. overlays.png:
        For each sample:
            - MRI input
            - MRI with predicted prostate channel overlaid in red

    Args:
        images:       [B, 1, H, W] tensor on CPU
        gts_onehot:   [B, C, H, W] tensor on CPU
        preds_probs:  [B, C, H, W] tensor on CPU (sigmoid outputs)
        prostate_ch:  int, index of prostate channel (e.g. 3)
        out_dir:      str, directory to save figs
        num_samples:  number of samples from the batch to visualize
    """

    os.makedirs(out_dir, exist_ok=True)

    images_np = images.numpy()
    gts_np = gts_onehot.numpy()
    preds_np = preds_probs.numpy()

    B, C, H, W = gts_np.shape
    num_samples = min(num_samples, B)

    # --------- Figure 1: GT vs Prediction ---------
    cmap = make_label_colormap(C)

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        # MRI input
        axes[i, 0].imshow(images_np[i, 0], cmap='gray')
        axes[i, 0].set_title(f'Sample {i+1}: MRI Input')
        axes[i, 0].axis('off')

        # GT argmax mask
        gt_mask = np.argmax(gts_np[i], axis=0)  # [H,W]
        axes[i, 1].imshow(gt_mask, cmap=cmap, vmin=0, vmax=C-1)
        axes[i, 1].set_title('Ground Truth (argmax)')
        axes[i, 1].axis('off')

        # Pred argmax mask
        pred_mask = np.argmax(preds_np[i], axis=0)  # [H,W]
        axes[i, 2].imshow(pred_mask, cmap=cmap, vmin=0, vmax=C-1)
        axes[i, 2].set_title('Prediction (argmax)')
        axes[i, 2].axis('off')

    plt.tight_layout()
    pred_path = os.path.join(out_dir, "predictions.png")
    plt.savefig(pred_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[predict] Saved {pred_path}")

    # --------- Figure 2: Overlay of prostate prediction ---------
    fig, axes = plt.subplots(2, num_samples, figsize=(4*num_samples, 8))
    if num_samples == 1:
        axes = axes.reshape(2, 1)

    for i in range(num_samples):
        mri_slice = images_np[i, 0]     # [H,W]
        pred_ch   = preds_np[i, prostate_ch]  # prostate probability map [H,W]

        # Normalise MRI to [0,1]
        mri_min, mri_max = mri_slice.min(), mri_slice.max()
        mri_norm = (mri_slice - mri_min) / (mri_max - mri_min + 1e-6)

        # Base grey RGB
        overlay_rgb = np.stack([mri_norm, mri_norm, mri_norm], axis=-1)

        # Threshold prostate mask at 0.5 to get binary prediction
        prostate_bin = (pred_ch > 0.5).astype(np.float32)

        # Paint prostate in red channel
        overlay_rgb[..., 0] = np.maximum(overlay_rgb[..., 0], prostate_bin)

        # Row 1: original MRI
        axes[0, i].imshow(mri_slice, cmap='gray')
        axes[0, i].set_title(f"Sample {i+1}: MRI")
        axes[0, i].axis('off')

        # Row 2: overlay
        axes[1, i].imshow(overlay_rgb)
        axes[1, i].set_title("Predicted Prostate Overlay")
        axes[1, i].axis('off')

    plt.tight_layout()
    overlay_path = os.path.join(out_dir, "overlays.png")
    plt.savefig(overlay_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[predict] Saved {overlay_path}")


@torch.no_grad()
def predict_and_report(
    data_path,
    checkpoint_path,
    device="cuda",
    num_samples=4,
    out_dir="./predictions"
):
    """
    Full inference pipeline.

    1. Rebuild loaders via dataset.get_data_loaders()
    2. Load the best saved model checkpoint
    3. Take one batch from the test loader
    4. Run the model, get probabilities, threshold
    5. Compute Dice per channel for that batch
    6. Visualize GT vs prediction and overlay prostate

    Prints batch Dice and saves visualizations.
    """

    # pick device
    device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    print(f"[predict] Using device: {device}")

    # 1. Load data: we'll only use test_loader here
    print("[predict] Loading data loaders...")
    train_loader, val_loader, test_loader, num_classes, label_to_ch = get_data_loaders(
        base_path=data_path,
        batch_size=8,
        num_workers=2,
        out_size=(256, 256)
    )

    # We identified in training that prostate was channel 3 (small gland).
    # If you ever change that mapping, update here.
    prostate_ch = 3 if num_classes > 3 else 0
    print(f"[predict] Assuming prostate channel index = {prostate_ch}")

    # 2. Load checkpoint
    print(f"[predict] Loading checkpoint from {checkpoint_path} ...")
    ckpt = torch.load(checkpoint_path, map_location=device)

    # ckpt was saved in train.py as:
    # {
    #   'epoch': ...,
    #   'model_state_dict': ...,
    #   'optimizer_state_dict': ...,
    #   'val_loss': ...,
    #   'num_classes': num_classes,
    #   'label_to_ch': label_to_ch,
    # }
    saved_num_classes = ckpt.get('num_classes', num_classes)
    print(f"[predict] Checkpoint num_classes = {saved_num_classes}")

    # build model and load weights
    model = ImprovedUNet2D(
        in_ch=1,
        num_classes=saved_num_classes,
        base=32
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # 3. Grab a batch from test_loader
    xb, yb = next(iter(test_loader))  # xb: [B,1,H,W], yb:[B,C,H,W]
    xb = xb.to(device)
    yb = yb.to(device)

    # 4. Forward pass
    logits = model(xb)                  # [B,C,H,W], raw
    probs  = torch.sigmoid(logits)      # [B,C,H,W], 0..1

    # 5. Compute Dice per class for this batch
    batch_dice = dice_per_channel(logits, yb)  # [C]
    print("\n[predict] Dice per channel on this batch:")
    for ch, dval in enumerate(batch_dice.tolist()):
        print(f"  Channel {ch}: {dval:.4f}")

    prostate_dice = batch_dice[prostate_ch].item()
    print(f"\n[predict] Prostate Dice (channel {prostate_ch}) on this batch: {prostate_dice:.4f}")
    if prostate_dice >= 0.75:
        print("[predict] ✓ Meets required >= 0.75 Dice on prostate (spec requirement).")
    else:
        print("[predict] ✗ Does NOT meet >= 0.75 Dice on prostate.")

    # 6. Visualize and save figures
    visualize_batch(
        images=xb.cpu(),
        gts_onehot=yb.cpu(),
        preds_probs=probs.cpu(),
        prostate_ch=prostate_ch,
        out_dir=out_dir,
        num_samples=num_samples
    )

    print("\n[predict] Done. Visualizations saved in:", out_dir)



