"""
modules.py - Neural network components for 2D Improved U-Net

Implements an improved U-Net architecture with:
- Dilated convolutions for increased receptive field
- Batch normalization for stable training
- Skip connections for preserving spatial information
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def center_crop(tensor, target_h, target_w):
    """
    Center crop a tensor to target height and width.
    
    Args:
        tensor: Input tensor of shape [B, C, H, W]
        target_h: Target height
        target_w: Target width
    
    Returns:
        Cropped tensor of shape [B, C, target_h, target_w]
    """
    _, _, h, w = tensor.shape
    dh = (h - target_h) // 2
    dw = (w - target_w) // 2
    return tensor[:, :, dh:dh + target_h, dw:dw + target_w]