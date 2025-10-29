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

def conv_block(c_in, c_out, dilation=1):
    """
    Convolutional block with two 3x3 convolutions, batch norm, and ReLU.
    
    The second convolution can use dilation to increase the receptive field
    without losing resolution, improving context understanding.
    
    Args:
        c_in: Number of input channels
        c_out: Number of output channels
        dilation: Dilation rate for the second convolution
    
    Returns:
        Sequential module containing the block
    """
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, kernel_size=3,
                  padding=dilation, dilation=dilation, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )

class Down(nn.Module):
    """
    Downsampling block: max pooling followed by convolutional block.
    
    Reduces spatial dimensions by 2x while increasing channel capacity.
    """
    
    def __init__(self, c_in, c_out):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.block = conv_block(c_in, c_out)
    
    def forward(self, x):
        x = self.pool(x)
        return self.block(x)
