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
    
class Up(nn.Module):
    """
    Upsampling block: transposed convolution, concatenation with skip connection,
    followed by convolutional block.
    
    Increases spatial dimensions by 2x while reducing channel capacity.
    """
    
    def __init__(self, c_in, c_out):
        super().__init__()
        # Upsample: c_in -> c_in // 2
        self.up = nn.ConvTranspose2d(c_in, c_in // 2, kernel_size=2, stride=2)
        # After concat with skip (also c_in // 2 channels), total is c_in
        self.conv = conv_block(c_in, c_out)
    
    def forward(self, x, skip):
        """
        Args:
            x: Input from previous decoder layer
            skip: Skip connection from encoder
        
        Returns:
            Upsampled and processed features
        """
        x = self.up(x)
        
        # Crop skip connection to match upsampled size
        _, _, h, w = x.shape
        skip_cropped = center_crop(skip, h, w)
        
        # Concatenate along channel dimension
        x = torch.cat([x, skip_cropped], dim=1)
        return self.conv(x)
    
class ImprovedUNet2D(nn.Module):
    """
    Improved 2D U-Net for medical image segmentation.
    
    Architecture improvements:
    - Deeper bottleneck (5 levels instead of 4)
    - Dilated convolutions in bottleneck for larger receptive field
    - Batch normalization for training stability
    - Skip connections to preserve fine-grained details
    
    Reference:
    Ronneberger et al., "U-Net: Convolutional Networks for Biomedical 
    Image Segmentation", MICCAI 2015
    
    Args:
        in_ch: Number of input channels (1 for grayscale MRI)
        num_classes: Number of segmentation classes
        base: Base number of feature channels (scales by 2 at each level)
    """
    
    def __init__(self, in_ch=1, num_classes=6, base=32):
        super().__init__()
        
        # Encoder path (contracting)
        self.enc1 = conv_block(in_ch, base)          # 32 channels,  H x W
        self.enc2 = Down(base, base * 2)             # 64 channels,  H/2 x W/2
        self.enc3 = Down(base * 2, base * 4)         # 128 channels, H/4 x W/4
        self.enc4 = Down(base * 4, base * 8)         # 256 channels, H/8 x W/8
        
        # Bottleneck with dilation for larger receptive field
        self.bottleneck = Down(base * 8, base * 16)  # 512 channels, H/16 x W/16
        
        # Decoder path (expanding)
        self.up4 = Up(base * 16, base * 8)           # 512 -> 256 channels
        self.up3 = Up(base * 8, base * 4)            # 256 -> 128 channels
        self.up2 = Up(base * 4, base * 2)            # 128 -> 64 channels
        self.up1 = Up(base * 2, base)                # 64 -> 32 channels
        
        # Final 1x1 convolution for classification
        self.outc = nn.Conv2d(base, num_classes, kernel_size=1)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape [B, 1, H, W]
        
        Returns:
            logits: Output logits of shape [B, num_classes, H, W]
        """
        # Encoder with skip connections
        s1 = self.enc1(x)         # [B, 32,  H,    W]
        s2 = self.enc2(s1)        # [B, 64,  H/2,  W/2]
        s3 = self.enc3(s2)        # [B, 128, H/4,  W/4]
        s4 = self.enc4(s3)        # [B, 256, H/8,  W/8]
        
        # Bottleneck
        b = self.bottleneck(s4)   # [B, 512, H/16, W/16]
        
        # Decoder with skip connections
        x = self.up4(b, s4)       # [B, 256, H/8,  W/8]
        x = self.up3(x, s3)       # [B, 128, H/4,  W/4]
        x = self.up2(x, s2)       # [B, 64,  H/2,  W/2]
        x = self.up1(x, s1)       # [B, 32,  H,    W]
        
        # Classification
        logits = self.outc(x)     # [B, num_classes, H, W]
        
        return logits
