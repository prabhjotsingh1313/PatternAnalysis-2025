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

