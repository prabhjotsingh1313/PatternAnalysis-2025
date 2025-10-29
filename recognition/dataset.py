"""
dataset.py - Data loader for HipMRI 2D segmentation dataset

Loads 2D NIfTI slices and corresponding segmentation masks,
performs preprocessing including normalization and resizing.
"""

import os
import glob
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def build_pairs(img_dir, seg_dir):
    """
    Pair image and segmentation files by matching filenames.
    
    Args:
        img_dir: Directory containing case_*.nii.gz files
        seg_dir: Directory containing seg_*.nii.gz files
    
    Returns:
        imgs: List of image file paths
        segs: List of corresponding segmentation file paths
    """
    img_paths_all = sorted(glob.glob(os.path.join(img_dir, "*.nii*")))
    imgs, segs = [], []
    
    for img_path in img_paths_all:
        base_img = os.path.basename(img_path)
        # Convert case_004_week_0_slice_0.nii.gz -> seg_004_week_0_slice_0.nii.gz
        seg_name = "seg_" + base_img[len("case_"):]
        seg_path = os.path.join(seg_dir, seg_name)
        
        if os.path.exists(seg_path):
            imgs.append(img_path)
            segs.append(seg_path)
    
    return imgs, segs

def discover_labels(seg_files, max_samples=100):
    """
    Discover unique label IDs in the dataset.
    
    Args:
        seg_files: List of segmentation file paths
        max_samples: Maximum number of files to scan
    
    Returns:
        label_ids: Sorted list of unique label IDs
        label_to_ch: Dictionary mapping label ID to channel index
        num_classes: Total number of classes
    """
    label_ids_global = set()
    
    for seg_path in seg_files[:max_samples]:
        seg_np = nib.load(seg_path).get_fdata(caching='unchanged')
        if seg_np.ndim == 3:
            seg_np = seg_np[:, :, 0]
        seg_np = seg_np.astype(np.uint8)
        
        for uid in np.unique(seg_np):
            label_ids_global.add(int(uid))
    
    label_ids = sorted(list(label_ids_global))
    label_to_ch = {lab: i for i, lab in enumerate(label_ids)}
    num_classes = len(label_ids)
    
    return label_ids, label_to_ch, num_classes