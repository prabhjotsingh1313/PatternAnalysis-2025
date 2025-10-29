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

class HipMRI2DSegDataset(Dataset):
    """
    PyTorch Dataset for HipMRI 2D segmentation.
    
    Loads 2D NIfTI MRI slices and segmentation masks, applies:
    - Per-slice intensity normalization (z-score)
    - One-hot encoding of segmentation masks
    - Resizing to fixed output dimensions
    
    Args:
        img_files: List of image file paths
        seg_files: List of segmentation file paths
        label_to_ch_map: Dictionary mapping label IDs to channel indices
        num_classes: Total number of segmentation classes
        out_size: Tuple of (height, width) for output images
        normalize: Whether to apply z-score normalization
    """
    
    def __init__(self, img_files, seg_files, label_to_ch_map, num_classes,
                 out_size=(256, 256), normalize=True):
        assert len(img_files) == len(seg_files), "Mismatch in image and segmentation file counts"
        
        self.img_files = img_files
        self.seg_files = seg_files
        self.label_to_ch = label_to_ch_map
        self.num_classes = num_classes
        self.out_size = out_size
        self.normalize = normalize
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        seg_path = self.seg_files[idx]
        
        # Load MRI slice
        img_nii = nib.load(img_path)
        img_np = img_nii.get_fdata(caching='unchanged')
        
        # Handle 3D shape (H, W, 1) -> (H, W)
        if img_np.ndim == 3:
            img_np = img_np[:, :, 0]
        
        img_np = img_np.astype(np.float32)
        
        # Per-slice z-score normalization
        if self.normalize:
            mean = img_np.mean()
            std = img_np.std() + 1e-6
            img_np = (img_np - mean) / std
        
        img_t = torch.from_numpy(img_np).unsqueeze(0)  # [1, H, W]
        
        # Load segmentation mask
        seg_nii = nib.load(seg_path)
        seg_np = seg_nii.get_fdata(caching='unchanged')
        
        if seg_np.ndim == 3:
            seg_np = seg_np[:, :, 0]
        
        seg_np = seg_np.astype(np.uint8)
        H, W = seg_np.shape
        
        # Convert to one-hot encoding with fixed channel order
        seg_onehot = np.zeros((self.num_classes, H, W), dtype=np.float32)
        for raw_label, ch_idx in self.label_to_ch.items():
            seg_onehot[ch_idx] = (seg_np == raw_label).astype(np.float32)
        
        seg_t = torch.from_numpy(seg_onehot)  # [C, H, W]
        
        # Resize to fixed dimensions for batching
        img_t = F.interpolate(
            img_t.unsqueeze(0),
            size=self.out_size,
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        
        seg_t = F.interpolate(
            seg_t.unsqueeze(0),
            size=self.out_size,
            mode='nearest'
        ).squeeze(0)
        
        return img_t, seg_t
    
def get_data_loaders(base_path, batch_size=8, num_workers=2, out_size=(256, 256)):
    """
    Create train, validation, and test data loaders.
    
    Args:
        base_path: Root directory containing keras_slices_* folders
        batch_size: Batch size for data loaders
        num_workers: Number of workers for parallel data loading
        out_size: Output size for images
    
    Returns:
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        num_classes: Total number of classes
        label_to_ch: Dictionary mapping label IDs to channels
    """
    # Define paths
    img_train = os.path.join(base_path, "keras_slices_train")
    seg_train = os.path.join(base_path, "keras_slices_seg_train")
    img_val = os.path.join(base_path, "keras_slices_validate")
    seg_val = os.path.join(base_path, "keras_slices_seg_validate")
    img_test = os.path.join(base_path, "keras_slices_test")
    seg_test = os.path.join(base_path, "keras_slices_seg_test")
    
    # Build file pairs
    train_imgs, train_segs = build_pairs(img_train, seg_train)
    val_imgs, val_segs = build_pairs(img_val, seg_val)
    test_imgs, test_segs = build_pairs(img_test, seg_test)
    
    print(f"Dataset splits:")
    print(f"  Train: {len(train_imgs)} samples")
    print(f"  Val:   {len(val_imgs)} samples")
    print(f"  Test:  {len(test_imgs)} samples")
    
    # Discover labels from training set
    label_ids, label_to_ch, num_classes = discover_labels(train_segs)
    print(f"\nDiscovered {num_classes} classes: {label_ids}")

    # Create datasets
    train_ds = HipMRI2DSegDataset(
        train_imgs, train_segs, label_to_ch, num_classes, out_size
    )
    val_ds = HipMRI2DSegDataset(
        val_imgs, val_segs, label_to_ch, num_classes, out_size
    )
    test_ds = HipMRI2DSegDataset(
        test_imgs, test_segs, label_to_ch, num_classes, out_size
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader, num_classes, label_to_ch