"""
train.py - Training, validation, and testing script for Improved U-Net

Trains the model using Dice loss, validates on validation set,
and evaluates final performance on test set.
"""

import os
import argparse
import torch
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

from modules import ImprovedUNet2D, dice_loss, dice_per_channel
from dataset import get_data_loaders


def run_one_epoch(model, loader, optimizer=None, device="cuda"):
    """
    Run one epoch of training or validation.
    
    Args:
        model: Neural network model
        loader: Data loader
        optimizer: Optimizer (None for validation)
        device: Device to run on
    
    Returns:
        avg_loss: Average loss over the epoch
    """
    train_mode = optimizer is not None
    model.train(train_mode)
    
    total_loss = 0.0
    steps = 0
    
    for xb, yb in tqdm(loader, desc="Training" if train_mode else "Validating"):
        xb = xb.to(device)
        yb = yb.to(device)
        
        # Forward pass
        logits = model(xb)
        loss = dice_loss(logits, yb)
        
        # Backward pass (only in training mode)
        if train_mode:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        steps += 1
    
    return total_loss / max(steps, 1)

