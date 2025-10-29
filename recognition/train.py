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

@torch.no_grad()
def evaluate_dice(model, loader, device="cuda"):
    """
    Evaluate Dice coefficient on a dataset.
    
    Args:
        model: Trained model
        loader: Data loader
        device: Device to run on
    
    Returns:
        avg_dice: Average Dice coefficient per channel [C]
    """
    model.eval()
    dice_sum = None
    n_batches = 0
    
    for xb, yb in tqdm(loader, desc="Evaluating"):
        xb = xb.to(device)
        yb = yb.to(device)
        
        logits = model(xb)
        dpc = dice_per_channel(logits, yb)  # [C]
        
        if dice_sum is None:
            dice_sum = dpc.clone()
        else:
            dice_sum += dpc
        n_batches += 1
    
    return (dice_sum / n_batches).cpu()

