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

def plot_training_curves(train_losses, val_losses, save_path="training_curves.png"):
    """
    Plot and save training and validation loss curves.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        save_path: Path to save the figure
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-o', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-s', label='Validation Loss', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Dice Loss', fontsize=12)
    plt.title('Training and Validation Loss Over Time', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train Improved U-Net for HipMRI segmentation')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to HipMRI_2D dataset')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--base_channels', type=int, default=32,
                        help='Base number of channels in U-Net')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save model checkpoints')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Set device
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading dataset...")
    train_loader, val_loader, test_loader, num_classes, label_to_ch = get_data_loaders(
        args.data_path,
        batch_size=args.batch_size,
        num_workers=2
    )
    
    # Initialize model
    print(f"\nInitializing Improved U-Net with {num_classes} classes...")
    model = ImprovedUNet2D(
        in_ch=1,
        num_classes=num_classes,
        base=args.base_channels
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")
        
        # Train
        train_loss = run_one_epoch(model, train_loader, optimizer=optimizer, device=device)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = run_one_epoch(model, val_loader, optimizer=None, device=device)
        val_losses.append(val_loss)
        
        print(f"\nEpoch {epoch} Results:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(args.save_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'num_classes': num_classes,
                'label_to_ch': label_to_ch,
            }, save_path)
            print(f"  ✓ Best model saved (val_loss: {val_loss:.4f})")
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses,
                        save_path=os.path.join(args.save_dir, 'training_curves.png'))
    
    # Final evaluation on test set
    print(f"\n{'='*60}")
    print("Evaluating on test set...")
    print(f"{'='*60}")
    
    # Load best model
    checkpoint = torch.load(os.path.join(args.save_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_dice = evaluate_dice(model, test_loader, device=device)
    
    print("\nDice Coefficient per Channel on TEST set:")
    print("-" * 40)
    for ch, dice_val in enumerate(test_dice.tolist()):
        print(f"  Channel {ch}: {dice_val:.4f}")
    
    print(f"\nMean Dice: {test_dice.mean().item():.4f}")
    
    # Identify prostate channel (typically channel 3)
    prostate_ch = 3 if num_classes > 3 else 0
    print(f"\n{'='*60}")
    print(f"PROSTATE Dice (Channel {prostate_ch}): {test_dice[prostate_ch].item():.4f}")
    print(f"{'='*60}")
    
    # Save final results
    results = {
        'test_dice_per_channel': test_dice.tolist(),
        'mean_dice': test_dice.mean().item(),
        'prostate_dice': test_dice[prostate_ch].item(),
        'num_classes': num_classes,
        'label_to_ch': label_to_ch,
    }
    
    import json
    with open(os.path.join(args.save_dir, 'test_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {args.save_dir}")
    
    # Check if requirement is met
    if test_dice[prostate_ch].item() >= 0.75:
        print("\n✓ PROJECT REQUIREMENT MET: Prostate Dice >= 0.75")
    else:
        print("\n✗ PROJECT REQUIREMENT NOT MET: Prostate Dice < 0.75")


if __name__ == '__main__':
    main()