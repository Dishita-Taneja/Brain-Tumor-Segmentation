import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from src.config import MODEL_SAVE_PATH, LOG_CSV_PATH, RESULTS_DIR, POS_WEIGHT
from tqdm import tqdm

class DiceLoss(nn.Module):
    """
    Soft Dice Loss.
    Formula: 1 - (2 * |P \cap T| + smooth) / (|P| + |T| + smooth)
    """
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, y_pred, y_true):
        # y_pred: sigmoid activations, shape (B, C, D, H, W)
        # y_true: binary targets, shape (B, C, D, H, W)
        batch_size = y_pred.size(0)
        num_channels = y_pred.size(1)
        
        # Flatten spatial dimensions
        y_pred_flat = y_pred.view(batch_size, num_channels, -1)
        y_true_flat = y_true.view(batch_size, num_channels, -1)
        
        intersection = torch.sum(y_pred_flat * y_true_flat, dim=2)
        denominator = torch.sum(y_pred_flat, dim=2) + torch.sum(y_true_flat, dim=2)
        
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        
        # Return average loss across all channels and batch
        return 1.0 - torch.mean(dice)

class BCEDiceLoss(nn.Module):
    """
    Combined BCE + Dice Loss.
    BCE handles voxel-wise classification, while Dice loss addresses structural overlaps 
    and class imbalance.
    """
    def __init__(self, bce_weight=1.0, dice_weight=1.0, pos_weight=None):
        super().__init__()
        if pos_weight is not None:
            if isinstance(pos_weight, (list, np.ndarray)):
                pos_weight = torch.tensor(pos_weight, dtype=torch.float32)
            # Reshape to (1, C, 1, 1, 1) for 5D tensor broadcasting
            pos_weight = pos_weight.view(1, -1, 1, 1, 1)
            self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        
    def forward(self, logits, targets):
        if self.bce.pos_weight is not None:
            self.bce.pos_weight = self.bce.pos_weight.to(logits.device)
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        dice_loss = self.dice(probs, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def compute_dice_coefficient(y_pred, y_true, threshold=0.5):
    """
    Vectorized calculation of the Dice coefficient (WT, TC, ET) per sample.
    If both prediction and ground truth are empty, returns 1.0.
    
    y_pred: tensor of shape (B, C, D, H, W) containing logits
    y_true: tensor of shape (B, C, D, H, W) containing binary ground truth
    """
    with torch.no_grad():
        # Always apply sigmoid to logits to get probabilities
        y_pred = torch.sigmoid(y_pred)
            
        y_pred_bin = (y_pred > threshold).float()
        y_true = y_true.float()
        
        # Flatten spatial dimensions -> (B, C, -1)
        p = y_pred_bin.view(y_pred_bin.size(0), y_pred_bin.size(1), -1)
        t = y_true.view(y_true.size(0), y_true.size(1), -1)
        
        intersection = torch.sum(p * t, dim=2)
        sum_p = torch.sum(p, dim=2)
        sum_t = torch.sum(t, dim=2)
        
        # Compute dice
        dice = (2.0 * intersection) / (sum_p + sum_t)
        
        # Handle case where both target and prediction are empty:
        # 0/0 division results in NaN in PyTorch. In clinical settings, predicting empty 
        # when it is indeed empty is a perfect prediction, so we map NaN to 1.0.
        dice[torch.isnan(dice)] = 1.0
        
        return dice.cpu().numpy()  # Return array of shape (B, C)

def train_epoch(model, dataloader, optimizer, criterion, device):
    """
    Runs one training epoch with tqdm progress logging.
    """
    model.train()
    epoch_loss = 0.0
    
    progress_bar = tqdm(dataloader, desc="  Training", leave=False)
    for idx, (images, labels) in enumerate(progress_bar):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        loss_val = loss.item()
        epoch_loss += loss_val
        progress_bar.set_postfix(loss=f"{loss_val:.4f}")
        
    return epoch_loss / len(dataloader)

def validate_epoch(model, dataloader, criterion, device):
    """
    Runs validation over the validation dataloader.
    Returns:
    - Average validation loss
    - Mean Dice scores for [WT, TC, ET]
    """
    model.eval()
    epoch_loss = 0.0
    
    all_dice_scores = []
    
    progress_bar = tqdm(dataloader, desc="  Validating", leave=False)
    with torch.no_grad():
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            
            logits = model(images)
            loss = criterion(logits, labels)
            loss_val = loss.item()
            epoch_loss += loss_val
            
            # Compute dice coefficient (B, C)
            dice_scores = compute_dice_coefficient(logits, labels)
            all_dice_scores.append(dice_scores)
            progress_bar.set_postfix(loss=f"{loss_val:.4f}")
            
    # Concatenate all batches -> shape (Total_Val_Samples, 3)
    all_dice_scores = np.concatenate(all_dice_scores, axis=0)
    mean_dice_per_class = np.mean(all_dice_scores, axis=0) # [WT_dice, TC_dice, ET_dice]
    
    return epoch_loss / len(dataloader), mean_dice_per_class

def plot_training_curves(log_csv_path):
    """
    Plots and saves training loss and validation Dice curves from the logged CSV.
    """
    df = pd.read_csv(log_csv_path)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot Loss
    axes[0].plot(df['epoch'], df['train_loss'], label='Train Loss', color='blue')
    axes[0].plot(df['epoch'], df['val_loss'], label='Val Loss', color='red')
    axes[0].set_title('Training and Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot Dice Scores
    axes[1].plot(df['epoch'], df['val_dice_mean'], label='Mean Val Dice', color='black', linewidth=2)
    axes[1].plot(df['epoch'], df['val_dice_wt'], label='Whole Tumor (WT)', color='green', linestyle='dashed')
    axes[1].plot(df['epoch'], df['val_dice_tc'], label='Tumor Core (TC)', color='orange', linestyle='dashed')
    axes[1].plot(df['epoch'], df['val_dice_et'], label='Enhancing Tumor (ET)', color='red', linestyle='dashed')
    
    axes[1].set_title('Validation Dice Scores')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Dice Coefficient')
    axes[1].set_ylim(0, 1.02)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    curve_path = os.path.join(RESULTS_DIR, "training_curves.png")
    plt.savefig(curve_path, dpi=200)
    plt.close()
    print(f"[Trainer] Saved training curves plot: {curve_path}")

def run_training(model, train_loader, val_loader, epochs, lr, device):
    """
    Trains the 3D U-Net model and handles checkpointing and logging.
    """
    print("=" * 60)
    print("STARTING TRAINING LOOP...")
    print(f"Epochs: {epochs} | Learning Rate: {lr} | Device: {device}")
    print("=" * 60)

    # Initialize optimizer, scheduler, loss criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = BCEDiceLoss(bce_weight=1.0, dice_weight=1.0, pos_weight=POS_WEIGHT)
    
    best_mean_dice = 0.0
    log_data = []
    
    for epoch in range(1, epochs + 1):
        # 1. Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # 2. Validate
        val_loss, val_dice_classes = validate_epoch(model, val_loader, criterion, device)
        mean_val_dice = np.mean(val_dice_classes)
        
        # 3. Print status
        print(f"Epoch {epoch:02d}/{epochs:02d} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Mean Val Dice: {mean_val_dice:.4f} "
              f"(WT: {val_dice_classes[0]:.4f}, TC: {val_dice_classes[1]:.4f}, ET: {val_dice_classes[2]:.4f})")
        
        # 4. Log stats
        log_entry = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_dice_mean': mean_val_dice,
            'val_dice_wt': val_dice_classes[0],
            'val_dice_tc': val_dice_classes[1],
            'val_dice_et': val_dice_classes[2]
        }
        log_data.append(log_entry)
        
        # Save logs to CSV on every epoch to prevent loss on interruption
        pd.DataFrame(log_data).to_csv(LOG_CSV_PATH, index=False)
        
        # Save a checkpoint of the latest epoch after every epoch to prevent loss of progress
        epoch_checkpoint_path = os.path.join(os.path.dirname(MODEL_SAVE_PATH), "latest_checkpoint.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_mean_dice': best_mean_dice,
            'log_data': log_data
        }, epoch_checkpoint_path)
        print(f"  --> Saved latest epoch checkpoint to {epoch_checkpoint_path}")
        
        # 5. Checkpoint (Save best model)
        if mean_val_dice > best_mean_dice:
            best_mean_dice = mean_val_dice
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  --> Best model checkpoint saved! (New Best Val Dice: {best_mean_dice:.4f})")
            
    print("\nTraining completed successfully!")
    print(f"Best validation mean Dice score: {best_mean_dice:.4f}")
    print(f"Model saved to: {MODEL_SAVE_PATH}")
    print(f"Log saved to: {LOG_CSV_PATH}")
    
    # Plot and save training curves
    plot_training_curves(LOG_CSV_PATH)
    print("=" * 60 + "\n")
