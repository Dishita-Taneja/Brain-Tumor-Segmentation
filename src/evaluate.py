import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from src.config import MODEL_SAVE_PATH, RESULTS_DIR, DEVICE
from src.training import compute_dice_coefficient
from src.dataset import BraTS3DDataset

def evaluate_model(model, val_patients):
    """
    Evaluates the model on the validation patient set.
    Calculates Dice scores for WT, TC, and ET, and saves visual overlays of predictions vs ground truth.
    """
    print("=" * 60)
    print("STARTING MODEL EVALUATION...")
    print(f"Loading weights from: {MODEL_SAVE_PATH}")
    print("=" * 60)
    
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"Error: Model checkpoint file '{MODEL_SAVE_PATH}' not found.")
        return
        
    # Load model state
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    val_dataset = BraTS3DDataset(val_patients, is_train=False)
    # Use only the successfully preprocessed patient IDs for evaluation
    val_patients = val_dataset.patient_ids
    
    all_dice_scores = []
    
    print(f"Running inference on {len(val_patients)} validation patients...")
    
    with torch.no_grad():
        for idx, p_id in enumerate(val_patients):
            image, label = val_dataset[idx]
            
            # Add batch dimension -> (1, 4, 128, 128, 128)
            image_batch = image.unsqueeze(0).to(DEVICE)
            label_batch = label.unsqueeze(0).to(DEVICE)
            
            # Forward pass
            logits = model(image_batch)
            
            # Compute dice (1, 3)
            dice = compute_dice_coefficient(logits, label_batch)
            all_dice_scores.append(dice)
            
            if (idx + 1) % 10 == 0 or (idx + 1) == len(val_patients):
                print(f"  Processed {idx + 1}/{len(val_patients)} patients...")

    all_dice_scores = np.concatenate(all_dice_scores, axis=0) # Shape: (Num_Val, 3)
    
    # Calculate statistics
    mean_dice = np.mean(all_dice_scores, axis=0)
    std_dice = np.std(all_dice_scores, axis=0)
    
    print("\n" + "=" * 40)
    print("VALIDATION METRICS SUMMARY")
    print("=" * 40)
    print(f"Whole Tumor (WT) Dice:     {mean_dice[0]:.4f} ± {std_dice[0]:.4f}")
    print(f"Tumor Core (TC) Dice:      {mean_dice[1]:.4f} ± {std_dice[1]:.4f}")
    print(f"Enhancing Tumor (ET) Dice: {mean_dice[2]:.4f} ± {std_dice[2]:.4f}")
    print(f"Overall Mean Dice:         {np.mean(mean_dice):.4f}")
    print("=" * 40 + "\n")
    
    # Save a CSV with validation dice scores per patient
    import pandas as pd
    df_results = pd.DataFrame(all_dice_scores, columns=['WT_Dice', 'TC_Dice', 'ET_Dice'])
    df_results.insert(0, 'Patient_ID', val_patients)
    val_csv_path = os.path.join(RESULTS_DIR, "validation_dice_scores.csv")
    df_results.to_csv(val_csv_path, index=False)
    print(f"[Eval] Patient-wise validation scores saved to: {val_csv_path}")

    # Generate a visual comparison overlay for a patient with a relatively large tumor
    # Let's find a patient in the validation list and visualize
    visualize_predictions(model, val_dataset, val_patients, sample_idx=0)

def visualize_predictions(model, val_dataset, val_patients, sample_idx=0):
    """
    Generates and saves a visualization showing a side-by-side comparison of 
    Ground Truth overlays vs. Model Prediction overlays.
    """
    p_id = val_patients[sample_idx]
    image, label = val_dataset[sample_idx]
    
    image_batch = image.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(image_batch)
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy() # Shape: (3, 128, 128, 128)
        
    label = label.numpy() # Shape: (3, 128, 128, 128)
    image = image.numpy() # Shape: (4, 128, 128, 128)
    
    # Find the axial slice with the largest ground truth tumor footprint in preprocessed image
    wt_gt = label[0] # WT channel
    slice_sums = np.sum(wt_gt, axis=(0, 1))
    best_slice_idx = np.argmax(slice_sums)
    
    # If the tumor is empty or very small, just pick the middle slice
    if slice_sums[best_slice_idx] < 10:
        best_slice_idx = image.shape[3] // 2
        
    print(f"[Eval] Visualizing slice {best_slice_idx} for patient {p_id}...")

    # Extract slices
    # Modalities: FLAIR is index 0
    flair_slice = image[0, :, :, best_slice_idx].T
    
    gt_wt = label[0, :, :, best_slice_idx].T
    gt_tc = label[1, :, :, best_slice_idx].T
    gt_et = label[2, :, :, best_slice_idx].T
    
    pred_wt = (probs[0, :, :, best_slice_idx].T > 0.5).astype(np.float32)
    pred_tc = (probs[1, :, :, best_slice_idx].T > 0.5).astype(np.float32)
    pred_et = (probs[2, :, :, best_slice_idx].T > 0.5).astype(np.float32)

    # Helper function to map overlapping channels to mutually-exclusive labels for visualization
    def map_to_rgb_mask(wt, tc, et):
        mask = np.zeros_like(wt, dtype=np.uint8)
        mask[wt > 0.5] = 2  # Edema (Label 2) -> Green
        mask[tc > 0.5] = 1  # Necrotic Core (Label 1) -> Red
        mask[et > 0.5] = 4  # Enhancing Tumor (Label 4) -> Yellow
        
        rgba = np.zeros((*wt.shape, 4))
        rgba[mask == 1] = [1.0, 0.0, 0.0, 0.6]  # Red
        rgba[mask == 2] = [0.0, 1.0, 0.0, 0.5]  # Green
        rgba[mask == 4] = [1.0, 1.0, 0.0, 0.7]  # Yellow
        return rgba

    gt_rgba = map_to_rgb_mask(gt_wt, gt_tc, gt_et)
    pred_rgba = map_to_rgb_mask(pred_wt, pred_tc, pred_et)

    # Local normalization of FLAIR slice for display
    flair_disp = (flair_slice - flair_slice.min()) / (flair_slice.max() - flair_slice.min() + 1e-8)
    flair_disp = np.clip(flair_disp, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Raw FLAIR Slice
    axes[0].imshow(flair_disp, cmap='gray', origin='lower')
    axes[0].set_title(f"FLAIR (Slice {best_slice_idx})")
    axes[0].axis('off')

    # 2. Ground Truth Overlay
    axes[1].imshow(flair_disp, cmap='gray', origin='lower')
    axes[1].imshow(gt_rgba, origin='lower')
    axes[1].set_title("Ground Truth Overlay")
    axes[1].axis('off')

    # 3. Model Prediction Overlay
    axes[2].imshow(flair_disp, cmap='gray', origin='lower')
    axes[2].imshow(pred_rgba, origin='lower')
    axes[2].set_title("Model Prediction Overlay")
    axes[2].axis('off')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.6, label='Necrotic Core'),
        Patch(facecolor='green', alpha=0.5, label='Edema'),
        Patch(facecolor='yellow', alpha=0.7, label='Enhancing Tumor')
    ]
    axes[2].legend(handles=legend_elements, loc='upper right', fontsize=8, framealpha=0.8)

    plt.tight_layout()
    pred_plot_path = os.path.join(RESULTS_DIR, "validation_predictions.png")
    plt.savefig(pred_plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Eval] Saved visual prediction overlays to: {pred_plot_path}")
