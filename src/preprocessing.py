import os
import json
import random
import glob
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from src.config import (
    RAW_DATA_DIR, 
    PREPROCESSED_DATA_DIR, 
    SPLITS_JSON, 
    INPUT_SHAPE,
    RANDOM_SEED,
    TRAIN_VAL_SPLIT,
    RESIZE_IN_PREPROCESS
)

def get_train_val_splits(valid_patients, train_ratio=TRAIN_VAL_SPLIT):
    """
    Splits the valid patients into train and validation sets.
    Saves the split to splits.json so that it remains consistent.
    If splits.json already exists, loads and returns it.
    """
    if os.path.exists(SPLITS_JSON):
        print(f"[Preprocessing] Loading existing splits from {SPLITS_JSON}")
        with open(SPLITS_JSON, 'r') as f:
            splits = json.load(f)
        return splits['train'], splits['val']

    print(f"[Preprocessing] Creating new train/validation splits (Ratio: {train_ratio})")
    
    # Shuffle using fixed random seed for reproducibility
    random.seed(RANDOM_SEED)
    shuffled_patients = list(valid_patients)
    random.shuffle(shuffled_patients)
    
    split_idx = int(len(shuffled_patients) * train_ratio)
    train_patients = shuffled_patients[:split_idx]
    val_patients = shuffled_patients[split_idx:]
    
    splits = {
        'train': train_patients,
        'val': val_patients
    }
    
    with open(SPLITS_JSON, 'w') as f:
        json.dump(splits, f, indent=4)
        
    print(f"[Preprocessing] Split saved. Train: {len(train_patients)}, Val: {len(val_patients)}")
    return train_patients, val_patients

def preprocess_patient(patient_id):
    """
    Loads, crops, normalizes, resizes, and remaps labels for a single patient.
    Saves the output to PREPROCESSED_DATA_DIR as a compressed .npz file.
    """
    patient_dir = os.path.join(RAW_DATA_DIR, patient_id)
    output_file = os.path.join(PREPROCESSED_DATA_DIR, f"{patient_id}_preprocessed.npz")
    
    # Check if already preprocessed
    if os.path.exists(output_file):
        return True

    try:
        # Load NIfTI files
        flair_img = nib.load(os.path.join(patient_dir, f"{patient_id}_flair.nii"))
        t1_img = nib.load(os.path.join(patient_dir, f"{patient_id}_t1.nii"))
        t1ce_img = nib.load(os.path.join(patient_dir, f"{patient_id}_t1ce.nii"))
        t2_img = nib.load(os.path.join(patient_dir, f"{patient_id}_t2.nii"))
        seg_img = nib.load(os.path.join(patient_dir, f"{patient_id}_seg.nii"))

        flair = flair_img.get_fdata().astype(np.float32)
        t1 = t1_img.get_fdata().astype(np.float32)
        t1ce = t1ce_img.get_fdata().astype(np.float32)
        t2 = t2_img.get_fdata().astype(np.float32)
        seg = seg_img.get_fdata().astype(np.float32)

        # 1. Bounding box cropping based on brain mask (flair > 0)
        brain_mask = flair > 0
        if not np.any(brain_mask):
            # Fallback if flair is completely empty (unlikely)
            brain_mask = (t1 > 0) | (t1ce > 0) | (t2 > 0)
            
        coords = np.argwhere(brain_mask)
        if len(coords) == 0:
            raise ValueError(f"Patient {patient_id} has empty brain volumes.")
            
        x_min, y_min, z_min = coords.min(axis=0)
        x_max, y_max, z_max = coords.max(axis=0)

        # Crop modalities and segmentation
        flair_crop = flair[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]
        t1_crop = t1[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]
        t1ce_crop = t1ce[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]
        t2_crop = t2[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]
        seg_crop = seg[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]
        crop_mask = brain_mask[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]

        # 2. Z-score normalization within the brain region
        def zscore_normalize(img_crop, mask):
            norm_img = np.zeros_like(img_crop)
            brain_voxels = img_crop[mask]
            if len(brain_voxels) > 0:
                mean = np.mean(brain_voxels)
                std = np.std(brain_voxels)
                if std > 0:
                    norm_img[mask] = (img_crop[mask] - mean) / std
            return norm_img

        flair_norm = zscore_normalize(flair_crop, crop_mask)
        t1_norm = zscore_normalize(t1_crop, crop_mask)
        t1ce_norm = zscore_normalize(t1ce_crop, crop_mask)
        t2_norm = zscore_normalize(t2_crop, crop_mask)

        # 3. Stack modalities into shape (4, H_crop, W_crop, D_crop)
        stacked_img = np.stack([flair_norm, t1_norm, t1ce_norm, t2_norm], axis=0)

        # 4. Remap segmentation labels to WT, TC, ET
        # WT = labels 1, 2, 4
        # TC = labels 1, 4
        # ET = label 4
        wt = np.isin(seg_crop, [1, 2, 4]).astype(np.float32)
        tc = np.isin(seg_crop, [1, 4]).astype(np.float32)
        et = (seg_crop == 4).astype(np.float32)
        stacked_seg = np.stack([wt, tc, et], axis=0)

        # 5. Resize using PyTorch F.interpolate if configured
        if RESIZE_IN_PREPROCESS:
            # Images: trilinear interpolation
            img_tensor = torch.from_numpy(stacked_img).unsqueeze(0)  # Add batch dim -> (1, 4, H, W, D)
            resized_img_tensor = F.interpolate(img_tensor, size=INPUT_SHAPE, mode='trilinear', align_corners=False)
            resized_img = resized_img_tensor.squeeze(0).numpy()

            # Labels: nearest-neighbor to preserve binary labels
            seg_tensor = torch.from_numpy(stacked_seg).unsqueeze(0)  # Add batch dim -> (1, 3, H, W, D)
            resized_seg_tensor = F.interpolate(seg_tensor, size=INPUT_SHAPE, mode='nearest')
            resized_seg = resized_seg_tensor.squeeze(0).numpy().astype(np.uint8)
        else:
            resized_img = stacked_img
            resized_seg = stacked_seg.astype(np.uint8)

        # 6. Save as compressed .npz file
        np.savez_compressed(output_file, image=resized_img, label=resized_seg)
        return True

    except Exception as e:
        print(f"Error preprocessing patient {patient_id}: {str(e)}")
        return False

def preprocess_dataset(patient_ids, limit=None):
    """
    Loops through all patients and runs the preprocessing step.
    Allows a limit parameter to preprocess only a subset of patients (e.g. for quick test mode).
    """
    print("=" * 60)
    print("STARTING OFFLINE PREPROCESSING...")
    print(f"Target shape: {INPUT_SHAPE}")
    print("=" * 60)
    
    if limit is not None:
        patient_ids = patient_ids[:limit]
        print(f"Limiting preprocessing to the first {limit} patients.")

    os.makedirs(PREPROCESSED_DATA_DIR, exist_ok=True)
    
    total = len(patient_ids)
    success_count = 0
    
    for idx, p_id in enumerate(patient_ids):
        success = preprocess_patient(p_id)
        if success:
            success_count += 1
            
        if (idx + 1) % 10 == 0 or (idx + 1) == total:
            print(f"Preprocessed {idx + 1}/{total} patients...")
            
    print(f"Successfully preprocessed {success_count}/{total} patients.")
    print(f"Saved to: {PREPROCESSED_DATA_DIR}")
    print("=" * 60 + "\n")
    return success_count

if __name__ == "__main__":
    # Test script with a single patient
    print("Testing preprocessing script...")
    p_id = "BraTS20_Training_001"
    success = preprocess_patient(p_id)
    print(f"Test preprocessing for {p_id}: {'SUCCESS' if success else 'FAILED'}")
