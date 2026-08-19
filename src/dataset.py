import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from src.config import PREPROCESSED_DATA_DIR, PATCH_SHAPE

def pad_and_crop_3d(image, label, patch_shape, is_train=True):
    """
    Pads the image and label if their dimensions are smaller than patch_shape,
    then crops a patch of patch_shape.
    image: torch.Tensor of shape (C, D, H, W)
    label: torch.Tensor of shape (C_label, D, H, W)
    patch_shape: tuple of (D_out, H_out, W_out)
    is_train: if True, performs random crop, else performs center crop.
    """
    C_img, D, H, W = image.shape
    C_lbl, _, _, _ = label.shape
    D_out, H_out, W_out = patch_shape

    # 1. Pad if necessary
    pad_d = max(0, D_out - D)
    pad_h = max(0, H_out - H)
    pad_w = max(0, W_out - W)

    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        # Pad with 0 for image and label
        # PyTorch pad syntax: (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
        # corresponds to (W, H, D)
        pad_width = (pad_w // 2, pad_w - pad_w // 2,
                     pad_h // 2, pad_h - pad_h // 2,
                     pad_d // 2, pad_d - pad_d // 2)
        image = F.pad(image, pad_width, mode='constant', value=0)
        label = F.pad(label, pad_width, mode='constant', value=0)
        
        # Update shapes
        _, D, H, W = image.shape

    # 2. Crop
    if is_train:
        # Random crop
        d_start = random.randint(0, D - D_out)
        h_start = random.randint(0, H - H_out)
        w_start = random.randint(0, W - W_out)
    else:
        # Center crop
        d_start = (D - D_out) // 2
        h_start = (H - H_out) // 2
        w_start = (W - W_out) // 2

    image_crop = image[:, d_start:d_start+D_out, h_start:h_start+H_out, w_start:w_start+W_out]
    label_crop = label[:, d_start:d_start+D_out, h_start:h_start+H_out, w_start:w_start+W_out]

    return image_crop, label_crop

class BraTS3DDataset(Dataset):
    """
    BraTS 3D Dataset class. Loads preprocessed compressed numpy arrays (.npz)
    containing 'image' of shape (4, 128, 128, 128) and 'label' of shape (3, 128, 128, 128).
    """
    def __init__(self, patient_ids, is_train=False):
        existing_ids = []
        missing_count = 0
        for p_id in patient_ids:
            file_path = os.path.join(PREPROCESSED_DATA_DIR, f"{p_id}_preprocessed.npz")
            if os.path.exists(file_path):
                existing_ids.append(p_id)
            else:
                missing_count += 1
                
        if missing_count > 0:
            print(f"[Dataset] Warning: {missing_count} patient files are missing preprocessed data on disk. "
                  f"Using {len(existing_ids)} available files.")
                  
        if len(existing_ids) == 0:
            raise FileNotFoundError(
                f"None of the requested patient files are preprocessed in '{PREPROCESSED_DATA_DIR}'.\n"
                f"Please run the preprocessing step first: python3 run_pipeline.py --step preprocess"
            )
            
        self.patient_ids = existing_ids
        self.is_train = is_train

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        p_id = self.patient_ids[idx]
        file_path = os.path.join(PREPROCESSED_DATA_DIR, f"{p_id}_preprocessed.npz")
        
        # Load the preprocessed file
        data = np.load(file_path)
        image = data['image']  # Shape: (4, 128, 128, 128)
        label = data['label']  # Shape: (3, 128, 128, 128)

        # Convert to torch Tensors
        image_tensor = torch.from_numpy(image).float()
        label_tensor = torch.from_numpy(label).float()

        # Apply 3D patch cropping
        image_tensor, label_tensor = pad_and_crop_3d(image_tensor, label_tensor, PATCH_SHAPE, is_train=self.is_train)

        # Data augmentation (only during training)
        if self.is_train:
            # 3D spatial flips along depth (dim=1), height (dim=2), and width (dim=3)
            # Dims of tensors are (Channel, D, H, W)
            if random.random() > 0.5:
                image_tensor = torch.flip(image_tensor, dims=[1])
                label_tensor = torch.flip(label_tensor, dims=[1])
            if random.random() > 0.5:
                image_tensor = torch.flip(image_tensor, dims=[2])
                label_tensor = torch.flip(label_tensor, dims=[2])
            if random.random() > 0.5:
                image_tensor = torch.flip(image_tensor, dims=[3])
                label_tensor = torch.flip(label_tensor, dims=[3])

        return image_tensor, label_tensor

def get_dataloader(patient_ids, batch_size=1, shuffle=True, is_train=False, num_workers=0):
    """
    Returns a DataLoader for the BraTS3DDataset.
    """
    dataset = BraTS3DDataset(patient_ids, is_train=is_train)
    
    # We set pin_memory=True if running on CUDA, on Apple M1 MPS pin_memory is not strictly necessary but harmless.
    # Set num_workers=0 or 2 depending on environment capabilities to avoid multiprocess overhead on macOS.
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False
    )
    return dataloader
