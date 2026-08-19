import os
import glob
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from src.config import RAW_DATA_DIR, RESULTS_DIR

def run_eda(num_patients_for_stats=50):
    """
    Runs EDA on the BraTS2020 dataset and saves visual plots to results/.
    """
    print("=" * 60)
    print("STARTING EXPLORATORY DATA ANALYSIS (EDA)...")
    print("=" * 60)

    # Get valid patients list (exclude BraTS20_Training_355)
    patient_paths = sorted([
        d for d in glob.glob(os.path.join(RAW_DATA_DIR, "BraTS20_Training_*")) 
        if os.path.isdir(d) and os.path.basename(d) != "BraTS20_Training_355"
    ])
    
    if not patient_paths:
        print("Error: No training patients found for EDA.")
        return

    sample_patient_path = patient_paths[0]
    p_id = os.path.basename(sample_patient_path)
    print(f"Loading sample patient: {p_id} for slice visualization...")

    # Load modalities for the sample patient
    flair_img = nib.load(os.path.join(sample_patient_path, f"{p_id}_flair.nii"))
    t1_img = nib.load(os.path.join(sample_patient_path, f"{p_id}_t1.nii"))
    t1ce_img = nib.load(os.path.join(sample_patient_path, f"{p_id}_t1ce.nii"))
    t2_img = nib.load(os.path.join(sample_patient_path, f"{p_id}_t2.nii"))
    seg_img = nib.load(os.path.join(sample_patient_path, f"{p_id}_seg.nii"))

    flair = flair_img.get_fdata()
    t1 = t1_img.get_fdata()
    t1ce = t1ce_img.get_fdata()
    t2 = t2_img.get_fdata()
    seg = seg_img.get_fdata()

    # Find the axial slice with the largest tumor area
    # Segment labels are 1 (NCR), 2 (ED), 4 (ET)
    tumor_mask = (seg > 0)
    slice_sums = np.sum(tumor_mask, axis=(0, 1))
    best_slice_idx = np.argmax(slice_sums)
    print(f"Max tumor footprint found on axial slice index: {best_slice_idx} ({slice_sums[best_slice_idx]} tumor voxels)")

    # ----------------- PLOT 1: MODALITIES & OVERLAY -----------------
    print("Generating sample modality slice plots with ground truth overlay...")
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    # Normalize images locally for visualization [0, 1]
    def local_norm(img_slice):
        p_min, p_max = np.percentile(img_slice, (1, 99))
        if p_max - p_min > 0:
            return np.clip((img_slice - p_min) / (p_max - p_min), 0, 1)
        return img_slice

    axes[0].imshow(local_norm(flair[:, :, best_slice_idx]).T, cmap='gray', origin='lower')
    axes[0].set_title("FLAIR")
    axes[0].axis('off')

    axes[1].imshow(local_norm(t1[:, :, best_slice_idx]).T, cmap='gray', origin='lower')
    axes[1].set_title("T1")
    axes[1].axis('off')

    axes[2].imshow(local_norm(t1ce[:, :, best_slice_idx]).T, cmap='gray', origin='lower')
    axes[2].set_title("T1CE (Contrast)")
    axes[2].axis('off')

    axes[3].imshow(local_norm(t2[:, :, best_slice_idx]).T, cmap='gray', origin='lower')
    axes[3].set_title("T2")
    axes[3].axis('off')

    # Segmentation Overlay on FLAIR
    # Create an RGB color mask for overlay
    # Label 1: Red (Necrotic Core), Label 2: Green (Edema), Label 4: Yellow (Enhancing Tumor)
    flair_slice = local_norm(flair[:, :, best_slice_idx]).T
    seg_slice = seg[:, :, best_slice_idx].T
    
    overlay = np.zeros((*flair_slice.shape, 4)) # RGBA
    overlay[seg_slice == 1] = [1.0, 0.0, 0.0, 0.6]  # Red (NCR)
    overlay[seg_slice == 2] = [0.0, 1.0, 0.0, 0.5]  # Green (ED)
    overlay[seg_slice == 4] = [1.0, 1.0, 0.0, 0.7]  # Yellow (ET)

    axes[4].imshow(flair_slice, cmap='gray', origin='lower')
    axes[4].imshow(overlay, origin='lower')
    axes[4].set_title("FLAIR + Seg Mask Overlay")
    axes[4].axis('off')
    
    # Add a custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.6, label='Necrotic Core (Label 1)'),
        Patch(facecolor='green', alpha=0.5, label='Edema (Label 2)'),
        Patch(facecolor='yellow', alpha=0.7, label='Enhancing Tumor (Label 4)')
    ]
    axes[4].legend(handles=legend_elements, loc='upper right', fontsize=8, framealpha=0.8)

    plt.tight_layout()
    plot_path1 = os.path.join(RESULTS_DIR, "eda_sample_slices.png")
    plt.savefig(plot_path1, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_path1}")

    # ----------------- PLOT 2: INTENSITY HISTOGRAMS -----------------
    print("Generating intensity histograms (excluding background zeros)...")
    plt.figure(figsize=(10, 5))
    
    # We sample voxels where FLAIR > 0 to focus on the brain tissue
    brain_idx = flair > 0
    sample_size = 50000  # Sample to make plotting fast
    
    for label, img_data, color in [
        ('FLAIR', flair, 'blue'),
        ('T1', t1, 'orange'),
        ('T1CE', t1ce, 'green'),
        ('T2', t2, 'red')
    ]:
        vals = img_data[brain_idx]
        if len(vals) > sample_size:
            vals = np.random.choice(vals, size=sample_size, replace=False)
        plt.hist(vals, bins=100, alpha=0.5, label=label, color=color, histtype='stepfilled')

    plt.title(f"Intensity Distributions for Patient {p_id} (Brain Voxels only)")
    plt.xlabel("Voxel Intensity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path2 = os.path.join(RESULTS_DIR, "eda_histograms.png")
    plt.savefig(plot_path2, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_path2}")

    # ----------------- PLOT 3: STATISTICS ACROSS DATASET -----------------
    print(f"Scanning {num_patients_for_stats} patients to analyze class imbalance and tumor sizes...")
    
    label_counts = {0: 0, 1: 0, 2: 0, 4: 0}
    tumor_volumes_cm3 = []

    # Let's count class labels and sizes
    for idx, path in enumerate(patient_paths[:num_patients_for_stats]):
        p_name = os.path.basename(path)
        seg_file = os.path.join(path, f"{p_name}_seg.nii")
        
        try:
            seg_data = nib.load(seg_file).get_fdata()
            # 1. Count classes
            unique, counts = np.unique(seg_data, return_counts=True)
            for val, count in zip(unique, counts):
                label_counts[int(val)] = label_counts.get(int(val), 0) + count
                
            # 2. Compute tumor volume
            # Voxel volume is 1x1x1 mm = 1 mm^3 = 0.001 cm^3
            tumor_voxels = np.sum(seg_data > 0)
            volume_cm3 = tumor_voxels * 0.001  # Convert to cm^3 (mL)
            tumor_volumes_cm3.append(volume_cm3)
            
        except Exception as e:
            print(f"Error reading {seg_file}: {e}")

        if (idx + 1) % 10 == 0:
            print(f"  Scanned {idx + 1}/{num_patients_for_stats} patients...")

    # Calculate percentages
    total_voxels = sum(label_counts.values())
    print("\nClass Imbalance Statistics (across sampled patients):")
    label_names = {
        0: "Background (0)",
        1: "Necrotic Core (1)",
        2: "Peritumoral Edema (2)",
        4: "Enhancing Tumor (4)"
    }
    for lbl, count in label_counts.items():
        pct = (count / total_voxels) * 100
        print(f"  - {label_names[lbl]}: {count:,} voxels ({pct:.4f}%)")

    # Plot double statistics: Class distribution (pie/bar) and tumor volume sizes
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Left: Bar chart of class counts (excluding background to make tumor labels readable)
    tumor_labels = [1, 2, 4]
    tumor_counts = [label_counts[lbl] for lbl in tumor_labels]
    tumor_names = [label_names[lbl] for lbl in tumor_labels]
    
    axes[0].bar(tumor_names, tumor_counts, color=['red', 'green', 'yellow'], edgecolor='black', alpha=0.7)
    axes[0].set_title("Voxel Count Distribution (Tumor Classes Only)")
    axes[0].set_ylabel("Total Voxel Count")
    for i, count in enumerate(tumor_counts):
        pct = (count / total_voxels) * 100
        axes[0].text(i, count + (max(tumor_counts) * 0.02), f"{count:,}\n({pct:.3f}%)", ha='center', fontsize=9)
    axes[0].grid(axis='y', alpha=0.3)

    # Right: Tumor size distribution
    axes[1].hist(tumor_volumes_cm3, bins=15, color='purple', edgecolor='black', alpha=0.7)
    axes[1].set_title("Tumor Volume Distribution Across Patients")
    axes[1].set_xlabel("Tumor Volume ($cm^3$ or mL)")
    axes[1].set_ylabel("Number of Patients")
    axes[1].grid(True, alpha=0.3)

    mean_vol = np.mean(tumor_volumes_cm3)
    median_vol = np.median(tumor_volumes_cm3)
    axes[1].axvline(mean_vol, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {mean_vol:.1f} $cm^3$')
    axes[1].axvline(median_vol, color='blue', linestyle='dotted', linewidth=1.5, label=f'Median: {median_vol:.1f} $cm^3$')
    axes[1].legend()

    plt.tight_layout()
    plot_path3 = os.path.join(RESULTS_DIR, "eda_tumor_sizes.png")
    plt.savefig(plot_path3, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_path3}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    # Scan all 368 valid patients for dataset-wide statistics to make it highly accurate
    run_eda(num_patients_for_stats=368)
