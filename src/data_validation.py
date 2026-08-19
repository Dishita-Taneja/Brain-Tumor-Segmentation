import os
import glob
import nibabel as nib
import numpy as np
from src.config import RAW_DATA_DIR

def validate_dataset(verbose=True):
    """
    Validates the BraTS2020 training dataset.
    Checks:
    - Existence of 5 files per patient (flair, t1, t1ce, t2, seg)
    - File readability using nibabel
    - Mismatched shapes across modalities for the same patient
    - Voxel spacing (zooms) consistency across patients
    - Orientation consistency across patients
    """
    print("=" * 60)
    print("STARTING DATA VALIDATION...")
    print(f"Scanning raw data directory: {RAW_DATA_DIR}")
    print("=" * 60)
    
    if not os.path.exists(RAW_DATA_DIR):
        print(f"Error: Raw data directory '{RAW_DATA_DIR}' does not exist.")
        return [], []

    # Get all patient folders
    patient_paths = sorted([
        d for d in glob.glob(os.path.join(RAW_DATA_DIR, "BraTS20_Training_*")) 
        if os.path.isdir(d)
    ])
    
    total_patients = len(patient_paths)
    print(f"Found {total_patients} patient directories.")

    valid_patients = []
    invalid_patients = []
    
    # Track spatial properties to check consistency
    unique_shapes = {}
    unique_spacings = {}
    unique_orientations = {}

    modalities = ['flair', 't1', 't1ce', 't2', 'seg']

    for idx, p_path in enumerate(patient_paths):
        p_id = os.path.basename(p_path)
        patient_errors = []
        patient_info = {}

        # 1. Check for presence of all 5 files
        files = {}
        for mod in modalities:
            file_name = f"{p_id}_{mod}.nii"
            file_path = os.path.join(p_path, file_name)
            if not os.path.exists(file_path):
                # Try .nii.gz just in case, though the dataset is listing .nii files
                gzip_path = file_path + ".gz"
                if os.path.exists(gzip_path):
                    files[mod] = gzip_path
                else:
                    patient_errors.append(f"Missing modality file: {file_name}")
            else:
                files[mod] = file_path

        # If any files are missing, we don't proceed to read them
        if patient_errors:
            invalid_patients.append((p_id, patient_errors))
            if verbose:
                print(f"[{idx+1}/{total_patients}] Patient {p_id} -> FAILED: {'; '.join(patient_errors)}")
            continue

        # 2. Check readability, shapes, spacing, orientation
        shapes = []
        spacings = []
        orientations = []
        read_failed = False

        for mod, f_path in files.items():
            try:
                # Load header first (very fast, no array loading)
                img = nib.load(f_path)
                
                # Check readability of data array (force loading a small slice or check shape)
                shape = img.shape
                shapes.append(shape)
                
                # Spacing (get zooms)
                zooms = img.header.get_zooms()[:3]
                # Round to 4 decimals for float comparison stability
                zooms = tuple(round(z, 4) for z in zooms)
                spacings.append(zooms)
                
                # Orientation
                orientation = nib.aff2axcodes(img.affine)
                orientations.append(orientation)
                
                # Verify we can read the actual data array without error
                # We can just read shape or a single voxel value to verify header/data block consistency
                _ = img.dataobj[0, 0, 0]
                
            except Exception as e:
                patient_errors.append(f"Read error on {mod} file ({os.path.basename(f_path)}): {str(e)}")
                read_failed = True
                break

        if read_failed:
            invalid_patients.append((p_id, patient_errors))
            if verbose:
                print(f"[{idx+1}/{total_patients}] Patient {p_id} -> FAILED: {'; '.join(patient_errors)}")
            continue

        # 3. Check for mismatched shapes/spacings/orientations for the SAME patient
        # All modalities for the same patient must have identical spatial attributes
        first_shape = shapes[0]
        first_spacing = spacings[0]
        first_orientation = orientations[0]

        for i, mod in enumerate(modalities[1:], start=1):
            if shapes[i] != first_shape:
                patient_errors.append(f"Mismatched shape: {modalities[i]} is {shapes[i]}, but {modalities[0]} is {first_shape}")
            if spacings[i] != first_spacing:
                patient_errors.append(f"Mismatched spacing: {modalities[i]} is {spacings[i]}, but {modalities[0]} is {first_spacing}")
            if orientations[i] != first_orientation:
                patient_errors.append(f"Mismatched orientation: {modalities[i]} is {orientations[i]}, but {modalities[0]} is {first_orientation}")

        if patient_errors:
            invalid_patients.append((p_id, patient_errors))
            if verbose:
                print(f"[{idx+1}/{total_patients}] Patient {p_id} -> FAILED: {'; '.join(patient_errors)}")
            continue

        # Store for dataset-wide statistics
        unique_shapes[first_shape] = unique_shapes.get(first_shape, 0) + 1
        unique_spacings[first_spacing] = unique_spacings.get(first_spacing, 0) + 1
        unique_orientations[first_orientation] = unique_orientations.get(first_orientation, 0) + 1

        valid_patients.append(p_id)
        if verbose and (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{total_patients} patients...")

    # Summary Report
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Total directories scanned: {total_patients}")
    print(f"Successfully validated:   {len(valid_patients)}")
    print(f"Failed validation:        {len(invalid_patients)}")
    
    if invalid_patients:
        print("\nPatients with errors:")
        for p_id, errors in invalid_patients:
            print(f"  - {p_id}:")
            for err in errors:
                print(f"    * {err}")
    else:
        print("\nNo corrupted, mismatched, or missing files found! All patients are valid.")

    print("\nDataset-Wide Structural Attributes:")
    print("Unique volume shapes:")
    for sh, count in unique_shapes.items():
        print(f"  - Shape {sh}: found in {count} patients ({count/len(valid_patients)*100:.1f}%)")

    print("Unique voxel spacings (mm):")
    for sp, count in unique_spacings.items():
        print(f"  - Spacing {sp}: found in {count} patients ({count/len(valid_patients)*100:.1f}%)")

    print("Unique orientations:")
    for ori, count in unique_orientations.items():
        print(f"  - Orientation {''.join(ori)}: found in {count} patients ({count/len(valid_patients)*100:.1f}%)")
    print("=" * 60 + "\n")

    return valid_patients, invalid_patients

if __name__ == "__main__":
    validate_dataset()
