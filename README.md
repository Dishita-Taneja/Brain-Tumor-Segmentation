# Brain Tumor Segmentation

It processes multi-modal MRI scans, trains a volumetric segmentation model, evaluates Dice scores, and saves visual prediction overlays.

## Project Overview

The model predicts three tumor regions from four MRI modalities:

- Input modalities: FLAIR, T1, T1CE, T2
- Output masks: Whole Tumor (WT), Tumor Core (TC), Enhancing Tumor (ET)
- Dataset format: NIfTI medical volumes (`.nii`)
- Framework: PyTorch
- Model architecture: 3D U-Net

## Repository Structure

```text
.
├── run_pipeline.py              # Main command-line pipeline entry point
├── src/
│   ├── config.py                # Paths, device selection, and hyperparameters
│   ├── data_validation.py       # Dataset validation utilities
│   ├── dataset.py               # PyTorch Dataset and DataLoader
│   ├── eda.py                   # Exploratory data analysis and plots
│   ├── evaluate.py              # Evaluation metrics and visualization
│   ├── model.py                 # 3D U-Net model definition
│   ├── preprocessing.py         # NIfTI loading, cropping, normalization, labels
│   └── training.py              # Losses, training loop, validation loop
├── DataSet/                     # Local BraTS data, ignored by Git
│   ├── Brain3D/                 # Raw BraTS2020 data
│   ├── preprocessed/            # Local preprocessed `.npz` files
│   └── splits.json              # Local train/validation split
└── results/
    ├── eda_histograms.png
    ├── eda_sample_slices.png
    ├── eda_tumor_sizes.png
    ├── training_curves.png
    ├── training_log.csv
    ├── validation_dice_scores.csv
    └── validation_predictions.png
```

## Technical Stack

The project uses:

- Python
- PyTorch
- NumPy
- pandas
- nibabel
- matplotlib
- tqdm

PyTorch is used for the model, tensor operations, data loading, loss computation, optimization, checkpointing, and inference.

## Dataset

This repository does not include the BraTS data because the raw MRI volumes and preprocessed arrays are large and should be downloaded separately.

- Official BraTS 2020 data page: https://www.med.upenn.edu/cbica/brats2020/data.html
- BraTS 2020 registration/data request page: https://www.med.upenn.edu/cbica/brats2020/registration.html

After downloading the dataset, place it locally under:

```text
DataSet/Brain3D/
├── BraTS2020_TrainingData/
└── BraTS2020_ValidationData/
```

The `DataSet/` directory is ignored by Git so the dataset stays on your machine and is not pushed to GitHub.

## Model Architecture

The model is defined in `src/model.py` as `UNet3D`.

It uses an encoder-decoder structure with skip connections:

```text
Input:  (B, 4, D, H, W)
Output: (B, 3, D, H, W)
```

Main components:

- `nn.Conv3d` for 3D convolution
- `nn.InstanceNorm3d` for normalization
- `nn.LeakyReLU` for activation
- `nn.Upsample` with trilinear interpolation for decoding
- `torch.cat` for skip connections
- Final `1x1x1` convolution for class logits

The model outputs raw logits. Sigmoid activation is applied inside the loss function and during inference.

## Data Processing

Preprocessing is handled in `src/preprocessing.py`.

For each patient, the pipeline:

1. Loads FLAIR, T1, T1CE, T2, and segmentation NIfTI files.
2. Crops the image using a brain mask.
3. Applies z-score normalization inside the brain region.
4. Stacks the four MRI modalities into a 4-channel volume.
5. Converts BraTS labels into three overlapping binary masks:
   - WT: labels `1`, `2`, `4`
   - TC: labels `1`, `4`
   - ET: label `4`
6. Saves the result as compressed `.npz` files.

During training, `src/dataset.py` loads the preprocessed arrays, applies 3D patch cropping, and performs random flips for augmentation.

## Training

Training is implemented in `src/training.py`.

The training setup uses:

- Optimizer: Adam
- Loss: BCEWithLogitsLoss + Dice Loss
- Metric: Dice coefficient
- Batch size: 1
- Default epochs: 40
- Default learning rate: `2e-4`

The model saves:

- `best_model.pth`: best model based on validation mean Dice
- `latest_checkpoint.pth`: latest epoch checkpoint
- `results/training_log.csv`: epoch-wise training logs
- `results/training_curves.png`: loss and Dice plots

## Evaluation

Evaluation is handled in `src/evaluate.py`.

The script:

1. Loads the best model weights from `best_model.pth`.
2. Runs inference on the validation patients.
3. Computes Dice scores for WT, TC, and ET.
4. Saves patient-wise scores to `results/validation_dice_scores.csv`.
5. Saves prediction overlays to `results/validation_predictions.png`.

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install torch numpy pandas nibabel matplotlib tqdm
```

If you are using Apple Silicon, the code automatically selects the MPS backend when available:

```python
torch.device("mps")
```

Otherwise, it falls back to CPU.

## How to Run

Run a quick end-to-end test on a small subset:

```bash
python3 run_pipeline.py --step quick-test
```

Validate the dataset:

```bash
python3 run_pipeline.py --step validate
```

Run exploratory data analysis:

```bash
python3 run_pipeline.py --step eda
```

Preprocess the dataset:

```bash
python3 run_pipeline.py --step preprocess
```

Train the model:

```bash
python3 run_pipeline.py --step train
```

Evaluate the model:

```bash
python3 run_pipeline.py --step eval
```

Run the full pipeline:

```bash
python3 run_pipeline.py --step all
```

Train on a smaller subset:

```bash
python3 run_pipeline.py --step all --subset-size 50
```

Override the number of epochs:

```bash
python3 run_pipeline.py --step train --epochs 10
```

## Pipeline Workflow

```text
Raw BraTS2020 NIfTI files
        ↓
Dataset validation
        ↓
EDA
        ↓
Preprocessing and train/validation split
        ↓
PyTorch DataLoader
        ↓
3D U-Net training
        ↓
Checkpoint saving
        ↓
Validation Dice evaluation
        ↓
Prediction visualization
```

## Configuration

Important settings are stored in `src/config.py`:

```python
BATCH_SIZE = 1
LEARNING_RATE = 2e-4
NUM_EPOCHS = 40
PATCH_SHAPE = (96, 96, 96)
NUM_MODALITIES = 4
NUM_CLASSES = 3
POS_WEIGHT = [5.0, 5.0, 5.0]
```

Paths for raw data, preprocessed data, checkpoints, and results are also configured there.

## Notes

- The pipeline is designed for memory-constrained 3D segmentation training.
- Instance normalization is used instead of batch normalization because 3D medical segmentation often requires very small batch sizes.
- The training code includes an out-of-memory fallback that reduces model width from `base_filters=32` to `base_filters=24`.
- Patient `BraTS20_Training_355` is explicitly excluded because it is missing `seg.nii`.

## Outputs

After successful training and evaluation, the main outputs are:

- Trained weights: `best_model.pth`
- Latest checkpoint: `latest_checkpoint.pth`
- Training log: `results/training_log.csv`
- Training plots: `results/training_curves.png`
- Validation metrics: `results/validation_dice_scores.csv`
- Prediction visualization: `results/validation_predictions.png`

## Summary

This project uses PyTorch to build and train a custom 3D U-Net for multi-modal MRI brain tumor segmentation. The complete workflow includes data validation, EDA, preprocessing, model training, checkpointing, evaluation, and visualization.
