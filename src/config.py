import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch

# Paths
BASE_DIR = "/Users/dishitataneja/Documents/brain"
RAW_DATA_DIR = os.path.join(BASE_DIR, "DataSet/Brain3D/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData")
PREPROCESSED_DATA_DIR = os.path.join(BASE_DIR, "DataSet/preprocessed")
SPLITS_JSON = os.path.join(BASE_DIR, "DataSet/splits.json")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "best_model.pth")
LOG_CSV_PATH = os.path.join(RESULTS_DIR, "training_log.csv")

# Create directories if they do not exist
os.makedirs(PREPROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Hyperparameters
INPUT_SHAPE = (128, 128, 128)  # Target resized shape (Depth, Height, Width)
PATCH_SHAPE = (96, 96, 96)     # Training/Validation patch shape (Depth, Height, Width)
RESIZE_IN_PREPROCESS = False   # Set to False to preserve raw cropped resolution
NUM_MODALITIES = 4             # FLAIR, T1, T1CE, T2
NUM_CLASSES = 3                # WT, TC, ET (overlapping target channels)

# Training Hyperparameters
BATCH_SIZE = 1                 # Keeping batch size 1 for memory safety on M1 GPU
LEARNING_RATE = 2e-4           # Default learning rate for Adam optimizer
NUM_EPOCHS = 40                # Standard number of epochs for full run
VAL_INTERVAL = 1               # How often to run validation (in epochs)
RANDOM_SEED = 42
TRAIN_VAL_SPLIT = 0.8
POS_WEIGHT = [5.0, 5.0, 5.0]   # Class weights for BCE component of loss


# Device selection: MPS if available, otherwise CPU
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"[Config] Using device: {DEVICE}")
print(f"[Config] Raw Data Dir: {RAW_DATA_DIR}")
print(f"[Config] Preprocessed Data Dir: {PREPROCESSED_DATA_DIR}")
print(f"[Config] Results Dir: {RESULTS_DIR}")
