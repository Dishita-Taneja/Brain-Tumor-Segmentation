import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import argparse
import torch
import json
from src.config import DEVICE, NUM_EPOCHS, LEARNING_RATE, BATCH_SIZE, SPLITS_JSON, MODEL_SAVE_PATH
from src.data_validation import validate_dataset
from src.eda import run_eda
from src.preprocessing import get_train_val_splits, preprocess_dataset
from src.dataset import get_dataloader
from src.model import UNet3D
from src.training import run_training
from src.evaluate import evaluate_model

def get_args():
    parser = argparse.ArgumentParser(description="BraTS2020 3D Brain Tumor Segmentation Pipeline")
    parser.add_argument(
        "--step", 
        type=str, 
        default="quick-test",
        choices=["validate", "eda", "preprocess", "train", "eval", "all", "quick-test"],
        help="Pipeline step to run: 'validate' (data cleaning), 'eda' (plotting & distributions), "
             "'preprocess' (offline normalization/scaling/cropping), 'train' (model training), "
             "'eval' (metrics & visualization), 'all' (runs full pipeline), or 'quick-test' (debug run)."
    )
    parser.add_argument(
        "--quick-test", 
        action="store_true",
        help="Enables quick test mode: runs on 5 patients for 2 epochs using a small network size."
    )
    parser.add_argument(
        "--subset-size", 
        type=int, 
        default=None,
        help="Optional limit on the number of patients to preprocess/train for standard runs (e.g., 50)."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the number of epochs to train."
    )
    return parser.parse_args()

def main():
    args = get_args()
    
    # Handle implicit quick-test selection
    is_quick_test = args.quick_test or (args.step == "quick-test")
    if is_quick_test:
        print("\n" + "*" * 60)
        print("RUNNING IN QUICK TEST MODE")
        print("This will run a 5-patient, 2-epoch subset with a lightweight U-Net.")
        print("*" * 60 + "\n")
        args.step = "all"  # Execute all stages on the subset

    # Step 1: Data Validation
    if args.step in ["validate", "all"]:
        valid_patients, invalid_patients = validate_dataset(verbose=True)
    else:
        # Load patient list without printing full details
        valid_patients, _ = validate_dataset(verbose=False)
        
    # Exclude invalid patients explicitly
    # Patient BraTS20_Training_355 is missing seg.nii
    valid_patients = [p for p in valid_patients if p != "BraTS20_Training_355"]

    # Step 2: Exploratory Data Analysis (EDA)
    if args.step == "eda":
        # Scan 100 patients for fast execution, or 368 for complete stats
        run_eda(num_patients_for_stats=100)
    elif args.step == "all" and is_quick_test:
        # In quick test mode, just run a quick 5-patient scan for EDA to verify
        run_eda(num_patients_for_stats=5)
    elif args.step == "all":
        run_eda(num_patients_for_stats=368)

    # Step 3: Preprocessing (Offline Transformation & Split)
    if args.step in ["preprocess", "all"]:
        train_patients, val_patients = get_train_val_splits(valid_patients)
        
        if is_quick_test:
            # For quick test, we preprocess only a few patients (e.g., 5 total, 4 train / 1 val)
            quick_train = train_patients[:4]
            quick_val = val_patients[:1]
            print(f"[Quick-Test] Preprocessing 4 train patients and 1 validation patient...")
            preprocess_dataset(quick_train)
            preprocess_dataset(quick_val)
        else:
            # For standard run, check if subset-size is specified
            if args.subset_size is not None:
                n_val = max(1, int(args.subset_size * 0.2))
                n_train = max(1, args.subset_size - n_val)
                train_subset = train_patients[:n_train]
                val_subset = val_patients[:n_val]
                print(f"[Preprocessing] Preprocessing subset of size {args.subset_size} (Train: {n_train}, Val: {n_val})...")
                preprocess_dataset(train_subset)
                preprocess_dataset(val_subset)
            else:
                print("[Preprocessing] Preprocessing FULL dataset. This might take a few minutes...")
                preprocess_dataset(train_patients)
                preprocess_dataset(val_patients)

    # Step 4: Model Training
    if args.step in ["train", "all"]:
        # Retrieve splits
        if not os.path.exists(SPLITS_JSON):
            print(f"Error: splits.json not found. Please run '--step preprocess' first.")
            return
            
        with open(SPLITS_JSON, 'r') as f:
            splits = json.load(f)
            
        if is_quick_test:
            train_list = splits['train'][:4]
            val_list = splits['val'][:1]
            epochs = args.epochs if args.epochs is not None else 2
            base_filters = 32  # Test target filter capacity (falls back if OOM)
        else:
            if args.subset_size is not None:
                n_val = max(1, int(args.subset_size * 0.2))
                n_train = max(1, args.subset_size - n_val)
                train_list = splits['train'][:n_train]
                val_list = splits['val'][:n_val]
            else:
                train_list = splits['train']
                val_list = splits['val']
            epochs = args.epochs if args.epochs is not None else NUM_EPOCHS
            base_filters = 32 # Standard optimized capacity

        # Setup dataloaders
        train_loader = get_dataloader(train_list, batch_size=BATCH_SIZE, shuffle=True, is_train=True)
        val_loader = get_dataloader(val_list, batch_size=BATCH_SIZE, shuffle=False, is_train=False)

        # Initialize 3D U-Net and start training, with fallback for OOM
        try:
            print(f"[Trainer] Initializing 3D U-Net with base_filters={base_filters}...")
            model = UNet3D(in_channels=4, out_channels=3, base_filters=base_filters).to(DEVICE)
            run_training(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=epochs,
                lr=LEARNING_RATE,
                device=DEVICE
            )
        except RuntimeError as e:
            err_msg = str(e).lower()
            is_oom = "out of memory" in err_msg or "oom" in err_msg or "allocation" in err_msg or "allocated" in err_msg
            if is_oom and base_filters == 32:
                print("\n" + "!" * 60)
                print("WARNING: Out of Memory (OOM) detected with base_filters=32.")
                print("Falling back to base_filters=24...")
                print("!" * 60 + "\n")
                
                # Clear device memory cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif hasattr(torch, "mps") and torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                
                base_filters = 24
                print(f"[Trainer] Re-initializing 3D U-Net with base_filters={base_filters}...")
                model = UNet3D(in_channels=4, out_channels=3, base_filters=base_filters).to(DEVICE)
                run_training(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    epochs=epochs,
                    lr=LEARNING_RATE,
                    device=DEVICE
                )
            else:
                raise e

    # Step 5: Model Evaluation
    if args.step in ["eval", "all"]:
        if not os.path.exists(SPLITS_JSON):
            print(f"Error: splits.json not found. Please run '--step preprocess' first.")
            return
            
        with open(SPLITS_JSON, 'r') as f:
            splits = json.load(f)

        if is_quick_test:
            val_list = splits['val'][:1]
            base_filters = 32
        else:
            if args.subset_size is not None:
                n_val = max(1, int(args.subset_size * 0.2))
                val_list = splits['val'][:n_val]
            else:
                val_list = splits['val']
            base_filters = 32

        # Dynamically detect base_filters from checkpoint if it exists to prevent size mismatches
        if os.path.exists(MODEL_SAVE_PATH):
            try:
                state_dict = torch.load(MODEL_SAVE_PATH, map_location=torch.device('cpu'))
                if 'enc1.double_conv.0.weight' in state_dict:
                    checkpoint_filters = state_dict['enc1.double_conv.0.weight'].shape[0]
                    if checkpoint_filters != base_filters:
                        print(f"[Eval] Checkpoint has base_filters={checkpoint_filters}, overriding default ({base_filters}) to avoid size mismatch.")
                        base_filters = checkpoint_filters
            except Exception as e:
                print(f"[Eval] Warning: Failed to inspect checkpoint to detect base_filters: {e}")

        print(f"[Eval] Initializing 3D U-Net for inference with base_filters={base_filters}...")
        model = UNet3D(in_channels=4, out_channels=3, base_filters=base_filters).to(DEVICE)
        
        evaluate_model(model, val_list)

if __name__ == "__main__":
    main()
