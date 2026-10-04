import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

# Setup Paths
BASE_DIR = Path(__file__).resolve().parent

# Check for subfolder nesting automatically
RAW_DIR_BASE = BASE_DIR / "data" / "raw"
if (RAW_DIR_BASE / "Dataset_BUSI_with_GT").exists():
    RAW_DIR = RAW_DIR_BASE / "Dataset_BUSI_with_GT"
else:
    RAW_DIR = RAW_DIR_BASE

PROCESSED_DIR = BASE_DIR / "data" / "processed"

CLASSES = ["benign", "malignant", "normal"]
SPLITS = ["train", "val", "test"]

def create_directory_structure():
    """Create directory structure for train, val, and test splits."""
    for split in SPLITS:
        for cls in CLASSES:
            (PROCESSED_DIR / split / cls).mkdir(parents=True, exist_ok=True)
    print("Created processed directory structure successfully.")

def clean_and_split_data():
    """Separate raw images from masks and split into train/val/test sets."""
    create_directory_structure()
    
    total_cleaned = 0
    print(f"Reading raw data from: {RAW_DIR}")
    
    for cls in CLASSES:
        cls_raw_path = RAW_DIR / cls
        if not cls_raw_path.exists():
            print(f"Warning: Directory '{cls_raw_path}' does not exist. Check raw folder path.")
            continue
            
        # Collect only primary images, ignoring all mask variants (*_mask.png, *_mask_1.png, etc.)
        all_files = os.listdir(cls_raw_path)
        image_files = [
            f for f in all_files 
            if f.endswith('.png') and '_mask' not in f
        ]
        
        print(f"Class '{cls}': Found {len(image_files)} raw ultrasound images (excluding masks).")
        total_cleaned += len(image_files)
        
        if len(image_files) == 0:
            continue

        # Perform Stratified Splits (70% Train, 15% Val, 15% Test)
        train_files, test_val_files = train_test_split(
            image_files, test_size=0.30, random_state=42
        )
        val_files, test_files = train_test_split(
            test_val_files, test_size=0.50, random_state=42
        )
        
        split_dict = {
            "train": train_files,
            "val": val_files,
            "test": test_files
        }
        
        for split_name, file_list in split_dict.items():
            for fname in file_list:
                src = cls_raw_path / fname
                dst = PROCESSED_DIR / split_name / cls / fname
                shutil.copy2(src, dst)
                
    print(f"\nData processing complete! Total valid images split: {total_cleaned}")
    print(f"Dataset location: {PROCESSED_DIR}")

if __name__ == "__main__":
    clean_and_split_data()