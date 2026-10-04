import os
import sys
from pathlib import Path

# Explicitly add current project root to Python search path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import joblib

# Import from src
from src.transforms import tf_cnn, tf_vit
from src.models import build_resnet, build_effnet, build_vit, CLASS_NAMES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def check_file_sizes():
    print("==========================================")
    print("1. ARTIFACT FILE SIZE AUDIT")
    print("==========================================")
    artifacts_dir = PROJECT_ROOT / "artifacts"
    for f in sorted(artifacts_dir.glob("*")):
        if f.is_file():
            size_mb = os.path.getsize(f) / (1024 * 1024)
            print(f"  - {f.name:<35} | Size: {size_mb:.2f} MB")

def test_checkpoint_loading():
    print("\n==========================================")
    print("2. CHECKPOINT INTEGRITY TEST")
    print("==========================================")
    
    # EfficientNet candidates
    eff_candidates = [
        "best_efficientnet_advanced.pth",
        "best_efficientnet_b4.pth",
        "best_efficientnet.pth"
    ]
    for filename in eff_candidates:
        path = PROJECT_ROOT / "artifacts" / filename
        if path.exists():
            try:
                m = build_effnet(str(path), device)
                print(f"  [SUCCESS] {filename:<35} loaded cleanly into EfficientNet-B4.")
            except Exception as e:
                print(f"  [FAILED]  {filename:<35} Error: {e}")

    # ResNet candidates
    resnet_candidates = [
        "best_resnet_50.pth",
        "best_resnet50.pth"
    ]
    for filename in resnet_candidates:
        path = PROJECT_ROOT / "artifacts" / filename
        if path.exists():
            try:
                m = build_resnet(str(path), device)
                print(f"  [SUCCESS] {filename:<35} loaded cleanly into ResNet-50.")
            except Exception as e:
                print(f"  [FAILED]  {filename:<35} Error: {e}")

if __name__ == "__main__":
    check_file_sizes()
    test_checkpoint_loading()