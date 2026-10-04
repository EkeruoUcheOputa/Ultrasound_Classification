import sys
import torch
import joblib
import warnings
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Suppress harmless scikit-learn version warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Ensure project root is in search path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.transforms import tf_cnn, tf_vit
from src.models import build_resnet, build_effnet, build_vit, CLASS_NAMES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_full_ensemble():
    print(f"Using device: {device}\n")
    print("1. Loading production base models from artifacts/...")
    res = build_resnet(PROJECT_ROOT / "artifacts" / "best_resnet_50.pth", device, extract_features=False)
    eff = build_effnet(PROJECT_ROOT / "artifacts" / "best_efficientnet_b4.pth", device, extract_features=False)
    vit = build_vit(PROJECT_ROOT / "artifacts" / "best_vit_b16.pth", device, extract_features=False)

    print("2. Loading stacked meta-learner (C=2.0)...")
    meta_learner = joblib.load(PROJECT_ROOT / "artifacts" / "stacked_meta_learner_c2.joblib")

    # Corrected local path
    val_dir = PROJECT_ROOT / "data" / "processed" / "val"
    image_paths = list(val_dir.rglob("*.png")) + list(val_dir.rglob("*.jpg"))

    if len(image_paths) == 0:
        print(f"[ERROR] No images found in: {val_dir}")
        print("Please verify your data directory path.")
        return

    print(f"3. Running evaluation on {len(image_paths)} validation images...\n")

    y_true = []
    y_pred = []

    for img_path in image_paths:
        # Resolve class label from folder name (benign, malignant, or normal)
        label_str = img_path.parent.name.lower()
        if label_str not in CLASS_NAMES:
            continue

        y_true.append(CLASS_NAMES.index(label_str))

        # Preprocess image
        img = Image.open(img_path).convert("RGB")
        t_cnn = tf_cnn(img).unsqueeze(0).to(device)
        t_vit = tf_vit(img).unsqueeze(0).to(device)

        # Get base model probability predictions
        with torch.no_grad():
            p_res = torch.softmax(res(t_cnn), dim=1).cpu().numpy()
            p_eff = torch.softmax(eff(t_cnn), dim=1).cpu().numpy()
            p_vit = torch.softmax(vit(t_vit), dim=1).cpu().numpy()
            
            # Stack into 9-D probability vector
            stacked_probs = np.hstack([p_res, p_eff, p_vit])

        # Predict using meta-learner
        pred_idx = meta_learner.predict(stacked_probs)[0]
        y_pred.append(pred_idx)

    # Compute and display metrics
    overall_acc = accuracy_score(y_true, y_pred) * 100

    print("==================================================")
    print("           FULL ENSEMBLE ACCURACY RESULT          ")
    print("==================================================")
    print(f" Total Evaluated Images : {len(y_true)}")
    print(f" Overall Accuracy       : {overall_acc:.2f}%")
    print("==================================================\n")

    print("Detailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    evaluate_full_ensemble()