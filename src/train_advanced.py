import os
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
from torchvision import transforms, datasets
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from torch.utils.data import DataLoader

# Advanced Medical Augmentation with Higher Resolution (384x384)
def get_advanced_transforms(img_size=384):
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, eval_transform

def build_advanced_efficientnet(num_classes=3):
    weights = EfficientNet_B4_Weights.DEFAULT
    model = efficientnet_b4(weights=weights)
    
    # Custom classifier with regularized Dropout
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, 256),
        nn.BatchNorm1d(256),
        nn.SiLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes)
    )
    return model

def run_advanced_training(epochs=35, batch_size=8, img_size=384):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f"   Starting High-Accuracy Fine-Tuning (EfficientNet-B4) ")
    print(f"   Resolution: {img_size}x{img_size} | Device: {device} ")
    print(f"=======================================================\n")

    BASE_DIR = Path(__file__).resolve().parent.parent
    processed_dir = BASE_DIR / "data" / "processed"
    artifacts_dir = BASE_DIR / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    train_tf, eval_tf = get_advanced_transforms(img_size=img_size)

    train_ds = datasets.ImageFolder(root=processed_dir / 'train', transform=train_tf)
    val_ds = datasets.ImageFolder(root=processed_dir / 'val', transform=eval_tf)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_advanced_efficientnet(num_classes=len(train_ds.classes)).to(device)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05) # Label smoothing prevents over-confidence
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_acc = 0.0
    checkpoint_path = artifacts_dir / "best_efficientnet_advanced.pth"

    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]", leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            train_total += labels.size(0)
            train_correct += preds.eq(labels).sum().item()

        # Validation Phase
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]", leave=False):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, preds = outputs.max(1)
                val_total += labels.size(0)
                val_correct += preds.eq(labels).sum().item()

        scheduler.step()

        t_acc = (train_correct / train_total) * 100
        v_acc = (val_correct / val_total) * 100
        t_loss = train_loss / train_total
        v_loss = val_loss / val_total

        print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Acc: {t_acc:.2f}% (Loss: {t_loss:.4f}) | Val Acc: {v_acc:.2f}% (Loss: {v_loss:.4f})")

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  --> Saved new best model checkpoint! Val Acc: {v_acc:.2f}%")

    print(f"\nAdvanced Training Complete! Best Validation Accuracy: {best_val_acc:.2f}%")

if __name__ == "__main__":
    run_advanced_training(epochs=30, batch_size=8, img_size=384)