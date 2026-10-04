import os
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
from dataset import get_dataloaders
from models import build_model

def compute_class_weights(train_loader, num_classes=3):
    """Computes inverse class frequencies to handle dataset imbalance."""
    class_counts = torch.zeros(num_classes)
    for _, labels in train_loader:
        for label in labels:
            class_counts[label] += 1
    total_samples = class_counts.sum()
    class_weights = total_samples / (num_classes * class_counts)
    return class_weights

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def run_training(model_name="resnet50", epochs=20, lr=1e-4, batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Training {model_name.upper()} on device: {device} ---")

    BASE_DIR = Path(__file__).resolve().parent.parent
    processed_dir = BASE_DIR / "data" / "processed"
    artifacts_dir = BASE_DIR / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    train_loader, val_loader, test_loader, classes = get_dataloaders(
        processed_dir=processed_dir, batch_size=batch_size
    )

    weights = compute_class_weights(train_loader, num_classes=len(classes)).to(device)
    print(f"Computed Class Weights (Benign, Malignant, Normal): {weights.tolist()}")

    model = build_model(model_name=model_name, num_classes=len(classes), pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    best_val_loss = float("inf")
    checkpoint_path = artifacts_dir / f"best_{model_name}.pth"

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] | "
              f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc*100:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f" --> Saved new best model checkpoint to {checkpoint_path.name}")

    print(f"\nTraining Complete for {model_name.upper()}. Best Val Loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet50", choices=["resnet50", "efficientnet"])
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    run_training(model_name=args.model, epochs=args.epochs)