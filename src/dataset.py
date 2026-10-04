import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

# Standard ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_data_transforms(img_size=224):
    """
    Returns image transformation pipelines for ultrasound classification.
    Applies spatial data augmentations to training set to combat overfitting.
    """
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomResizedCrop(size=img_size, scale=(0.85, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    return train_transform, eval_transform

def get_dataloaders(processed_dir, batch_size=16, img_size=224, num_workers=0):
    """
    Creates and returns PyTorch DataLoaders for train, val, and test splits.
    """
    train_transform, eval_transform = get_data_transforms(img_size=img_size)

    train_dir = os.path.join(processed_dir, 'train')
    val_dir = os.path.join(processed_dir, 'val')
    test_dir = os.path.join(processed_dir, 'test')

    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=eval_transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, train_dataset.classes