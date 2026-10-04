import torch
import torch.nn as nn
from torchvision.models import resnet50, efficientnet_b4, vit_b_16

CLASS_NAMES = ["benign", "malignant", "normal"]
NUM_CLASSES = len(CLASS_NAMES)

def build_resnet(checkpoint_path, device, extract_features=False):
    model = resnet50()
    model.fc = nn.Sequential(
        nn.Dropout(0.4), nn.Linear(model.fc.in_features, 512),
        nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, NUM_CLASSES)
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    if extract_features:
        model.fc = model.fc[:-1]
    model.to(device).eval()
    return model

def build_effnet(checkpoint_path, device, extract_features=False):
    model = efficientnet_b4()
    model.classifier = nn.Sequential(
        nn.Dropout(0.4), nn.Linear(model.classifier[1].in_features, 512),
        nn.BatchNorm1d(512), nn.SiLU(), nn.Dropout(0.3), nn.Linear(512, NUM_CLASSES)
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    if extract_features:
        model.classifier = model.classifier[:-1]
    model.to(device).eval()
    return model

def build_vit(checkpoint_path, device, extract_features=False):
    model = vit_b_16()
    model.heads.head = nn.Sequential(
        nn.Dropout(p=0.3), nn.Linear(model.heads.head.in_features, NUM_CLASSES)
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    if extract_features:
        model.heads.head = model.heads.head[:-1]
    model.to(device).eval()
    return model