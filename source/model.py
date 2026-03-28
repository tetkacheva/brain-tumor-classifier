import torch.nn as nn
from torchvision import models


def build_model(num_classes=2):
    model = models.efficientnet_b0(weights="IMAGENET1K_V1")

    for p in model.parameters():
        p.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier == nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes)
    )

    for p in model.classifier.parameters():
        p.requires_grad = True

    return model
