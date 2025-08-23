import torch.nn as nn
from torchvision import models

def build_efficientnet_b0(num_classes=7, pretrained=True):
    w = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    m = models.efficientnet_b0(weights=w)
    in_f = m.classifier[1].in_features
    m.classifier[1] = nn.Linear(in_f, num_classes)
    return m

# 전역 빌더(향후 b1/b2 추가 대비)
def build_model(name, num_classes, pretrained=True):
    if name in (None, "", "efficientnet_b0"):
        return build_efficientnet_b0(num_classes, pretrained)
    raise ValueError(f"Unknown EfficientNet variant: {name}")