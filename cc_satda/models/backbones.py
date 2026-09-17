import torch
import torch.nn as nn
from torchvision import models


def build_backbone(backbone_name: str, pretrained: bool = True):
    """
    Constructs backbone feature extractor and returns (encoder_module, feature_dim).
    
    Supported backbones:
    - 'resnet18': ResNet-18 (512-dim)
    - 'resnet50': ResNet-50 (2048-dim)
    - 'efficientnet_b0': EfficientNet-B0 (1280-dim) [Used for baselines in benchmark]
    - 'efficientnet_b2': EfficientNet-B2 (1408-dim) [Used for CC-SAT-DA in benchmark]
    """
    name = backbone_name.lower().strip()

    if name == 'resnet18':
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base = models.resnet18(weights=weights)
        encoder = nn.Sequential(*list(base.children())[:-1])
        feat_dim = base.fc.in_features
        return encoder, feat_dim

    elif name == 'resnet50':
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        base = models.resnet50(weights=weights)
        encoder = nn.Sequential(*list(base.children())[:-1])
        feat_dim = base.fc.in_features
        return encoder, feat_dim

    elif name in ('efficientnet_b0', 'efficientnet'):
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        base = models.efficientnet_b0(weights=weights)
        encoder = nn.Sequential(
            base.features,
            nn.AdaptiveAvgPool2d(1)
        )
        feat_dim = base.classifier[1].in_features  # 1280
        return encoder, feat_dim

    elif name == 'efficientnet_b2':
        weights = models.EfficientNet_B2_Weights.DEFAULT if pretrained else None
        base = models.efficientnet_b2(weights=weights)
        encoder = nn.Sequential(
            base.features,
            nn.AdaptiveAvgPool2d(1)
        )
        feat_dim = base.classifier[1].in_features  # 1408
        return encoder, feat_dim

    else:
        raise ValueError(
            f"Unsupported backbone '{backbone_name}'. "
            f"Choose from: 'resnet18', 'resnet50', 'efficientnet_b0', 'efficientnet_b2'."
        )
