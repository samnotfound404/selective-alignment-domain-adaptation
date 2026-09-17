import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from .backbones import build_backbone


class GradReverse(Function):
    """
    Gradient Reversal Layer (GRL).
    Forward: Identity mapping R(x) = x
    Backward: Multiplies gradient by -alpha
    """
    @staticmethod
    def forward(ctx, x, alpha=1.0):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class DynamicFeatureSelector(nn.Module):
    """
    Dynamic Feature Selector from Sultana et al. (MIUA / Springer LNCS 2024):
    "Selective Alignment Transfer for Domain Adaptation in Skin Lesion Analysis"
    
    Takes concatenated batch-mean feature vectors from source and target [f_bar_s, f_bar_t] in R^(2C),
    passes through a multi-layer perceptron, and outputs channel-wise importance weights w in (0, 1)^C:
        w = sigma( W3 * ReLU( W2 * ReLU( W1 * [f_bar_s, f_bar_t] + b1 ) + b2 ) + b3 )
    """
    def __init__(self, feature_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.feature_dim = feature_dim
        self.network = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim),
            nn.Sigmoid()
        )

    def forward(self, source_mean: torch.Tensor, target_mean: torch.Tensor) -> torch.Tensor:
        if source_mean.dim() == 1:
            source_mean = source_mean.unsqueeze(0)
        if target_mean.dim() == 1:
            target_mean = target_mean.unsqueeze(0)
            
        combined = torch.cat([source_mean, target_mean], dim=-1)
        weights = self.network(combined)
        return weights.squeeze(0)


class SATDANetwork(nn.Module):
    """
    Complete implementation of Sultana et al. SAT-DA baseline:
    - Backbone encoder G_f
    - Dynamic feature selector w in (0, 1)^C
    - Classification head G_y
    - Losses:
        L_cls: Standard cross-entropy
        L_align: Mean squared error between channel-weighted source and target representations
        L_div: Weight diversity loss maximizing variance/spread of w across channels
    """
    def __init__(
        self,
        num_classes: int = 5,
        backbone: str = 'efficientnet_b0',
        pretrained: bool = True,
        dropout: float = 0.5
    ):
        super().__init__()
        self.num_classes = num_classes
        self.encoder, self.feat_dim = build_backbone(backbone, pretrained=pretrained)
        self.selector = DynamicFeatureSelector(self.feat_dim)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)
        return feat.flatten(1)

    def forward(self, s_imgs: torch.Tensor, t_imgs: torch.Tensor = None):
        s_feat = self.extract_features(s_imgs)
        s_logits = self.classifier(s_feat)

        if t_imgs is None:
            return s_logits

        t_feat = self.extract_features(t_imgs)
        t_logits = self.classifier(t_feat)

        align_loss = torch.tensor(0.0, device=s_imgs.device)
        diverse_loss = torch.tensor(0.0, device=s_imgs.device)

        if self.training:
            s_mean = s_feat.mean(dim=0)
            t_mean = t_feat.mean(dim=0)
            weights = self.selector(s_mean, t_mean)

            # L_align: Selective moment alignment across weighted channels
            w_s_feat = s_feat * weights.unsqueeze(0)
            w_t_feat = t_feat * weights.unsqueeze(0)
            align_loss = F.mse_loss(w_s_feat.mean(dim=0), w_t_feat.mean(dim=0))

            # L_div: Negative standard deviation penalty to encourage diverse channel usage
            w_std = torch.std(weights)
            diverse_loss = -w_std

        return s_logits, t_logits, align_loss, diverse_loss
