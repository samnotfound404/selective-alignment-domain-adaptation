import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbones import build_backbone
from .satda_baseline import DynamicFeatureSelector


class CCSATDANet(nn.Module):
    """
    Class-Conditioned Selective Alignment Transfer Network (CC-SAT-DA).
    
    Key Innovations:
    1. Active Gradient Prototype Alignment (L_proto):
       Class centroids computed on active batches maintain computational graphs,
       propagating non-zero gradients directly into the backbone encoder.
    2. Momentum-Updated EMA Global Prototype Registers:
       Maintains long-term running representations mu_c^(s) and mu_c^(t) to stabilize
       alignments when rare classes (e.g. Melanoma, 2.3% prevalence) appear sporadically.
    3. Inter-Class Dispersion Regularization (L_disp):
       Enforces cosine angular margin separation between distinct diagnostic classes,
       preventing prototype collapse onto dominant majorities (BCC / AKIEC).
    4. Optional Dynamic Feature Selector:
       Learns channel weights w in (0, 1)^C to selectively downweight sensor-specific noise channels.
    """
    def __init__(
        self,
        num_classes: int = 5,
        backbone: str = 'efficientnet_b2',
        pretrained: bool = True,
        dropout: float = 0.5,
        ema_momentum: float = 0.9,
        use_channel_selector: bool = True
    ):
        super().__init__()
        self.num_classes = num_classes
        self.ema_momentum = ema_momentum
        self.use_channel_selector = use_channel_selector

        # 1. Feature Extractor
        self.encoder, self.feat_dim = build_backbone(backbone, pretrained=pretrained)

        # 2. Dynamic Feature Selector
        if self.use_channel_selector:
            self.selector = DynamicFeatureSelector(self.feat_dim)

        # 3. Diagnostic Classifier Head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

        # 4. Global EMA Memory Banks (Persistent buffers)
        self.register_buffer('src_prototypes', torch.zeros(num_classes, self.feat_dim))
        self.register_buffer('tgt_prototypes', torch.zeros(num_classes, self.feat_dim))
        self.register_buffer('src_counts', torch.zeros(num_classes))
        self.register_buffer('tgt_counts', torch.zeros(num_classes))

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)
        return feat.flatten(1)

    @torch.no_grad()
    def update_prototypes(self, feats: torch.Tensor, labels: torch.Tensor, is_target: bool = False):
        """Updates background EMA running averages with detached feature representations."""
        feats_det = feats.detach()
        for c in range(self.num_classes):
            mask = (labels == c)
            if mask.sum() > 0:
                c_mean = feats_det[mask].mean(dim=0)
                if is_target:
                    if self.tgt_counts[c] == 0:
                        self.tgt_prototypes[c] = c_mean
                    else:
                        self.tgt_prototypes[c] = (
                            self.ema_momentum * self.tgt_prototypes[c] +
                            (1.0 - self.ema_momentum) * c_mean
                        )
                    self.tgt_counts[c] += mask.sum()
                else:
                    if self.src_counts[c] == 0:
                        self.src_prototypes[c] = c_mean
                    else:
                        self.src_prototypes[c] = (
                            self.ema_momentum * self.src_prototypes[c] +
                            (1.0 - self.ema_momentum) * c_mean
                        )
                    self.src_counts[c] += mask.sum()

    def forward(
        self,
        s_imgs: torch.Tensor,
        t_imgs: torch.Tensor = None,
        s_lbls: torch.Tensor = None,
        t_lbls: torch.Tensor = None
    ):
        """
        Forward Pass with Active Gradient Propagation:
        Evaluates logits, L_proto, and L_disp directly on active feature tensors
        so that encoder parameters receive genuine autograd gradients.
        """
        s_feat = self.extract_features(s_imgs)
        s_logits = self.classifier(s_feat)

        if t_imgs is None:
            return s_logits

        t_feat = self.extract_features(t_imgs)
        t_logits = self.classifier(t_feat)

        proto_loss = torch.tensor(0.0, device=s_imgs.device)
        div_loss = torch.tensor(0.0, device=s_imgs.device)

        if self.training and s_lbls is not None and t_lbls is not None:
            proto_terms = []
            active_tgt_centroids = []
            active_tgt_classes = []

            # 1. Class-Conditional Prototype Alignment (L_proto)
            for c in range(self.num_classes):
                s_mask = (s_lbls == c)
                t_mask = (t_lbls == c)
                has_s = s_mask.any()
                has_t = t_mask.any()

                if has_s and has_t:
                    # Both present: direct batch centroid cosine alignment (gradients to both domains)
                    z_s_c = F.normalize(s_feat[s_mask].mean(dim=0, keepdim=True), dim=1)
                    z_t_c = F.normalize(t_feat[t_mask].mean(dim=0, keepdim=True), dim=1)
                    proto_terms.append(1.0 - (z_s_c * z_t_c).sum())
                    active_tgt_centroids.append(z_t_c)
                    active_tgt_classes.append(c)
                elif has_t and self.src_counts[c] > 0:
                    # Target present: align active target feature with accumulated source prototype
                    z_t_c = F.normalize(t_feat[t_mask].mean(dim=0, keepdim=True), dim=1)
                    mu_s_c = F.normalize(self.src_prototypes[c:c+1].detach(), dim=1)
                    proto_terms.append(1.0 - (z_t_c * mu_s_c).sum())
                    active_tgt_centroids.append(z_t_c)
                    active_tgt_classes.append(c)
                elif has_s and self.tgt_counts[c] > 0:
                    # Source present: align active source feature with accumulated target prototype
                    z_s_c = F.normalize(s_feat[s_mask].mean(dim=0, keepdim=True), dim=1)
                    mu_t_c = F.normalize(self.tgt_prototypes[c:c+1].detach(), dim=1)
                    proto_terms.append(1.0 - (z_s_c * mu_t_c).sum())

            if len(proto_terms) > 0:
                proto_loss = torch.stack(proto_terms).mean()

            # 2. Inter-Class Dispersion Loss (L_disp)
            # Enforces angular margin separation across distinct diagnostic categories
            if len(active_tgt_centroids) >= 2:
                tgt_mat = torch.cat(active_tgt_centroids, dim=0)
                sim_matrix = torch.mm(tgt_mat, tgt_mat.t())
                k = len(active_tgt_centroids)
                eye = torch.eye(k, device=s_imgs.device)
                div_loss = (sim_matrix * (1.0 - eye)).clamp(min=0.0).sum() / (k * (k - 1) + 1e-6)
            elif len(active_tgt_centroids) == 1 and (self.src_counts > 0).any():
                c_act = active_tgt_classes[0]
                z_act = active_tgt_centroids[0]
                other_protos = [
                    F.normalize(self.src_prototypes[k:k+1].detach(), dim=1)
                    for k in range(self.num_classes)
                    if k != c_act and self.src_counts[k] > 0
                ]
                if len(other_protos) > 0:
                    other_mat = torch.cat(other_protos, dim=0)
                    sims = torch.mm(z_act, other_mat.t())
                    div_loss = sims.clamp(min=0.0).mean()

            # 3. Update memory bank with detached features after current gradient step
            self.update_prototypes(s_feat, s_lbls, is_target=False)
            self.update_prototypes(t_feat, t_lbls, is_target=True)

        return s_logits, t_logits, proto_loss, div_loss
