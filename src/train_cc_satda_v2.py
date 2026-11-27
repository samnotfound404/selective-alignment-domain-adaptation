#!/usr/bin/env python3
# ==============================================================================
# CC-SATDA v2: CLASS-CONDITIONED PROTOTYPE ALIGNED SAT-DA
# Includes:
#   1. Runtime dynamic prior computation (pi_src, pi_tgt from actual data)
#   2. Multi-seed / multi-run training and evaluation
#   3. Explicit empty-class guards in prototype computation
#   4. Dynamic validation tau grid-search per N
#   5. Asymmetric loss weight configurability (w_source, w_target)
#   6. Epoch-level per-class prototype sample tracking
#   7. Exponential Moving Average (EMA) momentum prototype buffers
#   8. Reliability-weighted prototype alignment loss
# ==============================================================================

import argparse
import copy
import json
import os
import random
import sys
import time
from collections import Counter

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Add parent directory to path so imports from run_ultra_low_resource_study work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score
)
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from run_ultra_low_resource_study import (
    CLASSES, CLASS_TO_IDX, IMAGENET_MEAN, IMAGENET_STD,
    InMemorySkinDataset, SkinDomainDataset,
    build_transforms_b2_enhanced,
    EfficientNetFeatureExtractor, SecondOrderFeatureSelector, LabelClassifier,
    compute_cost_sensitive_weights, set_seed, infinite_loader
)

# ------------------------------------------------------------------------------
# 1. DYNAMIC CLASS PRIOR COMPUTATION
# ------------------------------------------------------------------------------
def compute_class_priors(dataset_items, num_classes: int = 5, smoothing: float = 1e-4) -> torch.Tensor:
    """
    Computes empirical class frequencies directly from dataset items at runtime.
    Guarantees no stale hardcoded priors on resplits or subset changes.
    """
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for _, label in dataset_items:
        counts[label] += 1.0
    total = counts.sum()
    if total == 0:
        return torch.ones(num_classes, dtype=torch.float32) / num_classes
    priors = (counts + smoothing) / (total + num_classes * smoothing)
    return priors


# ------------------------------------------------------------------------------
# DATASET BUILDER WITH SEED SUPPORT
# ------------------------------------------------------------------------------
PAD_TO_SHARED_CLASS = {
    "BCC": "BCC",
    "MEL": "MEL",
    "NEV": "NV",
    "ACK": "AKIEC",
    "SEK": "BKL",
}

def build_datasets_v2(data_dir: str, seed: int = 42):
    """
    Builds datasets while preserving fixed held-out test set and using seed for train subsets.
    """
    isic_train_csv = os.path.join(
        data_dir, "ISIC2018_Task3_Training_GroundTruth", "ISIC2018_Task3_Training_GroundTruth.csv"
    )
    isic_train_img_dir = os.path.join(data_dir, "ISIC2018_Task3_Training_Input")
    isic_test_csv = os.path.join(
        data_dir, "ISIC2018_Task3_Test_GroundTruth", "ISIC2018_Task3_Test_GroundTruth.csv"
    )
    isic_test_img_dir = os.path.join(data_dir, "ISIC2018_Task3_Test_Input")

    pad_metadata_csv = os.path.join(data_dir, "metadata.csv")
    pad_img_dirs = [
        os.path.join(data_dir, "images", "imgs_part_1"),
        os.path.join(data_dir, "images", "imgs_part_2"),
        os.path.join(data_dir, "images", "imgs_part_3"),
    ]

    df_isic_train = pd.read_csv(isic_train_csv)
    df_isic_test = pd.read_csv(isic_test_csv)
    train_files = set(os.listdir(isic_train_img_dir))
    test_files = set(os.listdir(isic_test_img_dir))

    src_train_full = []
    for _, row in df_isic_train.iterrows():
        img_id = str(row["image"])
        fname = f"{img_id}.jpg"
        if fname not in train_files:
            continue
        for c in CLASSES:
            if c in row and row[c] == 1.0:
                src_train_full.append((os.path.join(isic_train_img_dir, fname), CLASS_TO_IDX[c]))
                break

    src_test = []
    for _, row in df_isic_test.iterrows():
        img_id = str(row["image"])
        fname = f"{img_id}.jpg"
        if fname not in test_files:
            continue
        for c in CLASSES:
            if c in row and row[c] == 1.0:
                src_test.append((os.path.join(isic_test_img_dir, fname), CLASS_TO_IDX[c]))
                break

    labels_src_train = [x[1] for x in src_train_full]
    src_train, src_val = train_test_split(
        src_train_full, test_size=0.15, random_state=42, stratify=labels_src_train
    )

    pad_img_cache = {}
    for d in pad_img_dirs:
        if os.path.exists(d):
            for fname in os.listdir(d):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    pad_img_cache[fname] = os.path.join(d, fname)

    df_pad = pd.read_csv(pad_metadata_csv)
    target_items = []
    for _, row in df_pad.iterrows():
        diag = str(row.get("diagnostic", "")).strip()
        img_id = str(row.get("img_id", "")).strip()
        if diag in PAD_TO_SHARED_CLASS and img_id in pad_img_cache:
            shared_c = PAD_TO_SHARED_CLASS[diag]
            target_items.append((pad_img_cache[img_id], CLASS_TO_IDX[shared_c]))

    labels_tgt = [x[1] for x in target_items]
    # Fixed 70/15/15 split for target train, val, held-out test
    tgt_train_full, tgt_temp = train_test_split(
        target_items, test_size=0.30, random_state=42, stratify=labels_tgt
    )
    labels_temp = [x[1] for x in tgt_temp]
    tgt_val, tgt_test = train_test_split(
        tgt_temp, test_size=0.50, random_state=42, stratify=labels_temp
    )

    # Nested Subsets using specific seed: 200 -> 100 -> 50
    labels_full = [x[1] for x in tgt_train_full]
    tgt_train_200, _ = train_test_split(
        tgt_train_full, train_size=200, random_state=seed, stratify=labels_full
    )

    labels_200 = [x[1] for x in tgt_train_200]
    tgt_train_100, _ = train_test_split(
        tgt_train_200, train_size=100, random_state=seed, stratify=labels_200
    )

    labels_100 = [x[1] for x in tgt_train_100]
    tgt_train_50, _ = train_test_split(
        tgt_train_100, train_size=50, random_state=seed, stratify=labels_100
    )

    return {
        "src_train": src_train,
        "src_val": src_val,
        "src_test": src_test,
        "tgt_train_full": tgt_train_full,
        "tgt_train_200": tgt_train_200,
        "tgt_train_100": tgt_train_100,
        "tgt_train_50": tgt_train_50,
        "tgt_val": tgt_val,
        "tgt_test": tgt_test,
    }


# ------------------------------------------------------------------------------
# 2. CC-SATDA NET V2 WITH EMA PROTOTYPES & RELIABILITY WEIGHTING
# ------------------------------------------------------------------------------
class ClassConditionedSATDANetV2(nn.Module):
    def __init__(
        self,
        num_classes: int = 5,
        dropout: float = 0.3,
        use_ema_prototypes: bool = True,
        ema_momentum: float = 0.9,
        use_reliability_weighting: bool = True,
        reliability_threshold: float = 8.0,
    ):
        super().__init__()
        self.feature_extractor = EfficientNetFeatureExtractor(variant="b2", pretrained=True)
        self.out_dim = self.feature_extractor.out_dim
        self.feature_selector = SecondOrderFeatureSelector(self.out_dim)
        self.classifier = LabelClassifier(self.out_dim, num_classes, dropout)
        self.num_classes = num_classes
        self.last_feature_weights = None

        self.use_ema_prototypes = use_ema_prototypes
        self.ema_momentum = ema_momentum
        self.use_reliability_weighting = use_reliability_weighting
        self.reliability_threshold = reliability_threshold

        # Persistent EMA prototype buffers
        self.register_buffer("source_proto_ema", torch.zeros(num_classes, self.out_dim))
        self.register_buffer("target_proto_ema", torch.zeros(num_classes, self.out_dim))
        self.register_buffer("source_proto_init", torch.zeros(num_classes, dtype=torch.bool))
        self.register_buffer("target_proto_init", torch.zeros(num_classes, dtype=torch.bool))
        self.register_buffer("source_seen_counts", torch.zeros(num_classes, dtype=torch.long))
        self.register_buffer("target_seen_counts", torch.zeros(num_classes, dtype=torch.long))

        # Runtime batch tracking for instrumentation
        self.batch_counts = {
            "source": [0] * num_classes,
            "target": [0] * num_classes,
            "co_occurrences": [0] * num_classes
        }

    def reset_epoch_batch_counts(self):
        self.batch_counts = {
            "source": [0] * self.num_classes,
            "target": [0] * self.num_classes,
            "co_occurrences": [0] * self.num_classes
        }

    def forward(self, source_images, target_images=None, source_labels=None, target_labels=None):
        source_features = self.feature_extractor(source_images)
        if self.training and target_images is not None:
            target_features = self.feature_extractor(target_images)
            source_mean = source_features.mean(dim=0, keepdim=True)
            source_std = source_features.std(dim=0, keepdim=True) + 1e-6
            target_mean = target_features.mean(dim=0, keepdim=True)
            target_std = target_features.std(dim=0, keepdim=True) + 1e-6

            feature_weights = self.feature_selector(source_mean, source_std, target_mean, target_std)
            weighted_source = source_features * feature_weights.unsqueeze(0)
            weighted_target = target_features * feature_weights.unsqueeze(0)

            source_logits = self.classifier(weighted_source)
            target_logits = self.classifier(weighted_target)

            self.last_feature_weights = feature_weights.detach()

            # Top-k most discriminative domain-invariant channels
            k = int(0.5 * self.out_dim)
            _, topk_idx = torch.topk(feature_weights, k)
            s_sub = weighted_source[:, topk_idx]
            t_sub = weighted_target[:, topk_idx]

            proto_loss = torch.tensor(0.0, device=source_images.device)
            total_weight = 0.0

            if source_labels is not None and target_labels is not None:
                for c in range(self.num_classes):
                    s_mask = (source_labels == c)
                    t_mask = (target_labels == c)
                    s_count = int(s_mask.sum().item())
                    t_count = int(t_mask.sum().item())

                    # Track instrumentation stats
                    if s_count > 0:
                        self.batch_counts["source"][c] += s_count
                    if t_count > 0:
                        self.batch_counts["target"][c] += t_count
                    if s_count > 0 and t_count > 0:
                        self.batch_counts["co_occurrences"][c] += 1

                    # Update EMA prototypes when samples present
                    if s_count > 0:
                        s_batch_mean = weighted_source[s_mask].mean(dim=0)
                        if not self.source_proto_init[c]:
                            self.source_proto_ema[c] = s_batch_mean.detach()
                            self.source_proto_init[c] = True
                        else:
                            self.source_proto_ema[c] = (
                                self.ema_momentum * self.source_proto_ema[c]
                                + (1.0 - self.ema_momentum) * s_batch_mean.detach()
                            )
                        self.source_seen_counts[c] += s_count

                    if t_count > 0:
                        t_batch_mean = weighted_target[t_mask].mean(dim=0)
                        if not self.target_proto_init[c]:
                            self.target_proto_ema[c] = t_batch_mean.detach()
                            self.target_proto_init[c] = True
                        else:
                            self.target_proto_ema[c] = (
                                self.ema_momentum * self.target_proto_ema[c]
                                + (1.0 - self.ema_momentum) * t_batch_mean.detach()
                            )
                        self.target_seen_counts[c] += t_count

                    # Compute prototype alignment for class c
                    term_c = None
                    if s_count > 0 and t_count > 0:
                        # Both present in batch: align current representations directly
                        term_c = F.mse_loss(s_sub[s_mask].mean(dim=0), t_sub[t_mask].mean(dim=0))
                    elif self.use_ema_prototypes:
                        # Structural EMA fallback: prevents batch-starvation for rare classes
                        if s_count > 0 and self.target_proto_init[c]:
                            # Pull source batch mean towards accumulated target prototype
                            term_c = F.mse_loss(s_sub[s_mask].mean(dim=0), self.target_proto_ema[c, topk_idx].detach())
                        elif t_count > 0 and self.source_proto_init[c]:
                            # Pull target batch mean towards accumulated source prototype
                            term_c = F.mse_loss(t_sub[t_mask].mean(dim=0), self.source_proto_ema[c, topk_idx].detach())

                    if term_c is not None:
                        # Reliability weighting factor
                        if self.use_reliability_weighting:
                            min_seen = min(self.source_seen_counts[c].item(), self.target_seen_counts[c].item())
                            w_c = min(1.0, max(0.2, float(min_seen) / self.reliability_threshold))
                        else:
                            w_c = 1.0

                        proto_loss = proto_loss + w_c * term_c
                        total_weight += w_c

            if total_weight > 0.0:
                proto_loss = proto_loss / total_weight
            else:
                # Safe fallback to global moment alignment if zero matched/initialized classes
                proto_loss = F.mse_loss(s_sub.mean(dim=0), t_sub.mean(dim=0))

            # Feature diversity regularization
            norm_w = feature_weights[topk_idx] / (torch.norm(feature_weights[topk_idx]) + 1e-6)
            sim_matrix = torch.matmul(norm_w.unsqueeze(1), norm_w.unsqueeze(0))
            div_loss = (torch.sum(sim_matrix) - torch.trace(sim_matrix)) / (k * (k - 1) + 1e-6)

            return source_logits, target_logits, proto_loss, div_loss
        else:
            if self.last_feature_weights is None:
                self.last_feature_weights = torch.ones(source_features.size(1), device=source_features.device)
            weighted = source_features * self.last_feature_weights.unsqueeze(0)
            return self.classifier(weighted)


# ------------------------------------------------------------------------------
# 3. EVALUATION & TAU GRID SEARCH
# ------------------------------------------------------------------------------
def evaluate_metrics_from_logits(logits: torch.Tensor, labels: np.ndarray,
                                 pi_src: torch.Tensor, pi_tgt: torch.Tensor,
                                 tau: float = 0.35) -> dict:
    if tau > 0 and pi_src is not None and pi_tgt is not None:
        log_prior_ratio = torch.log((pi_tgt + 1e-6) / (pi_src + 1e-6))
        adj_logits = logits + tau * log_prior_ratio
    else:
        adj_logits = logits

    probs = F.softmax(adj_logits, dim=1).numpy()
    preds = np.argmax(probs, axis=1)

    acc = float(accuracy_score(labels, preds))
    f1_macro = float(f1_score(labels, preds, average="macro", zero_division=0))
    prec_macro = float(precision_score(labels, preds, average="macro", zero_division=0))
    rec_macro = float(recall_score(labels, preds, average="macro", zero_division=0))

    try:
        auroc_macro = float(roc_auc_score(labels, probs, multi_class="ovr", average="macro"))
    except Exception:
        auroc_macro = float("nan")

    recalls = recall_score(labels, preds, average=None, zero_division=0)
    f1s = f1_score(labels, preds, average=None, zero_division=0)

    mel_idx = CLASS_TO_IDX["MEL"]
    bcc_idx = CLASS_TO_IDX["BCC"]

    mel_rec = float(recalls[mel_idx]) if len(recalls) > mel_idx else 0.0
    mel_f1 = float(f1s[mel_idx]) if len(f1s) > mel_idx else 0.0
    bcc_rec = float(recalls[bcc_idx]) if len(recalls) > bcc_idx else 0.0
    bcc_f1 = float(f1s[bcc_idx]) if len(f1s) > bcc_idx else 0.0

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "auroc_macro_ovr": auroc_macro,
        "melanoma_recall": mel_rec,
        "melanoma_f1": mel_f1,
        "bcc_recall": bcc_rec,
        "bcc_f1": bcc_f1,
        "per_class_recall": {CLASSES[i]: float(recalls[i]) for i in range(len(recalls))},
        "per_class_f1": {CLASSES[i]: float(f1s[i]) for i in range(len(f1s))},
    }


@torch.no_grad()
def collect_raw_logits(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    for imgs, labels, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            logits = model(imgs)
        all_logits.append(logits.float().cpu())
        all_labels.append(labels.cpu())
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0).numpy()
    return logits, labels


def sweep_tau_on_validation(val_logits: torch.Tensor, val_labels: np.ndarray,
                            pi_src: torch.Tensor, pi_tgt: torch.Tensor,
                            candidate_taus=(0.0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.5)):
    """
    Grid searches candidate tau values on the target validation set.
    Selects tau* based on validation macro-F1.
    """
    best_tau = 0.0
    best_f1 = -1.0
    sweep_records = {}

    for tau in candidate_taus:
        m = evaluate_metrics_from_logits(val_logits, val_labels, pi_src, pi_tgt, tau=tau)
        score = m["f1_macro"]
        sweep_records[f"tau_{tau:.2f}"] = m
        if score > best_f1:
            best_f1 = score
            best_tau = tau

    return best_tau, best_f1, sweep_records


# ------------------------------------------------------------------------------
# 4. TRAINING CC-SATDA V2 SINGLE RUN
# ------------------------------------------------------------------------------
def train_cc_satda_v2_single(
    src_train_items,
    tgt_train_items,
    val_loader,
    test_loader,
    device,
    epochs: int = 15,
    steps_per_epoch: int = 60,
    w_source: float = 0.5,
    w_target: float = 1.5,
    use_ema_prototypes: bool = True,
    use_reliability_weighting: bool = True,
    candidate_taus=(0.0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.5),
    out_ckpt: str = "",
    tag: str = ""
):
    """
    Runs a single CC-SATDA v2 training lifecycle with:
      - Dynamic priors
      - Loss weighting (w_source, w_target)
      - EMA prototype tracking
      - Logging of per-class sample exposure
      - Validation tau selection applied to test evaluation
    """
    # 1. Compute dynamic priors at runtime from actual dataset splits
    pi_src = compute_class_priors(src_train_items, num_classes=5)
    pi_tgt = compute_class_priors(tgt_train_items, num_classes=5)

    src_tf = transforms.Compose([
        transforms.Resize((288, 288)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    tgt_tf, _ = build_transforms_b2_enhanced(288)

    src_ds = SkinDomainDataset(src_train_items, src_tf, domain_label=0)
    tgt_ds = InMemorySkinDataset(
        tgt_train_items, tgt_tf, domain_label=1, img_size=288,
        cache_name=f"tgt_train_{len(tgt_train_items)}"
    )

    src_loader = DataLoader(src_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    tgt_bs = min(16, len(tgt_train_items))
    tgt_loader = DataLoader(tgt_ds, batch_size=tgt_bs, shuffle=True, num_workers=0, drop_last=True)
    src_gen = infinite_loader(src_loader)
    tgt_gen = infinite_loader(tgt_loader)

    model = ClassConditionedSATDANetV2(
        num_classes=5,
        dropout=0.3,
        use_ema_prototypes=use_ema_prototypes,
        ema_momentum=0.9,
        use_reliability_weighting=use_reliability_weighting,
        reliability_threshold=8.0,
    ).to(device)

    backbone_params = list(model.feature_extractor.parameters())
    head_params = list(model.feature_selector.parameters()) + list(model.classifier.parameters())
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": 1e-4, "weight_decay": 1e-3},
        {"params": head_params, "lr": 5e-4, "weight_decay": 5e-4},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    src_weights = compute_cost_sensitive_weights(src_train_items, device, penalty_malignant=1.5)
    tgt_weights = compute_cost_sensitive_weights(tgt_train_items, device, penalty_malignant=2.0)
    src_criterion = nn.CrossEntropyLoss(weight=src_weights)
    tgt_criterion = nn.CrossEntropyLoss(weight=tgt_weights)

    best_val_f1 = -1.0
    best_state = None
    best_epoch = -1

    print(f"[{tag}] Dynamic Priors -> Source: {pi_src.numpy().round(4).tolist()} | Target: {pi_tgt.numpy().round(4).tolist()}", flush=True)
    print(f"[{tag}] Loss Weights: (w_src={w_source}, w_tgt={w_target}) | EMA Prototypes: {use_ema_prototypes} | Reliability: {use_reliability_weighting}", flush=True)

    for epoch in range(1, epochs + 1):
        model.train()
        model.reset_epoch_batch_counts()
        total_loss, total_cls, total_proto = 0.0, 0.0, 0.0

        for _ in range(steps_per_epoch):
            src_imgs, src_labels, _ = next(src_gen)
            tgt_imgs, tgt_labels, _ = next(tgt_gen)
            src_imgs, src_labels = src_imgs.to(device, non_blocking=True), src_labels.to(device, non_blocking=True)
            tgt_imgs, tgt_labels = tgt_imgs.to(device, non_blocking=True), tgt_labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                src_logits, tgt_logits, proto_loss, div_loss = model(src_imgs, tgt_imgs, src_labels, tgt_labels)
                l_cls = w_source * src_criterion(src_logits, src_labels) + w_target * tgt_criterion(tgt_logits, tgt_labels)
                loss = l_cls + 0.2 * proto_loss + 0.01 * div_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_cls += l_cls.item()
            total_proto += proto_loss.item() if isinstance(proto_loss, torch.Tensor) else proto_loss

        scheduler.step()

        # Validation tau search
        val_logits, val_labels = collect_raw_logits(model, val_loader, device)
        best_tau_val, val_f1, _ = sweep_tau_on_validation(val_logits, val_labels, pi_src, pi_tgt, candidate_taus)
        val_metrics = evaluate_metrics_from_logits(val_logits, val_labels, pi_src, pi_tgt, tau=best_tau_val)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            if out_ckpt:
                torch.save(best_state, out_ckpt)

        # Log epoch instrumentation
        if epoch % 3 == 0 or epoch == epochs:
            counts_s = model.batch_counts["source"]
            counts_t = model.batch_counts["target"]
            co_occ = model.batch_counts["co_occurrences"]
            mel_idx = CLASS_TO_IDX["MEL"]
            print(
                f"  [{tag}] Ep {epoch:2d}/{epochs} | Loss={total_loss/steps_per_epoch:.4f} (Cls={total_cls/steps_per_epoch:.4f}, Proto={total_proto/steps_per_epoch:.4f}) | "
                f"Val F1(tau={best_tau_val:.2f})={val_metrics['f1_macro']:.4f}, Acc={val_metrics['accuracy']*100:.1f}% | "
                f"MEL Samples Seen: (Src={counts_s[mel_idx]}, Tgt={counts_t[mel_idx]}, CoOcc={co_occ[mel_idx]})",
                flush=True
            )

    # Load best checkpoint
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # Final Evaluation on Val and Held-Out Test sets
    val_logits, val_labels = collect_raw_logits(model, val_loader, device)
    optimal_tau, best_val_score, val_sweep_records = sweep_tau_on_validation(
        val_logits, val_labels, pi_src, pi_tgt, candidate_taus
    )

    test_logits, test_labels = collect_raw_logits(model, test_loader, device)
    # Test evaluated with optimal validation tau
    test_metrics_optimal_tau = evaluate_metrics_from_logits(test_logits, test_labels, pi_src, pi_tgt, tau=optimal_tau)
    # Also evaluate raw tau=0 for baseline contrast
    test_metrics_raw = evaluate_metrics_from_logits(test_logits, test_labels, pi_src, pi_tgt, tau=0.0)

    # Full test sweep over all taus for analysis
    test_sweep_records = {
        f"tau_{tau:.2f}": evaluate_metrics_from_logits(test_logits, test_labels, pi_src, pi_tgt, tau=tau)
        for tau in candidate_taus
    }

    print(
        f"  ==> [{tag}] Optimal Tau: {optimal_tau:.2f} (Val F1={best_val_score:.4f}) | "
        f"Test Acc: {test_metrics_optimal_tau['accuracy']*100:.2f}% | F1: {test_metrics_optimal_tau['f1_macro']:.4f} | "
        f"AUROC: {test_metrics_optimal_tau['auroc_macro_ovr']:.4f} | MEL-Rec: {test_metrics_optimal_tau['melanoma_recall']*100:.1f}%",
        flush=True
    )

    return {
        "best_epoch": best_epoch,
        "optimal_tau": optimal_tau,
        "val_best_f1": best_val_score,
        "test_metrics_optimal_tau": test_metrics_optimal_tau,
        "test_metrics_raw_tau0": test_metrics_raw,
        "val_sweep_records": val_sweep_records,
        "test_sweep_records": test_sweep_records,
        "priors": {
            "source": pi_src.numpy().tolist(),
            "target": pi_tgt.numpy().tolist(),
        }
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=os.environ.get("CCSATDA_DATA", "data"))
    parser.add_argument("--device", type=str, default="cuda:3")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--steps-per-epoch", type=int, default=60)
    parser.add_argument("--subset", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--w-src", type=float, default=0.5)
    parser.add_argument("--w-tgt", type=float, default=1.5)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    datasets = build_datasets_v2(args.data_dir, seed=args.seed)
    _, eval_tf_b2 = build_transforms_b2_enhanced(288)

    val_ds = InMemorySkinDataset(datasets["tgt_val"], eval_tf_b2, 1, img_size=288, cache_name="tgt_val")
    test_ds = InMemorySkinDataset(datasets["tgt_test"], eval_tf_b2, 1, img_size=288, cache_name="tgt_test")
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    tgt_key = "tgt_train_full" if args.subset == -1 else f"tgt_train_{args.subset}"
    res = train_cc_satda_v2_single(
        datasets["src_train"], datasets[tgt_key], val_loader, test_loader, device,
        epochs=args.epochs, steps_per_epoch=args.steps_per_epoch,
        w_source=args.w_src, w_target=args.w_tgt,
        tag=f"SmokeTest N={args.subset} Seed={args.seed}"
    )
    print("Smoke test completed successfully:", res["test_metrics_optimal_tau"])
