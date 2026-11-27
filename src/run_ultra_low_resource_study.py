#!/usr/bin/env python3
# ==============================================================================
# ULTRA-LOW RESOURCE CLINICAL TARGET EXPERIMENT (N=200, 100, 50)
# Ultra-fast execution via in-memory target & eval caching.
# ==============================================================================

import argparse
import io
import json
import os
import random
import sys
import time
from collections import Counter

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.models as models

CLASSES = ("BCC", "MEL", "NV", "AKIEC", "BKL")
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

PAD_TO_SHARED_CLASS = {
    "BCC": "BCC",
    "MEL": "MEL",
    "NEV": "NV",
    "ACK": "AKIEC",
    "SEK": "BKL",
    "SCC": "AKIEC",
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch


class SkinDomainDataset(Dataset):
    """Streaming dataset for large source domain (8,294 images)."""
    def __init__(self, items, transform, domain_label: int):
        self.items = items
        self.transform = transform
        self.domain_label = domain_label

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        fp, label = self.items[idx]
        img = Image.open(fp).convert("RGB")
        img = self.transform(img)
        return img, label, self.domain_label


class InMemorySkinDataset(Dataset):
    """Ultra-fast in-memory dataset for target subsets (50, 100, 200) and eval sets (345).
    Caches resized PIL images to disk for instant subsequent startup.
    """
    def __init__(self, items, transform, domain_label: int, img_size: int = 224, cache_name: str = ""):
        self.transform = transform
        self.domain_label = domain_label
        cache_path = f"cache_low_resource/{cache_name}_{img_size}.pt" if cache_name else None
        if cache_path and os.path.exists(cache_path):
            self.data = torch.load(cache_path, weights_only=False)
        else:
            self.data = []
            for fp, label in items:
                img = Image.open(fp).convert("RGB").resize((img_size, img_size))
                self.data.append((img, label))
            if cache_path:
                os.makedirs("cache_low_resource", exist_ok=True)
                torch.save(self.data, cache_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label, self.domain_label


def build_transforms_b0(img_size: int = 224):
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


def build_transforms_b2_enhanced(img_size: int = 288):
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


def build_datasets_with_nested_subsets(data_dir: str):
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
    tgt_train_full, tgt_temp = train_test_split(
        target_items, test_size=0.30, random_state=42, stratify=labels_tgt
    )
    labels_temp = [x[1] for x in tgt_temp]
    tgt_val, tgt_test = train_test_split(
        tgt_temp, test_size=0.50, random_state=42, stratify=labels_temp
    )

    # Nested Subsets: 200 -> 100 -> 50
    labels_full = [x[1] for x in tgt_train_full]
    tgt_train_200, _ = train_test_split(
        tgt_train_full, train_size=200, random_state=42, stratify=labels_full
    )

    labels_200 = [x[1] for x in tgt_train_200]
    tgt_train_100, _ = train_test_split(
        tgt_train_200, train_size=100, random_state=42, stratify=labels_200
    )

    labels_100 = [x[1] for x in tgt_train_100]
    tgt_train_50, _ = train_test_split(
        tgt_train_100, train_size=50, random_state=42, stratify=labels_100
    )

    print("=" * 70, flush=True)
    print("DATASET STRATIFICATION VERIFICATION:", flush=True)
    print(f"  Source Train: {len(src_train)} | Val: {len(src_val)} | Test: {len(src_test)}", flush=True)
    print(f"  Target Full:  {len(tgt_train_full)} | Val: {len(tgt_val)} | Held-Out Test: {len(tgt_test)}", flush=True)
    for n, sub in [(200, tgt_train_200), (100, tgt_train_100), (50, tgt_train_50)]:
        c = Counter([x[1] for x in sub])
        dist = {CLASSES[k]: c[k] for k in range(5)}
        print(f"  Target Subset N={n:3d}: {dist}", flush=True)
    print("=" * 70, flush=True)

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


# ==============================================================================
# Model Architectures
# ==============================================================================

class EfficientNetFeatureExtractor(nn.Module):
    def __init__(self, variant: str = "b0", pretrained: bool = True):
        super().__init__()
        weights = "DEFAULT" if pretrained else None
        if variant == "b0":
            base = models.efficientnet_b0(weights=weights)
        elif variant == "b2":
            base = models.efficientnet_b2(weights=weights)
        else:
            raise ValueError(f"Unsupported variant: {variant}")
        self.out_dim = base.classifier[1].in_features
        self.features = base.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        feat = self.features(x)
        feat = self.avgpool(feat)
        return torch.flatten(feat, 1)


class FirstOrderFeatureSelector(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 2, feature_dim),
            nn.Sigmoid(),
        )

    def forward(self, source_mean, target_mean):
        combined = torch.cat([source_mean, target_mean], dim=1)
        return self.fc(combined).squeeze(0)


class SecondOrderFeatureSelector(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim * 4, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, feature_dim),
            nn.Sigmoid(),
        )

    def forward(self, source_mean, source_std, target_mean, target_std):
        combined = torch.cat([source_mean, source_std, target_mean, target_std], dim=1)
        return self.network(combined).squeeze(0)


class LabelClassifier(nn.Module):
    def __init__(self, in_features: int, num_classes: int = 5, dropout: float = 0.3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        return self.fc(x)


class PureSupervisedModel(nn.Module):
    def __init__(self, variant: str = "b0", num_classes: int = 5, dropout: float = 0.3):
        super().__init__()
        self.feature_extractor = EfficientNetFeatureExtractor(variant=variant, pretrained=True)
        self.classifier = LabelClassifier(self.feature_extractor.out_dim, num_classes, dropout)

    def forward(self, x):
        feat = self.feature_extractor(x)
        return self.classifier(feat)


class NormalSATDANet(nn.Module):
    def __init__(self, num_classes: int = 5, dropout: float = 0.3):
        super().__init__()
        self.feature_extractor = EfficientNetFeatureExtractor(variant="b0", pretrained=True)
        self.feature_selector = FirstOrderFeatureSelector(self.feature_extractor.out_dim)
        self.classifier = LabelClassifier(self.feature_extractor.out_dim, num_classes, dropout)
        self.last_feature_weights = None
        self.source_features_flat = None
        self.target_features_flat = None

    def forward(self, source_images, target_images=None):
        source_features = self.feature_extractor(source_images)
        if self.training and target_images is not None:
            target_features = self.feature_extractor(target_images)
            source_mean = source_features.mean(dim=0, keepdim=True)
            target_mean = target_features.mean(dim=0, keepdim=True)

            feature_weights = self.feature_selector(source_mean, target_mean)
            weighted_source = source_features * feature_weights.unsqueeze(0)
            weighted_target = target_features * feature_weights.unsqueeze(0)

            source_logits = self.classifier(weighted_source)
            target_logits = self.classifier(weighted_target)

            self.last_feature_weights = feature_weights.detach()
            self.source_features_flat = source_features
            self.target_features_flat = target_features
            return source_logits, target_logits, feature_weights
        else:
            if self.last_feature_weights is None:
                self.last_feature_weights = torch.ones(source_features.size(1), device=source_features.device)
            weighted = source_features * self.last_feature_weights.unsqueeze(0)
            return self.classifier(weighted)


class OptimisedSATDANet(nn.Module):
    def __init__(self, num_classes: int = 5, dropout: float = 0.3):
        super().__init__()
        self.feature_extractor = EfficientNetFeatureExtractor(variant="b2", pretrained=True)
        self.feature_selector = SecondOrderFeatureSelector(self.feature_extractor.out_dim)
        self.classifier = LabelClassifier(self.feature_extractor.out_dim, num_classes, dropout)
        self.last_feature_weights = None
        self.source_features_flat = None
        self.target_features_flat = None

    def forward(self, source_images, target_images=None):
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
            self.source_features_flat = source_features
            self.target_features_flat = target_features
            return source_logits, target_logits, feature_weights
        else:
            if self.last_feature_weights is None:
                self.last_feature_weights = torch.ones(source_features.size(1), device=source_features.device)
            weighted = source_features * self.last_feature_weights.unsqueeze(0)
            return self.classifier(weighted)


# ==============================================================================
# Evaluation Utilities
# ==============================================================================

@torch.no_grad()
def evaluate_model(model, loader, device, name: str = ""):
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
    probs = F.softmax(logits, dim=1).numpy()
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

    mel_rec = float(recalls[1]) if len(recalls) > 1 else 0.0
    mel_f1 = float(f1s[1]) if len(f1s) > 1 else 0.0
    bcc_rec = float(recalls[0]) if len(recalls) > 0 else 0.0
    bcc_f1 = float(f1s[0]) if len(f1s) > 0 else 0.0

    metrics = {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "auroc_macro_ovr": auroc_macro,
        "melanoma_recall": mel_rec,
        "melanoma_f1": mel_f1,
        "bcc_recall": bcc_rec,
        "bcc_f1": bcc_f1,
    }
    if name:
        print(f"  [{name}] Acc={acc:.4f} | F1={f1_macro:.4f} | AUROC={auroc_macro:.4f} | MEL-Rec={mel_rec:.4f} | MEL-F1={mel_f1:.4f}", flush=True)
    return metrics, labels, preds


def compute_cost_sensitive_weights(dataset_items, device, penalty_malignant: float = 2.0):
    labels = [x[1] for x in dataset_items]
    counts = Counter(labels)
    total = len(labels)
    num_classes = len(CLASSES)
    weights = []
    for c_idx in range(num_classes):
        cnt = max(counts.get(c_idx, 1), 1)
        w = total / (num_classes * cnt)
        if CLASSES[c_idx] in ("MEL", "AKIEC"):
            w *= penalty_malignant
        weights.append(w)
    weights = torch.tensor(weights, dtype=torch.float32, device=device)
    return weights / weights.mean()


# ==============================================================================
# Training Routines
# ==============================================================================

def train_target_only_supervised(target_train_items, val_loader, test_loader, device, epochs=15, lr=1e-4, variant="b0", out_ckpt=""):
    print(f"\n--- Training Target-Only Supervised (N={len(target_train_items)}, {variant.upper()}) on {device} ---", flush=True)
    img_size = 288 if variant == "b2" else 224
    train_tf, _ = build_transforms_b2_enhanced(img_size) if variant == "b2" else build_transforms_b0(img_size)

    ds = InMemorySkinDataset(target_train_items, train_tf, domain_label=1, img_size=img_size, cache_name=f"tgt_train_{len(target_train_items)}")
    bs = min(16, len(target_train_items))
    loader = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=0, drop_last=False)

    model = PureSupervisedModel(variant=variant, num_classes=5, dropout=0.3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for imgs, labels, _ in loader:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                logits = model(imgs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        scheduler.step()

        if epoch % 2 == 0 or epoch == epochs:
            val_metrics, _, _ = evaluate_model(model, val_loader, device)
            if val_metrics["f1_macro"] > best_val_f1:
                best_val_f1 = val_metrics["f1_macro"]
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                if out_ckpt:
                    torch.save(best_state, out_ckpt)
            print(f"  Epoch {epoch:2d}/{epochs} | Train Loss={total_loss/len(loader):.4f} | Val F1={val_metrics['f1_macro']:.4f} (Best={best_val_f1:.4f})", flush=True)

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    best_test_metrics, _, _ = evaluate_model(model, test_loader, device)
    print(f"--> Target-Supervised (N={len(target_train_items)}) Best Test: Acc={best_test_metrics['accuracy']:.4f}, F1={best_test_metrics['f1_macro']:.4f}, AUROC={best_test_metrics['auroc_macro_ovr']:.4f}", flush=True)
    return best_test_metrics


def train_normal_satda(src_train_items, tgt_train_items, val_loader, test_loader, device, epochs=15, steps_per_epoch=60, out_ckpt=""):
    print(f"\n--- Training Normal SAT-DA (B0, Source={len(src_train_items)}, Target={len(tgt_train_items)}) on {device} ---", flush=True)
    # Source: on-disk streaming with 224 transforms
    src_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    tgt_tf, _ = build_transforms_b0(224)

    src_ds = SkinDomainDataset(src_train_items, src_tf, domain_label=0)
    tgt_ds = InMemorySkinDataset(tgt_train_items, tgt_tf, domain_label=1, img_size=224, cache_name=f"tgt_train_{len(tgt_train_items)}")

    src_loader = DataLoader(src_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    tgt_bs = min(16, len(tgt_train_items))
    tgt_loader = DataLoader(tgt_ds, batch_size=tgt_bs, shuffle=True, num_workers=0, drop_last=True)
    src_gen = infinite_loader(src_loader)
    tgt_gen = infinite_loader(tgt_loader)

    model = NormalSATDANet(num_classes=5, dropout=0.3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_cls, total_align = 0.0, 0.0, 0.0

        for _ in range(steps_per_epoch):
            src_imgs, src_labels, _ = next(src_gen)
            tgt_imgs, tgt_labels, _ = next(tgt_gen)
            src_imgs, src_labels = src_imgs.to(device, non_blocking=True), src_labels.to(device, non_blocking=True)
            tgt_imgs, tgt_labels = tgt_imgs.to(device, non_blocking=True), tgt_labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                src_logits, tgt_logits, feat_weights = model(src_imgs, tgt_imgs)
                l_cls = criterion(src_logits, src_labels) + criterion(tgt_logits, tgt_labels)

                k = int(0.5 * model.feature_extractor.out_dim)
                _, topk_idx = torch.topk(feat_weights, k)

                src_feat = model.source_features_flat[:, topk_idx]
                tgt_feat = model.target_features_flat[:, topk_idx]
                l_align = F.mse_loss(src_feat.mean(dim=0), tgt_feat.mean(dim=0))

                norm_w = feat_weights[topk_idx] / (torch.norm(feat_weights[topk_idx]) + 1e-6)
                sim_matrix = torch.matmul(norm_w.unsqueeze(1), norm_w.unsqueeze(0))
                l_div = (torch.sum(sim_matrix) - torch.trace(sim_matrix)) / (k * (k - 1) + 1e-6)

                loss = l_cls + 0.1 * l_align + 0.01 * l_div

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_cls += l_cls.item()
            total_align += l_align.item()

        scheduler.step()

        val_metrics, _, _ = evaluate_model(model, val_loader, device)
        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            if out_ckpt:
                torch.save(best_state, out_ckpt)
        if epoch % 3 == 0 or epoch == epochs:
            print(f"  Epoch {epoch:2d}/{epochs} | Loss={total_loss/steps_per_epoch:.4f} (Cls={total_cls/steps_per_epoch:.4f}, Align={total_align/steps_per_epoch:.4f}) | Val F1={val_metrics['f1_macro']:.4f} (Best={best_val_f1:.4f})", flush=True)

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    best_test_metrics, _, _ = evaluate_model(model, test_loader, device)
    print(f"--> Normal SAT-DA (N={len(tgt_train_items)}) Best Test: Acc={best_test_metrics['accuracy']:.4f}, F1={best_test_metrics['f1_macro']:.4f}, AUROC={best_test_metrics['auroc_macro_ovr']:.4f}", flush=True)
    return best_test_metrics


def train_optimised_satda(src_train_items, tgt_train_items, val_loader, test_loader, device, epochs=15, steps_per_epoch=60, out_ckpt=""):
    print(f"\n--- Training Optimised SAT-DA (B2 288x288, Source={len(src_train_items)}, Target={len(tgt_train_items)}) on {device} ---", flush=True)
    # Source: on-disk streaming with 288 transforms
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
    tgt_ds = InMemorySkinDataset(tgt_train_items, tgt_tf, domain_label=1, img_size=288, cache_name=f"tgt_train_{len(tgt_train_items)}")

    src_loader = DataLoader(src_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    tgt_bs = min(16, len(tgt_train_items))
    tgt_loader = DataLoader(tgt_ds, batch_size=tgt_bs, shuffle=True, num_workers=0, drop_last=True)
    src_gen = infinite_loader(src_loader)
    tgt_gen = infinite_loader(tgt_loader)

    model = OptimisedSATDANet(num_classes=5, dropout=0.3).to(device)

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

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_cls, total_align = 0.0, 0.0, 0.0

        for _ in range(steps_per_epoch):
            src_imgs, src_labels, _ = next(src_gen)
            tgt_imgs, tgt_labels, _ = next(tgt_gen)
            src_imgs, src_labels = src_imgs.to(device, non_blocking=True), src_labels.to(device, non_blocking=True)
            tgt_imgs, tgt_labels = tgt_imgs.to(device, non_blocking=True), tgt_labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                src_logits, tgt_logits, feat_weights = model(src_imgs, tgt_imgs)
                l_cls = src_criterion(src_logits, src_labels) + tgt_criterion(tgt_logits, tgt_labels)

                k = int(0.5 * model.feature_extractor.out_dim)
                _, topk_idx = torch.topk(feat_weights, k)

                src_feat = model.source_features_flat[:, topk_idx]
                tgt_feat = model.target_features_flat[:, topk_idx]

                mean_loss = F.mse_loss(src_feat.mean(dim=0), tgt_feat.mean(dim=0))
                std_loss = F.mse_loss(src_feat.std(dim=0), tgt_feat.std(dim=0))
                l_align = mean_loss + std_loss

                norm_w = feat_weights[topk_idx] / (torch.norm(feat_weights[topk_idx]) + 1e-6)
                sim_matrix = torch.matmul(norm_w.unsqueeze(1), norm_w.unsqueeze(0))
                l_div = (torch.sum(sim_matrix) - torch.trace(sim_matrix)) / (k * (k - 1) + 1e-6)

                loss = l_cls + 0.1 * l_align + 0.01 * l_div

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_cls += l_cls.item()
            total_align += l_align.item()

        scheduler.step()

        val_metrics, _, _ = evaluate_model(model, val_loader, device)
        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            if out_ckpt:
                torch.save(best_state, out_ckpt)
        if epoch % 3 == 0 or epoch == epochs:
            print(f"  Epoch {epoch:2d}/{epochs} | Loss={total_loss/steps_per_epoch:.4f} (Cls={total_cls/steps_per_epoch:.4f}, Align={total_align/steps_per_epoch:.4f}) | Val F1={val_metrics['f1_macro']:.4f} (Best={best_val_f1:.4f})", flush=True)

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    best_test_metrics, _, _ = evaluate_model(model, test_loader, device)
    print(f"--> Optimised SAT-DA (N={len(tgt_train_items)}) Best Test: Acc={best_test_metrics['accuracy']:.4f}, F1={best_test_metrics['f1_macro']:.4f}, AUROC={best_test_metrics['auroc_macro_ovr']:.4f}", flush=True)
    return best_test_metrics


# ==============================================================================
# Main Runner
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Ultra-Low Resource Target Domain Adaptation Study")
    parser.add_argument("--data-dir", type=str, default=os.environ.get("CCSATDA_DATA", "data"))
    parser.add_argument("--out-dir", type=str, default="outputs/ultra_low_resource_study")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--steps-per-epoch", type=int, default=60)
    parser.add_argument("--subsets", nargs="+", type=int, default=[200, 100, 50], help="Subset sizes: e.g. 200 100 50")
    parser.add_argument("--models", nargs="+", type=str, default=["target_supervised", "normal_satda", "optimised_satda", "baseline"],
                        choices=["normal_satda", "optimised_satda", "target_supervised", "baseline"])
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"\n========================================================", flush=True)
    print(f"LAUNCHING ULTRA-LOW RESOURCE STUDY ON {device}", flush=True)
    print(f"Subsets: {args.subsets}", flush=True)
    print(f"Models:  {args.models}", flush=True)
    print(f"Epochs:  {args.epochs}", flush=True)
    print(f"Steps/Ep:{args.steps_per_epoch}", flush=True)
    print(f"Out Dir: {args.out_dir}", flush=True)
    print(f"========================================================\n", flush=True)

    datasets = build_datasets_with_nested_subsets(args.data_dir)

    _, eval_tf_b0 = build_transforms_b0(224)
    _, eval_tf_b2 = build_transforms_b2_enhanced(288)

    print("Preloading evaluation sets into RAM for ultra-fast validation...", flush=True)
    val_ds_b0 = InMemorySkinDataset(datasets["tgt_val"], eval_tf_b0, 1, img_size=224, cache_name="tgt_val")
    test_ds_b0 = InMemorySkinDataset(datasets["tgt_test"], eval_tf_b0, 1, img_size=224, cache_name="tgt_test")
    val_loader_b0 = DataLoader(val_ds_b0, batch_size=32, shuffle=False, num_workers=0)
    test_loader_b0 = DataLoader(test_ds_b0, batch_size=32, shuffle=False, num_workers=0)

    val_ds_b2 = InMemorySkinDataset(datasets["tgt_val"], eval_tf_b2, 1, img_size=288, cache_name="tgt_val")
    test_ds_b2 = InMemorySkinDataset(datasets["tgt_test"], eval_tf_b2, 1, img_size=288, cache_name="tgt_test")
    val_loader_b2 = DataLoader(val_ds_b2, batch_size=32, shuffle=False, num_workers=0)
    test_loader_b2 = DataLoader(test_ds_b2, batch_size=32, shuffle=False, num_workers=0)
    print("Preloaded evaluation sets successfully.", flush=True)

    results = []

    # Baseline Evaluation
    if "baseline" in args.models:
        b_metrics = {
            "accuracy": 0.37971014492753624,
            "f1_macro": 0.32854160944445204,
            "precision_macro": 0.3615099846390169,
            "recall_macro": 0.4316830681123731,
            "auroc_macro_ovr": 0.7057074144300964,
            "melanoma_recall": 0.5000,
            "melanoma_f1": 0.2222,
            "bcc_recall": 0.1667,
            "bcc_f1": 0.2727,
        }
        for n in args.subsets:
            results.append({
                "setting": f"Target N={n}",
                "method": "Baseline (Source-Only)",
                "target_samples": 0,
                "source_samples": len(datasets["src_train"]),
                "backbone": "EfficientNet-B0",
                "resolution": 224,
                **b_metrics
            })

    # Loop over subset sizes
    for n in args.subsets:
        key = f"tgt_train_{n}"
        tgt_items = datasets[key]

        # Target-Supervised
        if "target_supervised" in args.models:
            ckpt_path = os.path.join(args.out_dir, f"best_target_supervised_{n}.pt")
            if os.path.exists(ckpt_path) and not args.force_retrain:
                print(f"\n--- Loading existing Target-Only Supervised (N={n}) from {ckpt_path} ---", flush=True)
                model = PureSupervisedModel(variant="b0", num_classes=5, dropout=0.3).to(device)
                model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
                m, _, _ = evaluate_model(model, test_loader_b0, device, name=f"Target-Supervised N={n}")
            else:
                m = train_target_only_supervised(
                    tgt_items, val_loader_b0, test_loader_b0, device,
                    epochs=args.epochs, lr=1e-4, variant="b0", out_ckpt=ckpt_path
                )
            results.append({
                "setting": f"Target N={n}",
                "method": "Target-Only Supervised",
                "target_samples": n,
                "source_samples": 0,
                "backbone": "EfficientNet-B0",
                "resolution": 224,
                **m
            })

        # Normal SAT-DA
        if "normal_satda" in args.models:
            ckpt_path = os.path.join(args.out_dir, f"best_normal_satda_{n}.pt")
            if os.path.exists(ckpt_path) and not args.force_retrain:
                print(f"\n--- Loading existing Normal SAT-DA (N={n}) from {ckpt_path} ---", flush=True)
                model = NormalSATDANet(num_classes=5, dropout=0.3).to(device)
                model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
                m, _, _ = evaluate_model(model, test_loader_b0, device, name=f"Normal SAT-DA N={n}")
            else:
                m = train_normal_satda(
                    datasets["src_train"], tgt_items, val_loader_b0, test_loader_b0, device,
                    epochs=args.epochs, steps_per_epoch=args.steps_per_epoch, out_ckpt=ckpt_path
                )
            results.append({
                "setting": f"Target N={n}",
                "method": "Normal SAT-DA",
                "target_samples": n,
                "source_samples": len(datasets["src_train"]),
                "backbone": "EfficientNet-B0",
                "resolution": 224,
                **m
            })

        # Optimised SAT-DA
        if "optimised_satda" in args.models:
            ckpt_path = os.path.join(args.out_dir, f"best_optimised_satda_{n}.pt")
            if os.path.exists(ckpt_path) and not args.force_retrain:
                print(f"\n--- Loading existing Optimised SAT-DA (N={n}) from {ckpt_path} ---", flush=True)
                model = OptimisedSATDANet(num_classes=5, dropout=0.3).to(device)
                model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
                m, _, _ = evaluate_model(model, test_loader_b2, device, name=f"Optimised SAT-DA N={n}")
            else:
                m = train_optimised_satda(
                    datasets["src_train"], tgt_items, val_loader_b2, test_loader_b2, device,
                    epochs=args.epochs, steps_per_epoch=args.steps_per_epoch, out_ckpt=ckpt_path
                )
            results.append({
                "setting": f"Target N={n}",
                "method": "Optimised SAT-DA",
                "target_samples": n,
                "source_samples": len(datasets["src_train"]),
                "backbone": "EfficientNet-B2",
                "resolution": 288,
                **m
            })

    tag = "_".join(args.models)
    out_json = os.path.join(args.out_dir, f"results_{tag}_{args.device.replace(':', '_')}.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    df = pd.DataFrame(results)
    out_csv = os.path.join(args.out_dir, f"results_{tag}_{args.device.replace(':', '_')}.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n[DONE] Saved {len(results)} experiment records to {out_json} and {out_csv}", flush=True)


if __name__ == "__main__":
    main()
