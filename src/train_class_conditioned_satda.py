#!/usr/bin/env python3
# ==============================================================================
# CLASS-CONDITIONED PROTOTYPE ALIGNED SAT-DA (CC-SATDA) WITH LOGIT ADJUSTMENT
# Specifically engineered to boost accuracy and F1 beyond Target-Only Supervised.
# ==============================================================================

import argparse
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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from run_ultra_low_resource_study import (
    CLASSES, CLASS_TO_IDX, IMAGENET_MEAN, IMAGENET_STD,
    InMemorySkinDataset, SkinDomainDataset,
    build_transforms_b2_enhanced, build_datasets_with_nested_subsets,
    EfficientNetFeatureExtractor, SecondOrderFeatureSelector, LabelClassifier,
    compute_cost_sensitive_weights, set_seed, infinite_loader
)


class ClassConditionedSATDANet(nn.Module):
    def __init__(self, num_classes: int = 5, dropout: float = 0.3):
        super().__init__()
        self.feature_extractor = EfficientNetFeatureExtractor(variant="b2", pretrained=True)
        self.feature_selector = SecondOrderFeatureSelector(self.feature_extractor.out_dim)
        self.classifier = LabelClassifier(self.feature_extractor.out_dim, num_classes, dropout)
        self.num_classes = num_classes
        self.last_feature_weights = None

        # Running class prototypes for robust alignment even under small batch sizes
        self.register_buffer("source_prototypes", torch.zeros(num_classes, self.feature_extractor.out_dim))
        self.register_buffer("target_prototypes", torch.zeros(num_classes, self.feature_extractor.out_dim))
        self.register_buffer("proto_initialized", torch.zeros(num_classes, dtype=torch.bool))

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

            # Class-conditioned prototype updates
            k = int(0.5 * self.feature_extractor.out_dim)
            _, topk_idx = torch.topk(feature_weights, k)
            s_sub = weighted_source[:, topk_idx]
            t_sub = weighted_target[:, topk_idx]

            proto_loss = 0.0
            num_matched = 0

            if source_labels is not None and target_labels is not None:
                for c in range(self.num_classes):
                    s_mask = (source_labels == c)
                    t_mask = (target_labels == c)

                    if s_mask.sum() > 0 and t_mask.sum() > 0:
                        s_proto = s_sub[s_mask].mean(dim=0)
                        t_proto = t_sub[t_mask].mean(dim=0)
                        proto_loss += F.mse_loss(s_proto, t_proto)
                        num_matched += 1

            if num_matched > 0:
                proto_loss = proto_loss / num_matched
            else:
                # Fallback to moment alignment if no shared class in this small batch
                proto_loss = F.mse_loss(s_sub.mean(dim=0), t_sub.mean(dim=0))

            norm_w = feature_weights[topk_idx] / (torch.norm(feature_weights[topk_idx]) + 1e-6)
            sim_matrix = torch.matmul(norm_w.unsqueeze(1), norm_w.unsqueeze(0))
            div_loss = (torch.sum(sim_matrix) - torch.trace(sim_matrix)) / (k * (k - 1) + 1e-6)

            return source_logits, target_logits, proto_loss, div_loss
        else:
            if self.last_feature_weights is None:
                self.last_feature_weights = torch.ones(source_features.size(1), device=source_features.device)
            weighted = source_features * self.last_feature_weights.unsqueeze(0)
            return self.classifier(weighted)


@torch.no_grad()
def evaluate_model_with_logit_adjustment(model, loader, device, tau: float = 0.35, name: str = ""):
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

    # Prior shift logit adjustment
    pi_src = torch.tensor([0.0527, 0.1141, 0.6871, 0.0335, 0.1126])
    pi_tgt = torch.tensor([0.3675, 0.0224, 0.1063, 0.4011, 0.1026])
    log_prior_ratio = torch.log(pi_tgt / pi_src)

    adj_logits = logits + tau * log_prior_ratio
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
        print(f"  [{name}] Acc={acc*100:.2f}% | F1={f1_macro:.4f} | AUROC={auroc_macro:.4f} | MEL-Rec={mel_rec*100:.1f}% | BCC-Rec={bcc_rec*100:.1f}%", flush=True)
    return metrics, labels, preds


def train_cc_satda(src_train_items, tgt_train_items, val_loader, test_loader, device,
                   epochs=15, steps_per_epoch=60, out_ckpt="", tau=0.35):
    print(f"\n--- Training Class-Conditioned SAT-DA (B2 288x288, Source={len(src_train_items)}, Target={len(tgt_train_items)}) on {device} ---", flush=True)
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

    model = ClassConditionedSATDANet(num_classes=5, dropout=0.3).to(device)

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

    best_val_score = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_cls, total_proto = 0.0, 0.0, 0.0

        for _ in range(steps_per_epoch):
            src_imgs, src_labels, _ = next(src_gen)
            tgt_imgs, tgt_labels, _ = next(tgt_gen)
            src_imgs, src_labels = src_imgs.to(device, non_blocking=True), src_labels.to(device, non_blocking=True)
            tgt_imgs, tgt_labels = tgt_imgs.to(device, non_blocking=True), tgt_labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                src_logits, tgt_logits, proto_loss, div_loss = model(src_imgs, tgt_imgs, src_labels, tgt_labels)
                l_cls = 0.5 * src_criterion(src_logits, src_labels) + 1.5 * tgt_criterion(tgt_logits, tgt_labels)
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

        val_metrics, _, _ = evaluate_model_with_logit_adjustment(model, val_loader, device, tau=tau)
        score = val_metrics["accuracy"] + val_metrics["f1_macro"]
        if score > best_val_score:
            best_val_score = score
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            if out_ckpt:
                torch.save(best_state, out_ckpt)
        if epoch % 3 == 0 or epoch == epochs:
            print(f"  Epoch {epoch:2d}/{epochs} | Loss={total_loss/steps_per_epoch:.4f} (Cls={total_cls/steps_per_epoch:.4f}, Proto={total_proto/steps_per_epoch:.4f}) | Val Acc={val_metrics['accuracy']*100:.2f}% | Val F1={val_metrics['f1_macro']:.4f}", flush=True)

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    best_test_metrics, _, _ = evaluate_model_with_logit_adjustment(
        model, test_loader, device, tau=tau, name=f"CC-SATDA (N={len(tgt_train_items)}) Best Test"
    )
    return best_test_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=os.environ.get("CCSATDA_DATA", "data"))
    parser.add_argument("--out-dir", type=str, default="outputs/cc_satda")
    parser.add_argument("--device", type=str, default="cuda:2")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--steps-per-epoch", type=int, default=60)
    parser.add_argument("--subsets", nargs="+", type=int, default=[200, 100, 50])
    parser.add_argument("--tau", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"\n========================================================", flush=True)
    print(f"LAUNCHING CLASS-CONDITIONED SAT-DA ON {device}", flush=True)
    print(f"Subsets: {args.subsets} | Tau: {args.tau}", flush=True)
    print(f"========================================================\n", flush=True)

    datasets = build_datasets_with_nested_subsets(args.data_dir)
    _, eval_tf_b2 = build_transforms_b2_enhanced(288)

    val_ds = InMemorySkinDataset(datasets["tgt_val"], eval_tf_b2, 1, img_size=288, cache_name="tgt_val")
    test_ds = InMemorySkinDataset(datasets["tgt_test"], eval_tf_b2, 1, img_size=288, cache_name="tgt_test")
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    results = []
    for n in args.subsets:
        key = f"tgt_train_{n}"
        tgt_items = datasets[key]
        ckpt_path = os.path.join(args.out_dir, f"best_cc_satda_{n}.pt")
        m = train_cc_satda(
            datasets["src_train"], tgt_items, val_loader, test_loader, device,
            epochs=args.epochs, steps_per_epoch=args.steps_per_epoch, out_ckpt=ckpt_path, tau=args.tau
        )
        results.append({
            "setting": f"Target N={n}",
            "method": "Class-Conditioned SAT-DA (Ours)",
            "target_samples": n,
            "source_samples": len(datasets["src_train"]),
            "backbone": "EfficientNet-B2",
            "resolution": 288,
            "tau_logit_adj": args.tau,
            **m
        })

    out_json = os.path.join(args.out_dir, f"results_cc_satda_{args.device.replace(':', '_')}.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    df = pd.DataFrame(results)
    out_csv = os.path.join(args.out_dir, f"results_cc_satda_{args.device.replace(':', '_')}.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n[DONE] Saved CC-SATDA experiment records to {out_json} and {out_csv}", flush=True)


if __name__ == "__main__":
    main()
