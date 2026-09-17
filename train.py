#!/usr/bin/env python3
"""
CC-SAT-DA Training CLI:
Train Class-Conditioned Selective Alignment Transfer models or Baselines.
"""

import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from cc_satda.models import CCSATDANet, SATDANetwork, build_backbone
from cc_satda.data import (
    SkinDomainDataset,
    build_transforms,
    build_domain_splits,
    CLASSES,
)
from cc_satda.utils import (
    compute_class_priors,
    calibrate_logits,
    compute_classification_metrics,
)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch


def parse_args():
    parser = argparse.ArgumentParser(description="Train CC-SAT-DA on Skin Lesion Cross-Domain Benchmark")
    parser.add_argument("--data_dir", type=str, default="./data", help="Root path containing ISIC and PAD datasets")
    parser.add_argument("--method", type=str, default="cc_satda", choices=["cc_satda", "satda", "target_supervised", "source_only"])
    parser.add_argument("--backbone", type=str, default="efficientnet_b2", choices=["efficientnet_b2", "efficientnet_b0", "resnet18", "resnet50"])
    parser.add_argument("--target_n", type=int, default=None, choices=[50, 100, 200, 1608], help="Target training cohort size (None = full 1,608)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (half source, half target in DA)")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for AdamW")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay for AdamW")
    parser.add_argument("--w_src", type=float, default=0.5, help="Source classification loss weight")
    parser.add_argument("--w_tgt", type=float, default=1.5, help="Target classification loss weight")
    parser.add_argument("--lambda_proto", type=float, default=0.2, help="Prototype alignment loss weight")
    parser.add_argument("--lambda_disp", type=float, default=0.01, help="Inter-class dispersion loss weight")
    parser.add_argument("--tau", type=float, default=0.35, help="Bayesian prior calibration temperature")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory to save model weights")
    return parser.parse_args()


def evaluate(model, loader, device, pi_src=None, pi_tgt=None, tau=0.0):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            if tau > 0.0 and pi_src is not None and pi_tgt is not None:
                logits = calibrate_logits(logits, pi_src, pi_tgt, tau=tau)
            preds = logits.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_targets.extend(lbls.tolist())

    return compute_classification_metrics(all_targets, all_preds, class_names=CLASSES)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print("=" * 70)
    print(f"Training Method:   {args.method.upper()}")
    print(f"Backbone:          {args.backbone}")
    print(f"Target Size (N):   {args.target_n if args.target_n else 'Full (1,608)'}")
    print(f"Device:            {device}")
    print("=" * 70)

    # 1. Datasets & Transforms
    img_size = 288 if "b2" in args.backbone else 224
    train_tf, val_tf = build_transforms(img_size=img_size)
    splits = build_domain_splits(data_dir=args.data_dir, target_n=args.target_n, seed=args.seed)

    src_train_ds = SkinDomainDataset(splits["train"]["source"], transform=train_tf)
    tgt_train_ds = SkinDomainDataset(splits["train"]["target"], transform=train_tf)
    tgt_val_ds = SkinDomainDataset(splits["val"]["target"], transform=val_tf)
    tgt_test_ds = SkinDomainDataset(splits["test"]["target"], transform=val_tf)

    print(f"Source Train: {len(src_train_ds)} | Target Train: {len(tgt_train_ds)}")
    print(f"Target Val:   {len(tgt_val_ds)} | Target Test:  {len(tgt_test_ds)}")

    pi_src = compute_class_priors(splits["train"]["source"]).to(device)
    pi_tgt = compute_class_priors(splits["train"]["target"]).to(device)

    # 2. Data Loaders
    half_bs = max(args.batch_size // 2, 4)
    src_loader = DataLoader(src_train_ds, batch_size=half_bs, shuffle=True, drop_last=True, num_workers=2)
    tgt_loader = DataLoader(tgt_train_ds, batch_size=half_bs, shuffle=True, drop_last=True, num_workers=2)
    val_loader = DataLoader(tgt_val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(tgt_test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # 3. Model Initialization
    if args.method == "cc_satda":
        model = CCSATDANet(num_classes=len(CLASSES), backbone=args.backbone, pretrained=True).to(device)
    elif args.method == "satda":
        model = SATDANetwork(num_classes=len(CLASSES), backbone=args.backbone, pretrained=True).to(device)
    else:  # source_only or target_supervised
        encoder, fdim = build_backbone(args.backbone, pretrained=True)
        model = nn.Sequential(
            encoder,
            nn.Flatten(1),
            nn.Dropout(0.5),
            nn.Linear(fdim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, len(CLASSES))
        ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu')

    steps_per_epoch = max(len(tgt_loader), 20)
    src_iter = infinite_loader(src_loader)
    tgt_iter = infinite_loader(tgt_loader)

    best_val_f1 = 0.0
    save_path = os.path.join(args.checkpoint_dir, f"{args.method}_{args.backbone}_n{args.target_n}_best.pth")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        for _ in range(steps_per_epoch):
            optimizer.zero_grad()
            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                if args.method == "cc_satda":
                    s_x, s_y = next(src_iter)
                    t_x, t_y = next(tgt_iter)
                    s_x, s_y = s_x.to(device), s_y.to(device)
                    t_x, t_y = t_x.to(device), t_y.to(device)

                    s_logits, t_logits, proto_loss, div_loss = model(s_x, t_x, s_y, t_y)
                    loss_cls = args.w_src * criterion(s_logits, s_y) + args.w_tgt * criterion(t_logits, t_y)
                    loss = loss_cls + args.lambda_proto * proto_loss + args.lambda_disp * div_loss

                elif args.method == "satda":
                    s_x, s_y = next(src_iter)
                    t_x, t_y = next(tgt_iter)
                    s_x, s_y = s_x.to(device), s_y.to(device)
                    t_x, t_y = t_x.to(device), t_y.to(device)

                    s_logits, t_logits, align_loss, div_loss = model(s_x, t_x)
                    loss_cls = args.w_src * criterion(s_logits, s_y) + args.w_tgt * criterion(t_logits, t_y)
                    loss = loss_cls + 0.1 * align_loss + 0.01 * div_loss

                elif args.method == "target_supervised":
                    t_x, t_y = next(tgt_iter)
                    t_x, t_y = t_x.to(device), t_y.to(device)
                    logits = model(t_x)
                    loss = criterion(logits, t_y)

                else:  # source_only
                    s_x, s_y = next(src_iter)
                    s_x, s_y = s_x.to(device), s_y.to(device)
                    logits = model(s_x)
                    loss = criterion(logits, s_y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        scheduler.step()

        # Validation (with Bayes prior shift if DA method)
        tau_val = args.tau if args.method == "cc_satda" else 0.0
        val_metrics = evaluate(model, val_loader, device, pi_src, pi_tgt, tau=tau_val)

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_f1": best_val_f1,
                "args": vars(args)
            }, save_path)

        if epoch % 5 == 0 or epoch == args.epochs:
            print(f"Epoch [{epoch:02d}/{args.epochs:02d}] "
                  f"Loss: {epoch_loss/steps_per_epoch:.4f} | "
                  f"Val Acc: {val_metrics['accuracy']:.2f}% | "
                  f"Val F1: {val_metrics['macro_f1']:.4f} (Best: {best_val_f1:.4f})")

    # 4. Final Evaluation on Fixed 345 Held-Out Test Clinical Images
    print("\n" + "=" * 70)
    print("FINAL EVALUATION ON HELD-OUT TEST COHORT (345 Clinical Images)")
    print("=" * 70)

    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics = evaluate(model, test_loader, device, pi_src, pi_tgt, tau=args.tau if args.method == "cc_satda" else 0.0)

    print(f"Test Accuracy:   {test_metrics['accuracy']:.2f}%")
    print(f"Macro F1 Score:  {test_metrics['macro_f1']:.4f}")
    print(f"Melanoma Recall: {test_metrics['melanoma_recall']:.2f}% "
          f"({test_metrics['melanoma_tp']}/{test_metrics['melanoma_total']})")
    print("\nDetailed Classification Report:")
    print(test_metrics["report"])


if __name__ == "__main__":
    main()
