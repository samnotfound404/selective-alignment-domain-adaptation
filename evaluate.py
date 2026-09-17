#!/usr/bin/env python3
"""
CC-SAT-DA Evaluation & Prior Sweep CLI:
Evaluates a trained model checkpoint and sweeps Bayesian temperature tau.
"""

import argparse
import os
import torch
from torch.utils.data import DataLoader

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


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CC-SAT-DA checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument("--data_dir", type=str, default="./data", help="Root data folder")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--sweep_tau", action="store_true", help="Sweep tau on val split to pick best calibration")
    return parser.parse_args()


def run_eval(model, loader, device, pi_src=None, pi_tgt=None, tau=0.0):
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at: {args.checkpoint}")

    ckpt = torch.load(args.checkpoint, map_location=device)
    train_args = ckpt.get("args", {})
    backbone = train_args.get("backbone", "efficientnet_b2")
    method = train_args.get("method", "cc_satda")
    target_n = train_args.get("target_n", None)
    tau_default = train_args.get("tau", 0.35)

    print("=" * 70)
    print(f"Loaded Checkpoint: {args.checkpoint}")
    print(f"Method:            {method.upper()}")
    print(f"Backbone:          {backbone}")
    print(f"Target N:          {target_n if target_n else 'Full (1,608)'}")
    print("=" * 70)

    img_size = 288 if "b2" in backbone else 224
    _, val_tf = build_transforms(img_size=img_size)
    splits = build_domain_splits(data_dir=args.data_dir, target_n=target_n)

    val_loader = DataLoader(SkinDomainDataset(splits["val"]["target"], transform=val_tf), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(SkinDomainDataset(splits["test"]["target"], transform=val_tf), batch_size=args.batch_size, shuffle=False)

    pi_src = compute_class_priors(splits["train"]["source"]).to(device)
    pi_tgt = compute_class_priors(splits["train"]["target"]).to(device)

    # Initialize and load model
    if method == "cc_satda":
        model = CCSATDANet(num_classes=len(CLASSES), backbone=backbone, pretrained=False).to(device)
    elif method == "satda":
        model = SATDANetwork(num_classes=len(CLASSES), backbone=backbone, pretrained=False).to(device)
    else:
        encoder, fdim = build_backbone(backbone, pretrained=False)
        model = torch.nn.Sequential(
            encoder,
            torch.nn.Flatten(1),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(fdim, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(256, len(CLASSES))
        ).to(device)

    model.load_state_dict(ckpt["model_state"])

    # Optional tau grid-search on validation set only
    best_tau = tau_default
    if args.sweep_tau and method == "cc_satda":
        print("\nSweeping tau on Target Validation Set (isolated from test cohort)...")
        best_val_f1 = -1.0
        for cand_tau in [0.0, 0.1, 0.2, 0.25, 0.30, 0.35, 0.40]:
            vm = run_eval(model, val_loader, device, pi_src, pi_tgt, tau=cand_tau)
            print(f"  tau={cand_tau:4.2f} -> Val Acc: {vm['accuracy']:5.2f}% | Val Macro F1: {vm['macro_f1']:.4f}")
            if vm["macro_f1"] > best_val_f1:
                best_val_f1 = vm["macro_f1"]
                best_tau = cand_tau
        print(f"Selected Best tau: {best_tau:.2f} (Val F1: {best_val_f1:.4f})\n")

    # Evaluate on held-out test cohort
    tm = run_eval(model, test_loader, device, pi_src, pi_tgt, tau=best_tau if method == "cc_satda" else 0.0)
    print("=" * 70)
    print(f"FINAL HELD-OUT TEST EVALUATION (345 smartphone images, tau={best_tau:.2f}):")
    print("=" * 70)
    print(f"Accuracy:        {tm['accuracy']:.2f}%")
    print(f"Macro F1 Score:  {tm['macro_f1']:.4f}")
    print(f"Melanoma Recall: {tm['melanoma_recall']:.2f}% ({tm['melanoma_tp']}/{tm['melanoma_total']})")
    print("\nDetailed Per-Class Performance:")
    print(tm["report"])


if __name__ == "__main__":
    main()
