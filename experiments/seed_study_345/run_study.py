#!/usr/bin/env python3
"""Five-seed study on the report's 345-image PAD-UFES-20 split.

Trains ONE (method, N, seed) with the same recipe as the report and saves the
validation and test logits of the selected checkpoint, so that calibration
(fixed tau=0.35 as in the report, or validation-selected "safety" calibration)
can be applied afterwards identically to every method.

Methods (identical to the report's runs):
  sup   : target-only supervised, EfficientNet-B0 @224 for N<=200, B2 @288 for full N
          (run_ultra_low_resource_study.train_target_only_supervised)
  satda : SAT-DA, EfficientNet-B2 @288, global moment alignment
          (run_ultra_low_resource_study.train_optimised_satda)
  cc    : CC-SAT-DA, EfficientNet-B2 @288, class-conditional prototypes
          (train_class_conditioned_satda.train_cc_satda)

One deliberate change for both SAT-DA and CC-SAT-DA: the channel-weight vector
used at evaluation is the one from the SELECTED epoch (the originals kept the
last training step's weights, which are not part of the checkpoint).
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from torchvision import transforms

import run_ultra_low_resource_study as U
from train_class_conditioned_satda import ClassConditionedSATDANet

WORK_DIR = os.environ.get("CCSATDA_STUDY_DIR", os.path.join(REPO, "results", "seed_study_345"))
DATA_DIR = os.path.abspath(os.environ.get("CCSATDA_DATA", os.path.join(REPO, "data")))
PI_SRC = np.array([0.0527, 0.1141, 0.6871, 0.0335, 0.1126])
PI_TGT = np.array([0.3675, 0.0224, 0.1063, 0.4011, 0.1026])
LOG_RATIO = np.log(PI_TGT / PI_SRC)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def metrics_from_logits(logits, y, tau=0.0, delta=0.0):
    z = logits + tau * LOG_RATIO
    z = z.copy()
    z[:, 1] += delta
    p = softmax(z)
    pred = p.argmax(1)
    out = {
        "accuracy": float(accuracy_score(y, pred)),
        "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
    }
    try:
        out["auroc_macro_ovr"] = float(roc_auc_score(y, p, multi_class="ovr", average="macro"))
    except Exception:
        out["auroc_macro_ovr"] = float("nan")
    mel = y == 1
    out["mel_found"] = int(((pred == 1) & mel).sum())
    out["mel_total"] = int(mel.sum())
    out["mel_pred"] = int((pred == 1).sum())
    out["melanoma_recall"] = out["mel_found"] / max(out["mel_total"], 1)
    return out


@torch.no_grad()
def get_logits(model, loader, device):
    model.eval()
    L, Y = [], []
    for imgs, labels, _ in loader:
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            out = model(imgs.to(device, non_blocking=True))
        L.append(out.float().cpu())
        Y.append(labels.cpu())
    return torch.cat(L).numpy(), torch.cat(Y).numpy()


def src_transform():
    return transforms.Compose([
        transforms.Resize((288, 288)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(U.IMAGENET_MEAN, U.IMAGENET_STD),
    ])


# ----------------------------------------------------------------------------
# Training loops: copies of the report's functions; only change = return the
# model and keep the selected epoch's channel weights.
# ----------------------------------------------------------------------------

def train_sup(tgt_items, val_loader, device, variant, epochs=15, lr=1e-4):
    img_size = 288 if variant == "b2" else 224
    train_tf, _ = U.build_transforms_b2_enhanced(img_size) if variant == "b2" else U.build_transforms_b0(img_size)
    ds = U.InMemorySkinDataset(tgt_items, train_tf, domain_label=1, img_size=img_size, cache_name=f"tgt_train_{len(tgt_items)}")
    loader = DataLoader(ds, batch_size=min(16, len(tgt_items)), shuffle=True, num_workers=0, drop_last=False)
    model = U.PureSupervisedModel(variant=variant, num_classes=5, dropout=0.3).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    crit = nn.CrossEntropyLoss()
    best, best_state, best_ep = -1.0, None, -1
    for ep in range(1, epochs + 1):
        model.train()
        for imgs, labels, _ in loader:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                loss = crit(model(imgs), labels)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
        sched.step()
        if ep % 2 == 0 or ep == epochs:
            lg, y = get_logits(model, val_loader, device)
            m = metrics_from_logits(lg, y)
            if m["f1_macro"] > best:
                best, best_ep = m["f1_macro"], ep
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"  ep {ep:2d} val acc={m['accuracy']:.4f} f1={m['f1_macro']:.4f} (best {best:.4f})", flush=True)
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model, best_ep


def train_da(method, src_items, tgt_items, val_loader, device, epochs=15, steps=60, tau_select=0.35):
    tgt_tf, _ = U.build_transforms_b2_enhanced(288)
    src_ds = U.SkinDomainDataset(src_items, src_transform(), domain_label=0)
    tgt_ds = U.InMemorySkinDataset(tgt_items, tgt_tf, domain_label=1, img_size=288, cache_name=f"tgt_train_{len(tgt_items)}")
    src_loader = DataLoader(src_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    tgt_loader = DataLoader(tgt_ds, batch_size=min(16, len(tgt_items)), shuffle=True, num_workers=0, drop_last=True)
    src_gen, tgt_gen = U.infinite_loader(src_loader), U.infinite_loader(tgt_loader)

    model = (ClassConditionedSATDANet if method == "cc" else U.OptimisedSATDANet)(num_classes=5, dropout=0.3).to(device)
    opt = torch.optim.AdamW([
        {"params": list(model.feature_extractor.parameters()), "lr": 1e-4, "weight_decay": 1e-3},
        {"params": list(model.feature_selector.parameters()) + list(model.classifier.parameters()), "lr": 5e-4, "weight_decay": 5e-4},
    ])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    src_crit = nn.CrossEntropyLoss(weight=U.compute_cost_sensitive_weights(src_items, device, penalty_malignant=1.5))
    tgt_crit = nn.CrossEntropyLoss(weight=U.compute_cost_sensitive_weights(tgt_items, device, penalty_malignant=2.0))

    best, best_state, best_w, best_ep = -1.0, None, None, -1
    for ep in range(1, epochs + 1):
        model.train()
        for _ in range(steps):
            si, sl, _ = next(src_gen)
            ti, tl, _ = next(tgt_gen)
            si, sl = si.to(device, non_blocking=True), sl.to(device, non_blocking=True)
            ti, tl = ti.to(device, non_blocking=True), tl.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                if method == "cc":
                    s_lg, t_lg, proto, div = model(si, ti, sl, tl)
                    loss = 0.5 * src_crit(s_lg, sl) + 1.5 * tgt_crit(t_lg, tl) + 0.2 * proto + 0.01 * div
                else:
                    s_lg, t_lg, fw = model(si, ti)
                    l_cls = src_crit(s_lg, sl) + tgt_crit(t_lg, tl)
                    k = int(0.5 * model.feature_extractor.out_dim)
                    _, idx = torch.topk(fw, k)
                    sf, tf_ = model.source_features_flat[:, idx], model.target_features_flat[:, idx]
                    l_align = F.mse_loss(sf.mean(0), tf_.mean(0)) + F.mse_loss(sf.std(0), tf_.std(0))
                    nw = fw[idx] / (torch.norm(fw[idx]) + 1e-6)
                    sim = nw.unsqueeze(1) @ nw.unsqueeze(0)
                    l_div = (sim.sum() - torch.trace(sim)) / (k * (k - 1) + 1e-6)
                    loss = l_cls + 0.1 * l_align + 0.01 * l_div
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
        sched.step()
        lg, y = get_logits(model, val_loader, device)
        if method == "cc":   # report: select on val acc + F1 with tau = 0.35
            m = metrics_from_logits(lg, y, tau=tau_select)
            score = m["accuracy"] + m["f1_macro"]
        else:                # report: select on raw val F1
            m = metrics_from_logits(lg, y)
            score = m["f1_macro"]
        if score > best:
            best, best_ep = score, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_w = model.last_feature_weights.detach().clone()
        print(f"  ep {ep:2d} val acc={m['accuracy']:.4f} f1={m['f1_macro']:.4f} mel={m['mel_found']}/{m['mel_total']} (best ep {best_ep})", flush=True)
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.last_feature_weights = best_w.to(device)
    return model, best_ep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["sup", "satda", "cc"])
    ap.add_argument("--n", choices=["50", "100", "200", "full"])
    ap.add_argument("--seed", type=int)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--prep", action="store_true", help="only build the image caches")
    args = ap.parse_args()

    os.makedirs(WORK_DIR, exist_ok=True)
    os.chdir(WORK_DIR)   # caches go to WORK_DIR/cache_low_resource (fresh, 345-split only)
    ds = U.build_datasets_with_nested_subsets(DATA_DIR)
    _, ev224 = U.build_transforms_b0(224)
    _, ev288 = U.build_transforms_b2_enhanced(288)

    if args.prep:
        for size, tf in ((224, ev224), (288, ev288)):
            for name in ("tgt_val", "tgt_test"):
                U.InMemorySkinDataset(ds[name], tf, 1, img_size=size, cache_name=name)
            for key in ("tgt_train_50", "tgt_train_100", "tgt_train_200", "tgt_train_full"):
                U.InMemorySkinDataset(ds[key], tf, 1, img_size=size, cache_name=f"tgt_train_{len(ds[key])}")
        print("caches ready", flush=True)
        return

    out_path = os.path.join(WORK_DIR, "results", f"{args.method}_N{args.n}_s{args.seed}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    U.set_seed(args.seed)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "3")))
    device = torch.device(args.device)
    tgt_items = ds["tgt_train_full" if args.n == "full" else f"tgt_train_{args.n}"]

    if args.method == "sup" and args.n != "full":
        variant, size, ev = "b0", 224, ev224
    else:
        variant, size, ev = "b2", 288, ev288
    val_loader = DataLoader(U.InMemorySkinDataset(ds["tgt_val"], ev, 1, img_size=size, cache_name="tgt_val"), batch_size=32, shuffle=False)
    test_loader = DataLoader(U.InMemorySkinDataset(ds["tgt_test"], ev, 1, img_size=size, cache_name="tgt_test"), batch_size=32, shuffle=False)

    print(f"=== {args.method} N={args.n} ({len(tgt_items)}) seed={args.seed} on {device} backbone={variant}@{size}", flush=True)
    t0 = time.time()
    if args.method == "sup":
        model, best_ep = train_sup(tgt_items, val_loader, device, variant)
    else:
        model, best_ep = train_da(args.method, ds["src_train"], tgt_items, val_loader, device)
    vl, vy = get_logits(model, val_loader, device)
    tl, ty = get_logits(model, test_loader, device)
    raw = metrics_from_logits(tl, ty)
    rep = metrics_from_logits(tl, ty, tau=0.35 if args.method == "cc" else 0.0)
    print(f"  TEST raw: {raw}", flush=True)
    print(f"  TEST report-style: {rep}", flush=True)
    with open(out_path, "w") as f:
        json.dump({
            "method": args.method, "n": args.n, "n_items": len(tgt_items), "seed": args.seed,
            "backbone": f"{variant}@{size}", "best_epoch": best_ep, "train_seconds": time.time() - t0,
            "val_logits": vl.tolist(), "val_labels": vy.tolist(),
            "test_logits": tl.tolist(), "test_labels": ty.tolist(),
        }, f)
    print(f"  saved {out_path} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
