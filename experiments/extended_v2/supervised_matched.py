#!/usr/bin/env python3
"""
Pure target-supervised baseline matched EXACTLY to the CC-SAT-DA v2 "final" run
(results/cc_satda_v2, Sep 12, suite 3_multiseed_benchmark):

  * same data: build_datasets_v2(seed=42) splits (no-SCC mapping, 316 test / 316 val);
    the v2 run loaded the seed-42 target subsets from disk cache for EVERY seed,
    so we deliberately use the seed-42 subsets for every seed here too.
  * same backbone / input: EfficientNet-B2 @ 288, same LabelClassifier head (dropout 0.3)
  * same target augmentation (build_transforms_b2_enhanced)
  * same optimisation budget: 15 epochs x 60 steps, batch 16, AdamW 1e-4 (backbone) /
    5e-4 (head), cosine schedule, AMP, grad-clip 5
  * same model selection: best validation macro-F1 checked every epoch
  * plain cross-entropy (pure supervised), no tau / prior adjustment
Images are kept in memory only (no disk cache).
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
import numpy as np, torch, torch.nn as nn
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
WORKERS = int(os.environ.get("TGT_WORKERS", "4"))
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from train_cc_satda_v2 import build_datasets_v2, evaluate_metrics_from_logits
from run_ultra_low_resource_study import (
    PureSupervisedModel, build_transforms_b2_enhanced, set_seed, infinite_loader,
)

DATA = os.environ.get("CCSATDA_DATA", "data")


class MemDS(Dataset):
    def __init__(self, items, tf):
        self.data = [(Image.open(p).convert("RGB").resize((288, 288)), l) for p, l in items]
        self.tf = tf
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        im, l = self.data[i]
        return self.tf(im), l


@torch.no_grad()
def logits_of(model, loader, dev):
    model.eval(); L, Y = [], []
    for x, y in loader:
        with torch.amp.autocast("cuda"):
            L.append(model(x.to(dev)).float().cpu())
        Y.append(y)
    return torch.cat(L), torch.cat(Y).numpy()


def run(n, seed, dev, ds, tf_tr, val_loader, test_loader, epochs=15, steps=60):
    set_seed(seed)
    items = ds["tgt_train_full"] if n == "full" else ds[f"tgt_train_{n}"]
    tr = MemDS(items, tf_tr)
    bs = min(16, len(items))
    g = torch.Generator(); g.manual_seed(seed)
    it = infinite_loader(DataLoader(tr, batch_size=bs, shuffle=True, drop_last=True, num_workers=WORKERS, persistent_workers=WORKERS > 0, generator=g))
    model = PureSupervisedModel(variant="b2", num_classes=5, dropout=0.3).to(dev)
    opt = torch.optim.AdamW([
        {"params": model.feature_extractor.parameters(), "lr": 1e-4, "weight_decay": 1e-3},
        {"params": model.classifier.parameters(), "lr": 5e-4, "weight_decay": 5e-4},
    ])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda"); ce = nn.CrossEntropyLoss()
    best, best_state, best_ep = -1, None, -1
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        for _ in range(steps):
            x, y = next(it); x, y = x.to(dev), y.to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                loss = ce(model(x), y)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update()
        sch.step()
        vl, vy = logits_of(model, val_loader, dev)
        vf1 = evaluate_metrics_from_logits(vl, vy, None, None, tau=0.0)["f1_macro"]
        if vf1 > best:
            best, best_ep = vf1, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    tl, ty = logits_of(model, test_loader, dev)
    m = evaluate_metrics_from_logits(tl, ty, None, None, tau=0.0)
    m.update(setting_N=str(n), seed=seed, target_samples=len(items), val_best_f1=best,
             best_epoch=best_ep, train_time_sec=time.time() - t0)
    print(f"[N={n} seed={seed}] acc={m['accuracy']:.4f} f1={m['f1_macro']:.4f} "
          f"auroc={m['auroc_macro_ovr']:.4f} mel={m['melanoma_recall']:.3f} bcc={m['bcc_recall']:.3f}", flush=True)
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 1011])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    n = a.n if a.n == "full" else int(a.n)
    dev = torch.device(a.device)
    ds = build_datasets_v2(DATA, seed=42)
    tf_tr, tf_ev = build_transforms_b2_enhanced(288)
    val_loader = DataLoader(MemDS(ds["tgt_val"], tf_ev), batch_size=32)
    test_loader = DataLoader(MemDS(ds["tgt_test"], tf_ev), batch_size=32)
    print(f"N={n}: train={len(ds['tgt_train_full'] if n=='full' else ds[f'tgt_train_{n}'])} "
          f"val={len(ds['tgt_val'])} test={len(ds['tgt_test'])}", flush=True)
    res = json.load(open(a.out)) if os.path.exists(a.out) else []
    done = {r["seed"] for r in res}
    for s in a.seeds:
        if s in done:
            continue
        res.append(run(n, s, dev, ds, tf_tr, val_loader, test_loader))
        json.dump(res, open(a.out, "w"), indent=2)
