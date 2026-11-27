#!/usr/bin/env python3
"""Summarise the five-seed study. Rules fixed BEFORE looking at any result:

report  : calibration as in the report (CC-SAT-DA tau = 0.35; SAT-DA and
          target-only uncalibrated).
safety  : the SAME validation-only procedure for every method. Search
          tau in {0, 0.05, ..., 0.5} (prior shift) and a melanoma offset
          delta in {0, 0.1, ..., 3.0}; keep settings whose VALIDATION melanoma
          recall >= 0.75 (6 of 8); among them take the highest validation
          Macro F1 (ties: higher val accuracy, then smaller delta). If none
          reaches 0.75, take the highest val melanoma recall, then Macro F1.
          The test set is used only to report the chosen setting.
All seeds are reported; none is dropped.
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_study import metrics_from_logits  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
ROOT = os.environ.get("CCSATDA_STUDY_DIR", os.path.join(REPO, "results", "seed_study_345"))
sys.path.insert(0, HERE)
TAUS = np.round(np.arange(0, 0.501, 0.05), 2)
DELTAS = np.round(np.arange(0, 3.001, 0.1), 2)
MIN_MEL_RECALL = 0.75
NAMES = {"sup": "Target-only supervised", "satda": "SAT-DA", "cc": "CC-SAT-DA"}
N_ORDER = ["50", "100", "200", "full"]


def select_safety(vl, vy):
    best = None
    for t in TAUS:
        for d in DELTAS:
            m = metrics_from_logits(vl, vy, tau=t, delta=d)
            feas = m["melanoma_recall"] >= MIN_MEL_RECALL - 1e-9
            key = (feas, m["f1_macro"] if feas else m["melanoma_recall"], m["f1_macro"] if not feas else m["accuracy"], -d)
            if best is None or key > best[0]:
                best = (key, t, d)
    return best[1], best[2]


def main():
    rows = []
    for f in sorted(glob.glob(os.path.join(ROOT, "results", "*.json"))):
        r = json.load(open(f))
        vl, vy = np.array(r["val_logits"]), np.array(r["val_labels"])
        tl, ty = np.array(r["test_logits"]), np.array(r["test_labels"])
        rep = metrics_from_logits(tl, ty, tau=0.35 if r["method"] == "cc" else 0.0)
        t, d = select_safety(vl, vy)
        saf = metrics_from_logits(tl, ty, tau=t, delta=d)
        rows.append({"method": r["method"], "n": r["n"], "seed": r["seed"], "report": rep, "safety": saf, "tau": t, "delta": d,
                     "minutes": r["train_seconds"] / 60})

    with open(os.path.join(ROOT, "per_seed.json"), "w") as f:
        json.dump(rows, f, indent=1)

    lines = ["# Five-seed study on the 345-photo test split", "",
             f"Runs found: {len(rows)}. Mean ± std over seeds; MEL = melanomas found of 8; MEL prec = share of melanoma calls that are correct.", ""]
    for cal in ("report", "safety"):
        lines += [f"## Calibration: {cal}", "",
                  "| N | Method | seeds | Accuracy % | Macro F1 | AUROC | MEL found | MEL prec % |",
                  "|---|---|---|---|---|---|---|---|"]
        for n in N_ORDER:
            for m in ("sup", "satda", "cc"):
                sel = [x for x in rows if x["n"] == n and x["method"] == m]
                if not sel:
                    continue
                g = lambda k: np.array([x[cal][k] for x in sel], dtype=float)  # noqa: E731
                prec = np.array([x[cal]["mel_found"] / x[cal]["mel_pred"] if x[cal]["mel_pred"] else 0.0 for x in sel])
                lines.append(f"| {n} | {NAMES[m]} | {len(sel)} | {100*g('accuracy').mean():.1f} ± {100*g('accuracy').std():.1f} | "
                             f"{g('f1_macro').mean():.3f} ± {g('f1_macro').std():.3f} | {g('auroc_macro_ovr').mean():.3f} ± {g('auroc_macro_ovr').std():.3f} | "
                             f"{g('mel_found').mean():.1f} ± {g('mel_found').std():.1f} | {100*prec.mean():.0f} |")
        lines.append("")
    lines += ["## Per-seed results", "", "| N | Method | seed | report: acc / F1 / MEL | safety: acc / F1 / MEL | chosen tau, delta |", "|---|---|---|---|---|---|"]
    for x in sorted(rows, key=lambda x: (N_ORDER.index(x["n"]), x["method"], x["seed"])):
        a, b = x["report"], x["safety"]
        lines.append(f"| {x['n']} | {NAMES[x['method']]} | {x['seed']} | {100*a['accuracy']:.1f} / {a['f1_macro']:.3f} / {a['mel_found']} | "
                     f"{100*b['accuracy']:.1f} / {b['f1_macro']:.3f} / {b['mel_found']} | {x['tau']}, {x['delta']} |")
    open(os.path.join(ROOT, "SUMMARY.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
