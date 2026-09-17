from typing import Dict, Any, List
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report


def compute_classification_metrics(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str] = ("BCC", "MEL", "NV", "AKIEC", "BKL")
) -> Dict[str, Any]:
    """
    Computes rigorous multiclass benchmark metrics:
    - Overall Accuracy (%)
    - Macro-averaged F1 Score
    - Per-class Sensitivity / Recall (specifically tracking Melanoma TP/FN counts)
    - Full Confusion Matrix
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = float(accuracy_score(y_true, y_pred)) * 100.0
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))

    # Per-class recall
    recalls = {}
    tp_counts = {}
    total_counts = {}
    for i, cname in enumerate(class_names):
        total_c = int((y_true == i).sum())
        tp_c = int(((y_true == i) & (y_pred == i)).sum())
        rec = (tp_c / total_c * 100.0) if total_c > 0 else 0.0
        recalls[cname] = rec
        tp_counts[cname] = tp_c
        total_counts[cname] = total_c

    report = classification_report(
        y_true, y_pred, target_names=class_names, digits=4, zero_division=0
    )

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "class_recall": recalls,
        "class_tp": tp_counts,
        "class_total": total_counts,
        "melanoma_recall": recalls.get("MEL", 0.0),
        "melanoma_tp": tp_counts.get("MEL", 0),
        "melanoma_total": total_counts.get("MEL", 0),
        "confusion_matrix": cm,
        "report": report,
    }
