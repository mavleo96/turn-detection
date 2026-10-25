import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_recall_curve, roc_auc_score

METRICS = ["accuracy", "f1", "roc_auc", "pr_auc", "fnr", "fpr"]


def eot_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict:
    """accuracy, f1, fpr, fnr and (when both classes occur) roc_auc and pr_auc."""
    labels = np.asarray(labels).astype(int)
    preds = (np.asarray(probs) >= threshold).astype(int)
    complete, incomplete = labels == 1, labels == 0

    m = {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, zero_division=0),
        "fpr": preds[incomplete].sum() / max(incomplete.sum(), 1),
        "fnr": (1 - preds[complete]).sum() / max(complete.sum(), 1),
    }

    if len(np.unique(labels)) == 2:  # undefined with one class
        m["roc_auc"] = roc_auc_score(labels, probs)
        m["pr_auc"] = average_precision_score(labels, probs)

    return {"threshold": float(threshold), **{k: float(v) for k, v in m.items()}}


def best_f1_threshold(labels: np.ndarray, probs: np.ndarray) -> float:
    """Threshold maximizing F1 (chosen on val, then fixed for test)."""
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    # Note: the last (precision 1, recall 0) point has no threshold
    best = int(np.argmax(f1[:-1]))
    return float(thresholds[best])


def breakdown(df: pd.DataFrame, by: str, threshold: float) -> pd.DataFrame:
    """Metrics per group of `df` (columns: label, prob, and `by`): n, complete share and METRICS."""

    rows = []
    for key, g in df.groupby(by, sort=True):
        m = eot_metrics(g["label"].to_numpy(), g["prob"].to_numpy(), threshold)
        row = {by: key, "n": len(g), "complete": g["label"].mean()}
        # Note: roc_auc and pr_auc are missing for single-class groups
        for k in METRICS:
            row[k] = m.get(k)
        rows.append(row)

    return pd.DataFrame(rows)
