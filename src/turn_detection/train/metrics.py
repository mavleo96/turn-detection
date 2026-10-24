"""Metrics for P(turn complete) with scikit-learn. Positive class = turn complete.

`false_endpoint_rate` is the share of incomplete turns predicted complete: the agent would interrupt
the user, which is the costly error for a voice agent.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_curve, precision_recall_fscore_support, roc_auc_score
from transformers import EvalPrediction


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.asarray(logits, dtype=np.float64)))


def eot_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict:
    """accuracy, precision, recall, f1, false_endpoint_rate and (when both classes occur) roc_auc."""
    labels = np.asarray(labels).astype(int)
    preds = (np.asarray(probs) >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    m = {"accuracy": accuracy_score(labels, preds), "precision": precision, "recall": recall, "f1": f1}

    incomplete = labels == 0
    m["false_endpoint_rate"] = preds[incomplete].sum() / max(incomplete.sum(), 1)

    if len(np.unique(labels)) == 2:  # ROC AUC is undefined with one class
        m["roc_auc"] = roc_auc_score(labels, probs)

    return {"threshold": float(threshold), **{k: float(v) for k, v in m.items()}}


def compute_metrics(eval_pred: EvalPrediction) -> dict:
    """Trainer hook: metrics at threshold 0.5 from the model's logits."""
    return eot_metrics(eval_pred.label_ids, sigmoid(eval_pred.predictions))


def best_f1_threshold(labels: np.ndarray, probs: np.ndarray) -> float:
    """Threshold maximizing F1 (chosen on val, then fixed for test)."""
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    best = int(np.argmax(f1[:-1]))  # the last (precision 1, recall 0) point has no threshold
    return float(thresholds[best])


def breakdown(df: pd.DataFrame, by: str, threshold: float) -> pd.DataFrame:
    """Metrics per group of `df` (columns: label, prob, and `by`)."""
    columns = ("accuracy", "f1", "false_endpoint_rate", "roc_auc")

    rows = []
    for key, g in df.groupby(by, sort=True):
        m = eot_metrics(g["label"].to_numpy(), g["prob"].to_numpy(), threshold)
        row = {by: key, "n": len(g)}
        for k in columns:
            row[k] = m.get(k)  # roc_auc is missing for single-class groups
        rows.append(row)

    return pd.DataFrame(rows)
