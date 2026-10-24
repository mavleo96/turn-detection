"""Metrics for P(turn complete) with Hugging Face `evaluate`. Positive class = turn complete.

`false_endpoint_rate` is the share of incomplete turns predicted complete: the agent would interrupt
the user, which is the costly error for a voice agent.
"""

from functools import cache

import evaluate
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
from transformers import EvalPrediction


@cache
def _metrics():
    return (
        evaluate.combine(["accuracy", "precision", "recall", "f1"]),
        evaluate.load("roc_auc"),
        evaluate.load("confusion_matrix"),
    )


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.asarray(logits, dtype=np.float64)))


def eot_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict:
    """accuracy, precision, recall, f1, false_endpoint_rate and (when both classes occur) roc_auc."""
    labels = np.asarray(labels).astype(int)
    preds = (np.asarray(probs) >= threshold).astype(int)
    classification, roc_auc, confusion = _metrics()

    m = classification.compute(predictions=preds, references=labels)

    matrix = confusion.compute(predictions=preds, references=labels, labels=[0, 1])["confusion_matrix"]
    tn, fp, _, _ = np.ravel(matrix)
    m["false_endpoint_rate"] = fp / max(fp + tn, 1)

    if len(np.unique(labels)) == 2:  # ROC AUC is undefined with one class
        m["roc_auc"] = roc_auc.compute(prediction_scores=probs, references=labels)["roc_auc"]

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
