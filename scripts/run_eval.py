"""Evaluate one of our checkpoints on the test split (smart-turn English + TTD, data.dataset.load_split).

Preprocessing (window, band limit, loudness) comes from the model's config and the feature extractor from the
checkpoint, as in training; predictions use the model's validation threshold (config.threshold). Writes to --out
(default results/eval/<run>/<checkpoint>/, e.g. results/eval/both-8s-5ep-<timestamp>/best/):
  predictions.parquet  one row per test clip: id, source, corpus, label, prob, pred
  metrics.csv          per group (all, corpus, source): n, complete share, accuracy, f1, roc_auc, pr_auc, fnr, fpr
  metrics.md           the same, readable

Run (from the repo root):  python scripts/run_eval.py artifacts/runs/<run>/best
"""

import argparse
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import Trainer, TrainingArguments, WhisperFeatureExtractor

from turn_detection.data import RESULTS_DIR
from turn_detection.data.dataset import EoTCollator, EoTDataset, load_split
from turn_detection.model import EoTModel
from turn_detection.train.metrics import METRICS, breakdown

GROUPS = ["all", "corpus", "source"]


def predict(
    model: EoTModel, feature_extractor: WhisperFeatureExtractor, test, batch_size: int, workers: int, use_cpu: bool
) -> pd.DataFrame:
    """Per-clip predictions on `test` (load_split("test")): id, source, corpus, label, prob, pred."""
    config = model.config
    dataset = EoTDataset(
        test, narrowband=config.narrowband, loudness_dbfs=config.loudness_dbfs, window_s=config.window_s
    )
    with tempfile.TemporaryDirectory() as tmp:
        args = TrainingArguments(
            output_dir=tmp,
            per_device_eval_batch_size=batch_size,
            dataloader_num_workers=workers,
            bf16=torch.cuda.is_available() and not use_cpu,
            use_cpu=use_cpu,
            report_to="none",
            remove_unused_columns=False,
        )
        trainer = Trainer(model=model, args=args, data_collator=EoTCollator(feature_extractor))
        # Note: predict keeps the dataset's order, so the logits line up with the rows of `test`
        logits = trainer.predict(dataset).predictions  # (N,)

    preds = test.select_columns(["id", "dataset", "corpus"]).to_pandas().rename(columns={"dataset": "source"})
    preds["label"] = dataset.labels.astype(int)
    preds["prob"] = 1 / (1 + np.exp(-np.clip(logits, -80, 80)))  # clipped so exp can't overflow in float32
    preds["pred"] = (preds["prob"] >= config.threshold).astype(int)
    return preds


def summarize(preds: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Metrics for all clips, per corpus and per source: group, name, n, complete share, METRICS."""
    tables = []
    for group in GROUPS:
        table = breakdown(preds.assign(all="all"), group, threshold).rename(columns={group: "name"})
        tables.append(table.assign(group=group))
    return pd.concat(tables, ignore_index=True)[["group", "name", "n", "complete", *METRICS]]


def markdown(metrics: pd.DataFrame, title: str) -> str:
    """The metrics as one markdown table per group."""
    lines = [f"# {title}\n"]
    for group in GROUPS:
        table = metrics[metrics["group"] == group].drop(columns="group")
        lines += [f"\n## {group}\n", "| " + " | ".join(table.columns) + " |", "|" + "---|" * len(table.columns)]
        for row in table.itertuples(index=False):
            lines.append("| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in row) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a checkpoint on the test split")
    ap.add_argument("model", type=Path, help="a checkpoint with model + feature extractor, e.g. <run>/best")
    ap.add_argument("--out", type=Path, default=None, help="default: results/eval/<run>/<checkpoint>")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--cpu", action="store_true", help="run on the CPU even if a GPU is available")
    args = ap.parse_args()

    out = args.out or RESULTS_DIR / "eval" / args.model.resolve().parent.name / args.model.resolve().name
    out.mkdir(parents=True, exist_ok=True)
    model = EoTModel.from_pretrained(args.model)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model)
    threshold = model.config.threshold

    preds = predict(model, feature_extractor, load_split("test"), args.batch_size, args.workers, args.cpu)
    metrics = summarize(preds, threshold)
    preds.to_parquet(out / "predictions.parquet", index=False)
    metrics.to_csv(out / "metrics.csv", index=False)
    title = f"Test metrics: {args.model} (window {model.config.window_s} s, threshold {threshold:.4f})"
    (out / "metrics.md").write_text(markdown(metrics, title))

    print(metrics[metrics["group"] != "source"].round(4).to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
