"""Evaluate a trained model on the test split and optionally time inference.

The test split of --datasets (data.dataset.load_split; default both): English smart-turn test (including the holdout
sources, never trained on) + ttd test (SpokenWOZ + Seamless). Preprocessing comes from the model's config, as in
training. Reports metrics at 0.5 and at the model's val-chosen threshold (config.threshold, never tuned on test),
overall and per source (`dataset`). Writes eval.json, predictions.parquet and eval_report.md next to the model
(or to --out).

Run (from the repo root):  python scripts/run_eval.py --model artifacts/runs/<run>/best [--datasets ttd] [--benchmark]
"""

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import Trainer, TrainingArguments, WhisperFeatureExtractor

from turn_detection.data import DATASETS, SAMPLE_RATE
from turn_detection.data.audio import pad_window, prepare_clip
from turn_detection.data.dataset import EoTCollator, EoTDataset, load_split
from turn_detection.model import EoTModel
from turn_detection.train.metrics import breakdown, eot_metrics

REPORT_METRICS = ["accuracy", "f1", "fpr", "fnr", "roc_auc", "pr_auc"]


def evaluate_test(model: EoTModel, feature_extractor: WhisperFeatureExtractor, test, args) -> tuple[dict, pd.DataFrame]:
    """Metrics on `test` (load_split("test")), overall and per source, and the per-clip predictions."""
    c = model.config
    dataset = EoTDataset(test, narrowband=c.narrowband, loudness_dbfs=c.loudness_dbfs, window_s=c.window_s)
    with tempfile.TemporaryDirectory() as tmp:
        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=tmp,
                per_device_eval_batch_size=args.batch_size,
                dataloader_num_workers=args.workers,
                bf16=args.bf16,
                use_cpu=args.device == "cpu",
                report_to="none",
                remove_unused_columns=False,
            ),
            data_collator=EoTCollator(feature_extractor),
        )
        logits = trainer.predict(dataset).predictions  # (N,)
        probs = 1 / (1 + np.exp(-np.clip(logits, -80, 80)))  # clipped so exp can't overflow in float32
    threshold = model.config.threshold
    preds = test.select_columns(["id", "dataset"]).to_pandas().assign(label=dataset.labels.astype(int), prob=probs)

    result = {
        "n": len(preds),
        "at_0.5": eot_metrics(dataset.labels, probs, 0.5),
        "at_threshold": eot_metrics(dataset.labels, probs, threshold),
        "by_dataset": breakdown(preds, "dataset", threshold).to_dict("records"),
    }
    return result, preds


@torch.no_grad()
def benchmark(model: EoTModel, feature_extractor: WhisperFeatureExtractor, device: str, runs: int = 100) -> dict:
    """Latency of one decision (batch 1), as a voice agent would call it at every pause: preprocessing and
    feature extraction of the last `window_s` seconds (CPU) + the model on `device`."""
    model = model.to(device).eval()
    config = model.config
    audio = np.zeros(config.window_s * SAMPLE_RATE, dtype=np.float32)
    sync = torch.cuda.synchronize if device.startswith("cuda") else (lambda: None)

    def decide():
        x = prepare_clip(
            audio, window_s=config.window_s, use_narrowband=config.narrowband, loudness_dbfs=config.loudness_dbfs
        )
        window = pad_window(x, config.window_s)
        features = feature_extractor(window, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features
        model(input_features=features.to(device))
        sync()

    for _ in range(10):
        decide()
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        decide()
        times.append((time.perf_counter() - t) * 1000)
    return {
        "device": device,
        "runs": runs,
        "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
    }


def _fmt(v) -> str:
    return "–" if v is None else f"{v:.4f}" if isinstance(v, float) else str(v)


def report(result: dict, threshold: float, bench: list[dict]) -> str:
    lines = ["# EoT evaluation\n", f"Val-chosen threshold {threshold:.4f}.\n", f"\n## Test ({result['n']} clips)\n"]
    lines.append("| threshold | " + " | ".join(REPORT_METRICS) + " |")
    lines.append("|" + "---|" * (len(REPORT_METRICS) + 1))
    for at in ("at_0.5", "at_threshold"):
        m = result[at]
        lines.append(f"| {m['threshold']:.4f} | " + " | ".join(_fmt(m.get(k)) for k in REPORT_METRICS) + " |")

    df = pd.DataFrame(result["by_dataset"])
    lines.append(f"\n**by source** (threshold {threshold:.4f})\n")
    lines.append("| " + " | ".join(df.columns) + " |")
    lines.append("|" + "---|" * len(df.columns))
    lines += ["| " + " | ".join(_fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]

    if bench:
        lines.append("\n**Latency per decision (batch 1, feature extraction + model)**\n")
        lines += [f"- {b['device']}: p50 {b['p50_ms']:.1f} ms, p95 {b['p95_ms']:.1f} ms" for b in bench]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate an EoT model on the test split")
    ap.add_argument(
        "--model", type=Path, required=True, help="a training output with model + feature extractor, e.g. <run>/best"
    )
    ap.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS), help="test data")
    ap.add_argument("--out", type=Path, default=None, help="default: the model directory")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--benchmark", action="store_true", help="also time batch-1 inference on --device and CPU")
    args = ap.parse_args()
    args.bf16 = args.device != "cpu"

    out = args.out or args.model
    out.mkdir(parents=True, exist_ok=True)
    model = EoTModel.from_pretrained(args.model)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model)

    result, preds = evaluate_test(model, feature_extractor, load_split("test", args.datasets), args)
    (out / "eval.json").write_text(json.dumps(result, indent=1, default=float))
    preds.to_parquet(out / "predictions.parquet", index=False)

    m = result["at_threshold"]
    print("test: " + " | ".join(f"{k} {_fmt(m.get(k))}" for k in REPORT_METRICS))

    bench = []
    if args.benchmark:
        bench.append(benchmark(model, feature_extractor, args.device))
        if args.device != "cpu":
            bench.append(benchmark(model, feature_extractor, "cpu", runs=30))
    (out / "eval_report.md").write_text(report(result, model.config.threshold, bench))
    print(f"wrote {out / 'eval_report.md'}")


if __name__ == "__main__":
    main()
