"""Evaluate a trained model on the English test split (16 kHz, as recorded) and optionally time inference.

Reports metrics at 0.5 and at the model's val-chosen threshold (config.threshold, never tuned on
test), broken down by source and by synthetic flag. Writes eval_test.json, predictions_test.parquet
and eval_report.md next to the model (or to --out).

(Named run_eval.py, not evaluate.py: with scripts/ on the path, evaluate.py would shadow the
Hugging Face `evaluate` package used for the metrics.)

Run (from the repo root):  python scripts/run_eval.py --model artifacts/runs/<run>/best [--benchmark]
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

from data.constants import MANIFESTS_DIR, SAMPLE_RATE, WINDOW_S
from data.dataset import EoTCollator, EoTDataset, load_split
from model import EoTModel
from train.metrics import breakdown, eot_metrics
from train.trainer import predict_probs

REPORT_METRICS = ["accuracy", "precision", "recall", "f1", "false_endpoint_rate", "roc_auc"]


def evaluate_test(
    model: EoTModel, feature_extractor: WhisperFeatureExtractor, source, manifest: pd.DataFrame, args
) -> tuple[dict, pd.DataFrame]:
    """Metrics on the test clips; `source` is the HF test split, `manifest` the clips (scripts/split.py)."""
    dataset = EoTDataset(source, manifest["id"], pad=model.config.padding)
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
        probs = predict_probs(trainer, dataset)
    threshold = model.config.threshold
    preds = pd.DataFrame(
        {
            "id": manifest["id"].to_numpy(),
            "dataset": manifest["dataset"].to_numpy(),
            "synthetic": manifest["synthetic"].to_numpy(),
            "label": dataset.labels.astype(int),
            "prob": probs,
        }
    )
    result = {
        "n": len(preds),
        "at_0.5": eot_metrics(dataset.labels, probs, 0.5),
        "at_threshold": eot_metrics(dataset.labels, probs, threshold),
        "by_source": breakdown(preds, "dataset", threshold).to_dict("records"),
        "by_synthetic": breakdown(preds, "synthetic", threshold).to_dict("records"),
    }
    return result, preds


@torch.no_grad()
def benchmark(model: EoTModel, feature_extractor: WhisperFeatureExtractor, device: str, runs: int = 100) -> dict:
    """Latency of one decision (batch 1), as a voice agent would call it at every pause: feature
    extraction of the last 8 s (CPU) + the model on `device`."""
    model = model.to(device).eval()
    audio = np.zeros(WINDOW_S * SAMPLE_RATE, dtype=np.float32)
    sync = torch.cuda.synchronize if device.startswith("cuda") else (lambda: None)

    def decide():
        features = feature_extractor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features.to(device)
        model(input_features=features)
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
    lines = [
        f"# EoT evaluation (English test split, {result['n']} clips)\n",
        f"Val-chosen threshold {threshold:.4f}.\n",
    ]
    lines.append("| threshold | " + " | ".join(REPORT_METRICS) + " |")
    lines.append("|" + "---|" * (len(REPORT_METRICS) + 1))
    for at in ("at_0.5", "at_threshold"):
        m = result[at]
        lines.append(f"| {m['threshold']:.4f} | " + " | ".join(_fmt(m.get(k)) for k in REPORT_METRICS) + " |")
    for key in ("by_source", "by_synthetic"):
        df = pd.DataFrame(result[key])
        lines.append(f"\n**{key.replace('_', ' ')}** (threshold {threshold:.4f})\n")
        lines.append("| " + " | ".join(df.columns) + " |")
        lines.append("|" + "---|" * len(df.columns))
        lines += ["| " + " | ".join(_fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    if bench:
        lines.append("\n**Latency per decision (batch 1, feature extraction + model)**\n")
        lines += [f"- {b['device']}: p50 {b['p50_ms']:.1f} ms, p95 {b['p95_ms']:.1f} ms" for b in bench]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate an EoT model on the English test split")
    ap.add_argument(
        "--model", type=Path, required=True, help="a training output with model + feature extractor, e.g. <run>/best"
    )
    ap.add_argument("--manifest", type=Path, default=MANIFESTS_DIR / "test.parquet")
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
    result, preds = evaluate_test(model, feature_extractor, load_split("test"), pd.read_parquet(args.manifest), args)
    (out / "eval_test.json").write_text(json.dumps(result, indent=1, default=float))
    preds.to_parquet(out / "predictions_test.parquet", index=False)
    m = result["at_threshold"]
    print(f"test: roc_auc {_fmt(m.get('roc_auc'))} | f1 {m['f1']:.4f} | false endpoints {m['false_endpoint_rate']:.4f}")
    bench = []
    if args.benchmark:
        bench.append(benchmark(model, feature_extractor, args.device))
        if args.device != "cpu":
            bench.append(benchmark(model, feature_extractor, "cpu", runs=30))
    (out / "eval_report.md").write_text(report(result, model.config.threshold, bench))
    print(f"wrote {out / 'eval_report.md'}")


if __name__ == "__main__":
    main()
