"""Evaluate a trained model on the test sets and optionally time inference.

Test sets (--datasets, default both): the ttd test split (SpokenWOZ + Seamless) and the English smart-turn test
split (the clips of scripts/split.py's test manifest). Preprocessing comes from the model's config, as in
training. Reports metrics at 0.5 and at the model's val-chosen threshold (config.threshold, never tuned on
test), broken down by source (and, for smart-turn, by synthetic flag). Writes eval_<dataset>.json,
predictions_<dataset>.parquet and eval_report.md next to the model (or to --out).

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

from turn_detection.data.audio import pad_window, prepare_clip
from turn_detection.data.constants import DATASETS, MANIFESTS_DIR, SAMPLE_RATE, WINDOW_S
from turn_detection.data.dataset import EoTCollator, EoTDataset, load_split
from turn_detection.model import EoTModel
from turn_detection.train.metrics import breakdown, eot_metrics, sigmoid

REPORT_METRICS = ["accuracy", "precision", "recall", "f1", "false_endpoint_rate", "roc_auc"]
BREAKDOWNS = {"ttd": ["source"], "smart-turn": ["source", "synthetic"]}  # per-group metrics in the report


def load_test_clips(name: str, source, manifest_path: Path) -> pd.DataFrame:
    """The test clips of dataset `name`: id, label, source (+ synthetic for smart-turn)."""
    if name == "smart-turn":  # the English clips of scripts/split.py's test manifest
        manifest = pd.read_parquet(manifest_path)
        clips = {"id": manifest["id"], "source": manifest["dataset"], "synthetic": manifest["synthetic"]}
        return pd.DataFrame(clips)

    spec = DATASETS[name]
    meta = source.select_columns(["id", spec["source"]]).to_pandas()
    return meta.rename(columns={spec["source"]: "source"})


def evaluate_test(
    model: EoTModel,
    feature_extractor: WhisperFeatureExtractor,
    source,
    clips: pd.DataFrame,
    args,
    name: str = "ttd",
) -> tuple[dict, pd.DataFrame]:
    """Metrics on `clips` (load_test_clips) of `source`, the HF test split of dataset `name`."""
    dataset = EoTDataset(
        source,
        clips["id"],
        narrowband=model.config.narrowband,
        loudness_dbfs=model.config.loudness_dbfs,
        label_column=DATASETS[name]["label"],
    )
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
        probs = sigmoid(trainer.predict(dataset).predictions)
    threshold = model.config.threshold
    preds = clips.reset_index(drop=True).assign(label=dataset.labels.astype(int), prob=probs)

    result = {
        "dataset": name,
        "n": len(preds),
        "at_0.5": eot_metrics(dataset.labels, probs, 0.5),
        "at_threshold": eot_metrics(dataset.labels, probs, threshold),
    }
    for column in BREAKDOWNS[name]:
        result[f"by_{column}"] = breakdown(preds, column, threshold).to_dict("records")

    return result, preds


@torch.no_grad()
def benchmark(model: EoTModel, feature_extractor: WhisperFeatureExtractor, device: str, runs: int = 100) -> dict:
    """Latency of one decision (batch 1), as a voice agent would call it at every pause: preprocessing and
    feature extraction of the last 8 s (CPU) + the model on `device`."""
    model = model.to(device).eval()
    audio = np.zeros(WINDOW_S * SAMPLE_RATE, dtype=np.float32)
    sync = torch.cuda.synchronize if device.startswith("cuda") else (lambda: None)

    def decide():
        x = prepare_clip(audio, model.config.narrowband, model.config.loudness_dbfs)
        window = pad_window(x)
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


def report(results: list[dict], threshold: float, bench: list[dict]) -> str:
    lines = ["# EoT evaluation\n", f"Val-chosen threshold {threshold:.4f}.\n"]

    for result in results:
        lines.append(f"\n## {result['dataset']} test ({result['n']} clips)\n")
        lines.append("| threshold | " + " | ".join(REPORT_METRICS) + " |")
        lines.append("|" + "---|" * (len(REPORT_METRICS) + 1))
        for at in ("at_0.5", "at_threshold"):
            m = result[at]
            lines.append(f"| {m['threshold']:.4f} | " + " | ".join(_fmt(m.get(k)) for k in REPORT_METRICS) + " |")

        for column in BREAKDOWNS[result["dataset"]]:
            df = pd.DataFrame(result[f"by_{column}"])
            lines.append(f"\n**by {column}** (threshold {threshold:.4f})\n")
            lines.append("| " + " | ".join(df.columns) + " |")
            lines.append("|" + "---|" * len(df.columns))
            lines += ["| " + " | ".join(_fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]

    if bench:
        lines.append("\n**Latency per decision (batch 1, feature extraction + model)**\n")
        lines += [f"- {b['device']}: p50 {b['p50_ms']:.1f} ms, p95 {b['p95_ms']:.1f} ms" for b in bench]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate an EoT model on the test sets")
    ap.add_argument(
        "--model", type=Path, required=True, help="a training output with model + feature extractor, e.g. <run>/best"
    )
    ap.add_argument("--datasets", nargs="+", choices=sorted(BREAKDOWNS), default=["ttd", "smart-turn"])
    ap.add_argument("--manifest", type=Path, default=MANIFESTS_DIR / "test.parquet", help="smart-turn test clips")
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

    results = []
    for name in args.datasets:
        source = load_split("test", dataset=name)
        clips = load_test_clips(name, source, args.manifest)
        result, preds = evaluate_test(model, feature_extractor, source, clips, args, name=name)

        (out / f"eval_{name}.json").write_text(json.dumps(result, indent=1, default=float))
        preds.to_parquet(out / f"predictions_{name}.parquet", index=False)
        results.append(result)

        m = result["at_threshold"]
        scores = f"roc_auc {_fmt(m.get('roc_auc'))} | f1 {m['f1']:.4f} | false endpoints {m['false_endpoint_rate']:.4f}"
        print(f"{name} test: {scores}")

    bench = []
    if args.benchmark:
        bench.append(benchmark(model, feature_extractor, args.device))
        if args.device != "cpu":
            bench.append(benchmark(model, feature_extractor, "cpu", runs=30))
    (out / "eval_report.md").write_text(report(results, model.config.threshold, bench))
    print(f"wrote {out / 'eval_report.md'}")


if __name__ == "__main__":
    main()
