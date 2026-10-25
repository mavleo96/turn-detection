"""Evaluate Pipecat's Smart Turn v3 (ONNX, pipecat-ai/smart-turn-v3) on our test split, as a baseline.

Smart Turn's own preprocessing (github.com/pipecat-ai/smart-turn, inference.py + audio_utils.py): the last 8 s,
zero-padded on the left, WhisperFeatureExtractor(chunk_length=8) with do_normalize=True; full band, no loudness
normalization. Its output is P(complete); pred at 0.5, Smart Turn's default. Smart Turn was trained on all of
smart-turn-data-v3.2-train (every language and source, so human_5 and rime_2 too) and never saw TTD. Writes the same
files as run_eval.py to --out (default results/eval/<model file name>/).

Run (from the repo root):  python scripts/run_eval_baseline.py [--model smart-turn-v3.2-cpu.onnx]
"""

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from huggingface_hub import hf_hub_download
from transformers import WhisperFeatureExtractor

from run_eval import markdown, summarize
from turn_detection.data import RESULTS_DIR, SAMPLE_RATE
from turn_detection.data.dataset import AUDIO, load_split

REPO = "pipecat-ai/smart-turn-v3"
WINDOW = 8 * SAMPLE_RATE
THRESHOLD = 0.5


class SmartTurnFeatures(torch.utils.data.Dataset):
    """Clips of a split -> Smart Turn's input features, (80, 800)."""

    def __init__(self, ds):
        self.audio = ds.select_columns(["audio"]).cast_column("audio", AUDIO)
        self.feature_extractor = WhisperFeatureExtractor(chunk_length=8)

    def __len__(self) -> int:
        return len(self.audio)

    def __getitem__(self, i: int) -> np.ndarray:
        x = self.audio[i]["audio"].get_all_samples().data[0].numpy()  # (T,)
        x = np.pad(x[-WINDOW:], (max(WINDOW - len(x), 0), 0))  # (128000,): the last 8 s, zeros on the left
        out = self.feature_extractor(
            x,
            sampling_rate=SAMPLE_RATE,
            return_tensors="np",
            padding="max_length",
            max_length=WINDOW,
            truncation=True,
            do_normalize=True,
        )
        return out.input_features[0]  # (80, 800)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate Smart Turn v3 on the test split")
    ap.add_argument("--model", default="smart-turn-v3.2-gpu.onnx", help=f"an ONNX file of {REPO}")
    ap.add_argument("--out", type=Path, default=None, help="default: results/eval/<model file name>")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=16, help="DataLoader workers for decoding + features")
    ap.add_argument("--threads", type=int, default=32, help="onnxruntime threads")
    args = ap.parse_args()

    out = args.out or RESULTS_DIR / "eval" / Path(args.model).stem
    out.mkdir(parents=True, exist_ok=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    session = ort.InferenceSession(hf_hub_download(REPO, args.model), options, providers=["CPUExecutionProvider"])

    test = load_split("test")
    loader = torch.utils.data.DataLoader(SmartTurnFeatures(test), batch_size=args.batch_size, num_workers=args.workers)
    probs = np.concatenate([session.run(None, {"input_features": f.numpy()})[0][:, 0] for f in loader])  # (N,)
    # Note: the output is named "logits", but Smart Turn's inference.py reads it as a probability; check that it is
    if probs.min() < 0 or probs.max() > 1:
        raise ValueError(f"outputs in [{probs.min()}, {probs.max()}]: not probabilities")

    preds = test.select_columns(["id", "dataset", "corpus"]).to_pandas().rename(columns={"dataset": "source"})
    preds["label"] = np.asarray(test["label"], dtype=int)
    preds["prob"] = probs
    preds["pred"] = (probs >= THRESHOLD).astype(int)
    metrics = summarize(preds, THRESHOLD)
    preds.to_parquet(out / "predictions.parquet", index=False)
    metrics.to_csv(out / "metrics.csv", index=False)
    (out / "metrics.md").write_text(markdown(metrics, f"Test metrics: {REPO}/{args.model} (threshold {THRESHOLD})"))

    print(metrics[metrics["group"] != "source"].round(4).to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
