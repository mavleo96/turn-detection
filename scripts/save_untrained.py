"""Save an untrained EoT model (pretrained Whisper encoder, random head) with its feature extractor, laid out
like a training run's `best/`: a stand-in for building and benchmarking serve/ before a model is trained.
Latency and throughput don't depend on the head's weights; its predictions are meaningless.

Run (from the repo root):  python scripts/save_untrained.py   # -> artifacts/models/untrained/
"""

import argparse
from pathlib import Path

import torch
from transformers import WhisperFeatureExtractor

from turn_detection.data.constants import RUNS_DIR, WINDOW_S
from turn_detection.model import EoTModel


def main() -> None:
    ap = argparse.ArgumentParser(description="Save an untrained EoT model for serving tests")
    ap.add_argument("--whisper", default="openai/whisper-tiny")
    ap.add_argument("--out", type=Path, default=RUNS_DIR.parent / "models" / "untrained")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    model = EoTModel.from_whisper(args.whisper)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.whisper, chunk_length=WINDOW_S)

    model.save_pretrained(args.out)
    feature_extractor.save_pretrained(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
