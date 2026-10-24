"""Inference wrapper: raw audio of any length/rate -> P(turn complete), for a voice-agent pipeline.
Loads the model and its feature extractor from a training output directory (e.g. <run>/best).

    predictor = EoTPredictor("artifacts/runs/<run>/best", device="cuda")
    p = predictor.predict(audio, sample_rate=16000)   # audio: float32 mono, the turn so far
    done = p >= predictor.threshold

Run (from the repo root):  python scripts/predict.py --model artifacts/runs/<run>/best clip1.wav [clip2.flac ...]
"""

import argparse
from pathlib import Path

import numpy as np
import soxr
import torch
from transformers import WhisperFeatureExtractor

from data.constants import SAMPLE_RATE
from data.dataset import crop_window, decode, narrowband, pad_window
from model import EoTModel


class EoTPredictor:
    def __init__(self, model_dir: str | Path, device: str = "cpu"):
        self.model = EoTModel.from_pretrained(model_dir).to(device).eval()
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_dir)
        self.device = device
        self.threshold = self.model.config.threshold

    @torch.no_grad()
    def predict(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> float:
        """`audio`: mono float waveform in [-1, 1] ending at the pause to classify."""
        x = np.asarray(audio, dtype=np.float32)
        if sample_rate != SAMPLE_RATE:
            x = soxr.resample(x, sample_rate, SAMPLE_RATE, quality="HQ").astype(np.float32)
        # Same preprocessing as training (data.dataset.EoTDataset), minus augmentation.
        x = crop_window(x)
        if self.model.config.narrowband:
            x = narrowband(x)
        window = pad_window(x, padding=self.model.config.padding)

        features = self.feature_extractor(window, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        features = features.input_features.to(self.device)
        logits = self.model(input_features=features).logits
        return float(torch.sigmoid(logits)[0])


def main() -> None:
    ap = argparse.ArgumentParser(description="P(turn complete) for audio files")
    ap.add_argument("--model", type=Path, required=True, help="a save_pretrained directory, e.g. <run>/best")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("files", nargs="+", type=Path)
    args = ap.parse_args()
    predictor = EoTPredictor(args.model, args.device)
    for path in args.files:
        p = predictor.predict(*decode(path.read_bytes()))
        print(f"{path}: P(complete) = {p:.3f} -> {'complete' if p >= predictor.threshold else 'incomplete'}")


if __name__ == "__main__":
    main()
