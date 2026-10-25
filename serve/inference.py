"""The model side of the server: an exported checkpoint -> per-request features -> batched P(turn complete).

The checkpoint directory (a training run's best/) holds config.json (threshold, window, preprocessing), the
feature extractor and model.onnx (scripts/export_onnx.py: the model plus the sigmoid, fp32). The model runs in
ONNX Runtime on CPU, so serving needs no torch. Preprocessing is turn_detection.data.audio, the same steps as
training, minus augmentation, with the window (2, 4 or 8 s) and settings from config.json:

    inference = EoTInference("artifacts/models/both-8s")
    features = inference.features(audio, sample_rate)      # (80, 100 * window_s), one request
    probs = inference.probabilities(np.stack([features]))   # (B,) float32, one batched forward pass
"""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soxr
from transformers import WhisperFeatureExtractor

from turn_detection.data import SAMPLE_RATE
from turn_detection.data.audio import pad_window, prepare_clip


class EoTInference:
    def __init__(self, model_dir: str | Path, threads: int = 0):
        model_dir = Path(model_dir)
        onnx_path = model_dir / "model.onnx"
        if not onnx_path.exists():
            raise FileNotFoundError(f"{onnx_path} not found: run `python scripts/export_onnx.py {model_dir}` first")

        # Note: EoTConfig read as JSON: importing turn_detection.model would import torch
        config = json.loads((model_dir / "config.json").read_text())
        self.window_s = config["window_s"]
        self.narrowband = config["narrowband"]
        self.loudness_dbfs = config["loudness_dbfs"]
        self.threshold = float(config["threshold"])

        # Note: several batcher workers call run() at once; with threads=1 each runs on its own calling thread
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads  # 0: ONNX Runtime's default, one per core
        self.threads = threads
        self.session = ort.InferenceSession(onnx_path, options, providers=["CPUExecutionProvider"])
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_dir)

    def features(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Mono float32 audio at any rate, ending at the pause to classify -> log-mel features (80, frames)."""
        x = np.asarray(audio, dtype=np.float32)

        # Only the last window_s seconds matter: cut before resampling so long inputs cost no more than that.
        x = x[-self.window_s * sample_rate :]
        if sample_rate != SAMPLE_RATE:
            x = soxr.resample(x, sample_rate, SAMPLE_RATE, quality="HQ").astype(np.float32)

        x = prepare_clip(x, window_s=self.window_s, narrowband=self.narrowband, loudness_dbfs=self.loudness_dbfs)
        x = pad_window(x, window_s=self.window_s)

        out = self.feature_extractor(x, sampling_rate=SAMPLE_RATE, return_tensors="np")
        return out.input_features[0]

    def warmup(self, max_batch: int = 1) -> None:
        """Run the preprocessing and the model once (batch 1 and `max_batch`): the first passes are slow."""
        features = self.features(np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)
        for batch in sorted({1, max_batch}):
            self.probabilities(np.stack([features] * batch))

    def probabilities(self, features: np.ndarray) -> np.ndarray:
        """(B, 80, frames) features -> (B,) P(turn complete)."""
        return self.session.run(None, {"input_features": features})[0]
