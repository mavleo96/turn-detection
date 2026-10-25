"""The model side of the server: a trained model directory -> per-request features -> batched P(turn complete).

Preprocessing is turn_detection.data.audio, the same steps as training, minus augmentation, with the window
(2, 4 or 8 s) and settings from the model's config:

    inference = EoTInference("artifacts/runs/<run>/best")
    features = inference.features(audio, sample_rate)      # (80, 100 * window_s), one request, CPU
    probs = inference.probabilities(np.stack([features]))   # (B,) float32, one batched forward pass
"""

from pathlib import Path

import numpy as np
import soxr
import torch
from transformers import WhisperFeatureExtractor

from turn_detection.data import SAMPLE_RATE
from turn_detection.data.audio import pad_window, prepare_clip
from turn_detection.model import EoTModel


class EoTInference:
    def __init__(self, model_dir: str | Path, device: str = "cpu"):
        self.model = EoTModel.from_pretrained(model_dir).to(device).eval()
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_dir)
        self.device = device
        self.threshold = float(self.model.config.threshold)

    def features(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Mono float32 audio at any rate, ending at the pause to classify -> log-mel features (80, frames)."""
        config = self.model.config
        x = np.asarray(audio, dtype=np.float32)

        # Only the last window_s seconds matter: cut before resampling so long inputs cost no more than that.
        x = x[-config.window_s * sample_rate :]
        if sample_rate != SAMPLE_RATE:
            x = soxr.resample(x, sample_rate, SAMPLE_RATE, quality="HQ").astype(np.float32)

        x = prepare_clip(x, window_s=config.window_s, narrowband=config.narrowband, loudness_dbfs=config.loudness_dbfs)
        x = pad_window(x, window_s=config.window_s)

        out = self.feature_extractor(x, sampling_rate=SAMPLE_RATE, return_tensors="np")
        return out.input_features[0]

    def warmup(self, max_batch: int = 1) -> None:
        """Run the preprocessing and the model once (batch 1 and `max_batch`): the first passes are slow."""
        features = self.features(np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)
        for batch in sorted({1, max_batch}):
            self.probabilities(np.stack([features] * batch))

    @torch.inference_mode()
    def probabilities(self, features: np.ndarray) -> np.ndarray:
        """(B, 80, frames) features -> (B,) P(turn complete)."""
        inputs = torch.from_numpy(features).to(self.device)
        logits = self.model(input_features=inputs).logits
        return torch.sigmoid(logits).cpu().numpy()
