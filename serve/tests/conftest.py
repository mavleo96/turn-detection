"""A tiny random EoT model saved like a training run's best/ (no network), and an app around it."""

import io

import numpy as np
import pytest
import soundfile as sf
from transformers import WhisperConfig, WhisperFeatureExtractor

from turn_detection.data import SAMPLE_RATE
from turn_detection.model import EoTConfig, EoTModel


def save_tiny_model(path, window_s: int):
    encoder = WhisperConfig(d_model=16, encoder_layers=2, encoder_attention_heads=2, encoder_ffn_dim=32)
    config = EoTConfig(encoder_config=encoder.to_dict(), window_s=window_s, head_hidden=8, threshold=0.5)
    EoTModel(config).save_pretrained(path)
    WhisperFeatureExtractor(chunk_length=window_s).save_pretrained(path)


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("model")
    save_tiny_model(path, window_s=8)
    return path


@pytest.fixture(scope="session")
def inference(model_dir):
    from serve.inference import EoTInference

    return EoTInference(model_dir)


def wav_bytes(seconds: float = 2.0, sample_rate: int = SAMPLE_RATE) -> bytes:
    x = (np.random.default_rng(0).standard_normal(int(seconds * sample_rate)) * 0.1).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, x, sample_rate, format="WAV")
    return buf.getvalue()
