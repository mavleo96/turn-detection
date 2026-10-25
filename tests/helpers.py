"""Test helpers: fake Hub splits (FLAC audio + metadata) and a tiny random Whisper-shaped model for CPU tests."""

import io

import numpy as np
import soundfile as sf
import torch
from datasets import Audio, ClassLabel, Dataset, Features, Value
from transformers import WhisperConfig

from turn_detection.data import WINDOW_S
from turn_detection.model import EoTConfig, EoTModel


def flac_bytes(x: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, x, sr, format="FLAC")
    return buf.getvalue()


HF_FEATURES = Features(
    {
        "audio": {"bytes": Value("binary"), "path": Value("string")},  # cast to Audio below
        "id": Value("string"),
        "language": Value("string"),
        "endpoint_bool": Value("bool"),
        "midfiller": Value("bool"),
        "endfiller": Value("bool"),
        "synthetic": Value("bool"),
        "spoken_text": Value("string"),
        "dataset": Value("string"),
    }
)


def fake_hf_split(n: int, prefix: str, seed: int = 0, sample_rate: int = 16000) -> Dataset:
    """A tiny split with the real schema: 16 kHz FLAC clips, 0.5-9.5 s long, odd rows ending in 250 ms of
    silence; every 5th row Spanish; humans (synthetic False) at i = 0, 3, 6, ... with both labels; every 7th row
    from the holdout source rime_2."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        sr = sample_rate
        speech = 0.3 * rng.standard_normal(int(sr * (0.5 + i % 10)))
        audio = np.concatenate([speech, np.zeros(sr * (i % 2) // 4)])
        synthetic = i % 3 != 0
        rows.append(
            {
                "audio": {"bytes": flac_bytes(audio, sr), "path": None},
                "id": f"{prefix}-{i:03d}",
                "language": "spa" if i % 5 == 4 else "eng",
                "endpoint_bool": i % 2 == 0,
                "midfiller": None if i % 5 == 0 else i % 3 == 0,
                "endfiller": None if i % 5 == 0 else i % 7 == 0,
                "synthetic": synthetic,
                "spoken_text": None,
                "dataset": "rime_2" if i % 7 == 6 else "tts" if synthetic else "human",  # rime_2: a holdout source
            }
        )
    return Dataset.from_list(rows, features=HF_FEATURES).cast_column("audio", Audio())


TTD_FEATURES = Features(
    {
        "id": Value("string"),
        "audio": {"bytes": Value("binary"), "path": Value("string")},  # cast to Audio below
        "data_source": Value("string"),
        "original_data_id": Value("string"),
        "channel": Value("int8"),
        "timestamp_start": Value("float64"),
        "timestamp_end": Value("float64"),
        "label": ClassLabel(names=["incomplete", "complete"]),
        "meta": Value("string"),
    }
)


def fake_ttd_split(n: int, split: str, seed: int = 0) -> Dataset:
    """A tiny split with the ttd schema (mavleo96/turn-detection-data): rows ordered by source like the real
    one (all spokenwoz, then all seamless), 16 kHz FLAC clips 0.5-4.5 s long, labels alternating."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        source = "spokenwoz" if i < n // 2 else "seamless"
        audio = 0.3 * rng.standard_normal(int(16000 * (0.5 + i % 5)))
        rows.append(
            {
                "id": f"{source}_{split}_{i:03d}",
                "audio": {"bytes": flac_bytes(audio, 16000), "path": None},
                "data_source": source,
                "original_data_id": f"{source}-conv{i // 4}",
                "channel": i % 2,
                "timestamp_start": float(i),
                "timestamp_end": float(i) + 0.5 + i % 5,
                "label": i % 2,
                "meta": "{}",
            }
        )
    return Dataset.from_list(rows, features=TTD_FEATURES).cast_column("audio", Audio(sampling_rate=16000))


def tiny_whisper_config() -> WhisperConfig:
    """Whisper-shaped config (30 s / 1500 positions like the real one), small enough for CPU tests."""
    return WhisperConfig(
        num_mel_bins=80,
        d_model=32,
        encoder_layers=2,
        encoder_attention_heads=2,
        encoder_ffn_dim=64,
        decoder_layers=1,
        decoder_attention_heads=2,
        decoder_ffn_dim=64,
        max_source_positions=1500,
    )


def tiny_model(seed: int = 0, window_s: int = WINDOW_S) -> EoTModel:
    torch.manual_seed(seed)
    return EoTModel(EoTConfig(encoder_config=tiny_whisper_config().to_dict(), window_s=window_s))
