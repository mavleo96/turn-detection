"""Per-clip audio preprocessing shared by training (data.dataset), evaluation and serving.

Only numpy, soundfile and soxr: serving imports this without the training stack. The order used everywhere:

    x, sr = decode(audio_bytes)          # mono float32 at its own rate (resample to 16 kHz if needed)
    x = prepare_clip(x, use_narrowband, loudness_dbfs)   # last 8 s -> 8 kHz round trip -> loudness, not padded
    ...                                  # (training only: augmentation, see data.transforms)
    x = pad_window(x)                    # zero-pad on the left to exactly 8 s

The settings come from the model's config (narrowband, loudness_dbfs), so training, evaluation and
serving preprocess the same way.
"""

import io

import numpy as np
import soundfile as sf
import soxr

from .constants import SAMPLE_RATE, WINDOW_S

SILENCE_DBFS = -70.0  # clips quieter than this (RMS) are left as they are by normalize_loudness


def decode(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    """Encoded audio (FLAC, WAV, ...) -> mono float32 at its original sample rate."""
    x, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)  # (T, channels)
    return x.mean(axis=1), sr


def crop_window(x: np.ndarray) -> np.ndarray:
    """The last 8 s (WINDOW_S) of the clip, or all of it when shorter (not padded: see pad_window)."""
    n = WINDOW_S * SAMPLE_RATE
    x = np.asarray(x, dtype=np.float32)
    return x[max(len(x) - n, 0) :]


def narrowband(x: np.ndarray) -> np.ndarray:
    """Telephone bandwidth: a 16 kHz -> 8 kHz -> 16 kHz round trip, which removes everything above 4 kHz.
    Applied to every clip (train, eval and inference) so 8 kHz and 16 kHz sources look the same."""
    if len(x) == 0:
        return x

    down = soxr.resample(x, SAMPLE_RATE, SAMPLE_RATE // 2, quality="HQ")
    up = soxr.resample(down, SAMPLE_RATE // 2, SAMPLE_RATE, quality="HQ")

    y = np.zeros_like(x)
    n = min(len(x), len(up))  # the round trip can be off by a sample
    y[:n] = up[:n]
    return y


def normalize_loudness(x: np.ndarray, target_dbfs: float = -30.0) -> np.ndarray:
    """Scale the clip to an RMS level of `target_dbfs`, so loudness can't tell sources (or datasets) apart.
    Near-silent clips (RMS below SILENCE_DBFS) are left alone; the result is scaled down if it would clip."""
    if len(x) == 0:
        return x

    rms = np.sqrt(np.mean(np.square(x, dtype=np.float64)))
    if rms < 10 ** (SILENCE_DBFS / 20):
        return x

    y = x * (10 ** (target_dbfs / 20) / rms)
    peak = np.abs(y).max()
    if peak > 1.0:
        y = y / peak
    return y.astype(np.float32)


def prepare_clip(x: np.ndarray, use_narrowband: bool = True, loudness_dbfs: float | None = -30.0) -> np.ndarray:
    """16 kHz clip -> its last 8 s, 8 kHz round trip (if `use_narrowband`), loudness normalized (unless
    `loudness_dbfs` is None). Not padded: augmentation goes in between, then pad_window."""
    x = crop_window(x)
    if use_narrowband:
        x = narrowband(x)
    if loudness_dbfs is not None:
        x = normalize_loudness(x, loudness_dbfs)
    return x


def pad_window(x: np.ndarray) -> np.ndarray:
    """Zero-pad a cropped clip to 8 s on the left, so the clip end (the pause) is always at the window end."""
    zeros = np.zeros(WINDOW_S * SAMPLE_RATE - len(x), dtype=np.float32)
    return np.concatenate([zeros, x])
