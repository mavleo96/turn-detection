"""Per-clip audio preprocessing shared by training (data.dataset), evaluation and serving.

Only numpy, soundfile and soxr: serving imports this without the training stack. The order used everywhere:

    x, sr = decode(audio_bytes)          # mono float32 at its own rate (resample to 16 kHz if needed)
    x = crop_window(x)                   # last 8 s, not padded
    x = narrowband(x)                    # if model.config.narrowband
    ...                                  # (training only: augmentation, see data.transforms)
    x = pad_window(x, padding)           # zero-pad to exactly 8 s
"""

import io

import numpy as np
import soundfile as sf
import soxr

from .constants import SAMPLE_RATE, WINDOW_S


def decode(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    """Encoded audio (FLAC, WAV, ...) -> mono float32 at its original sample rate."""
    x, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)  # (T, channels)
    return x.mean(axis=1), sr


def crop_window(x: np.ndarray) -> np.ndarray:
    """The last 8 s (WINDOW_S) of the clip, or all of it when shorter (not padded: see pad_window)."""
    n = WINDOW_S * SAMPLE_RATE
    x = np.asarray(x, dtype=np.float32)
    return x[max(len(x) - n, 0) :].copy()


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


def pad_window(x: np.ndarray, padding: str = "left") -> np.ndarray:
    """Zero-pad a cropped clip to 8 s on the `padding` side: "left" keeps the clip end at the window end,
    "right" keeps the clip start at the window start."""
    if padding not in ("left", "right"):
        raise ValueError(f"padding must be 'left' or 'right', got {padding!r}")

    n = WINDOW_S * SAMPLE_RATE
    zeros = np.zeros(n - len(x), dtype=np.float32)
    if padding == "left":
        return np.concatenate([zeros, x])
    return np.concatenate([x, zeros])
