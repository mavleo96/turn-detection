import numpy as np
import soxr

from . import LOUDNESS_DBFS, NARROWBAND, SAMPLE_RATE, WINDOW_S

SILENCE_DBFS = -70.0  # clips quieter than this (RMS) are left as they are by normalize_loudness


def narrowband(x: np.ndarray) -> np.ndarray:
    """Telephone bandwidth: a 16 kHz -> 8 kHz -> 16 kHz round trip, which removes everything above 4 kHz."""
    if len(x) == 0:
        return x

    down = soxr.resample(x, SAMPLE_RATE, SAMPLE_RATE // 2, quality="HQ")
    up = soxr.resample(down, SAMPLE_RATE // 2, SAMPLE_RATE, quality="HQ")[: len(x)]

    # Note: the round trip can be a sample short
    return np.pad(up, (0, len(x) - len(up)))


def normalize_loudness(x: np.ndarray, target_dbfs: float = LOUDNESS_DBFS) -> np.ndarray:
    """Scale the clip to an RMS level of `target_dbfs`; near-silent clips are left alone, loud ones not clipped."""
    if len(x) == 0:
        return x

    rms = np.sqrt(np.mean(np.square(x, dtype=np.float64)))
    if rms < 10 ** (SILENCE_DBFS / 20):
        return x

    y = x * (10 ** (target_dbfs / 20) / rms)
    peak = np.abs(y).max()
    if peak > 1.0:
        y = y / peak

    # Note: the float64 RMS promotes the result to float64
    return y.astype(np.float32)


def prepare_clip(
    x: np.ndarray,
    *,
    window_s: int = WINDOW_S,
    use_narrowband: bool = NARROWBAND,
    loudness_dbfs: float | None = LOUDNESS_DBFS,
) -> np.ndarray:
    """16 kHz clip -> its last `window_s` seconds, band-limited and loudness normalized (each optional); not padded."""
    x = np.asarray(x, dtype=np.float32)[-window_s * SAMPLE_RATE :]
    if use_narrowband:
        x = narrowband(x)
    if loudness_dbfs is not None:
        x = normalize_loudness(x, loudness_dbfs)
    return x


def pad_window(x: np.ndarray, window_s: int = WINDOW_S) -> np.ndarray:
    """Zero-pad a prepared clip to `window_s` seconds on the left, so the clip end (the pause) is at the window end."""
    zeros = np.zeros(window_s * SAMPLE_RATE - len(x), dtype=np.float32)
    return np.concatenate([zeros, x])
