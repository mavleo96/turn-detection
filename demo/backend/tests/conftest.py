import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from backend.config import Config


def _test_wav() -> Path | None:
    p = os.environ.get("TURNDEMO_TEST_WAV")
    return Path(p) if p and Path(p).exists() else None


@pytest.fixture(scope="session")
def speech_wav() -> np.ndarray:
    """16 kHz mono English speech from $TURNDEMO_TEST_WAV (GPU tests skip without it)."""
    path = _test_wav()
    if path is None:
        pytest.skip("set TURNDEMO_TEST_WAV to a 16 kHz mono English speech WAV")
    audio, sr = sf.read(path, dtype="float32")
    assert sr == 16000 and audio.ndim == 1, "test WAV must be 16 kHz mono"
    return audio


@pytest.fixture(scope="session")
def asr_engine():
    from backend.asr.nemotron_hf import NemotronHFEngine

    return NemotronHFEngine(Config().asr)


@pytest.fixture(scope="session")
def silero_vad():
    from backend.vad.silero import SileroVAD

    return SileroVAD(Config().vad)
