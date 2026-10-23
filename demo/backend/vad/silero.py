from dataclasses import dataclass

import numpy as np
import torch
from silero_vad import VADIterator, load_silero_vad

from ..config import VADConfig
from ..timing import FRAME_SAMPLES, SAMPLE_RATE


@dataclass
class VADEvent:
    kind: str  # "start" | "end"
    sample: int  # boundary as reported by Silero (includes speech_pad_ms)
    raw_sample: int  # unpadded boundary: start of first speech frame / end of last speech frame


class SileroVAD:
    """Streaming Silero VAD over 512-sample frames at 16 kHz.

    One instance per process (the model is stateful); call `reset()` at the start of each session.
    """

    def __init__(self, cfg: VADConfig):
        self.cfg = cfg
        self.model = load_silero_vad()
        self._iter = VADIterator(
            self.model,
            threshold=cfg.threshold,
            sampling_rate=SAMPLE_RATE,
            min_silence_duration_ms=cfg.min_silence_ms,
            speech_pad_ms=cfg.speech_pad_ms,
        )
        self._pad = int(SAMPLE_RATE * cfg.speech_pad_ms / 1000)

    def reset(self) -> None:
        self._iter.reset_states()

    def process(self, frame: np.ndarray) -> VADEvent | None:
        if frame.shape != (FRAME_SAMPLES,):
            raise ValueError(f"expected a ({FRAME_SAMPLES},) frame, got {frame.shape}")
        out = self._iter(torch.from_numpy(frame))
        if not out:
            return None
        if "start" in out:
            # Silero triggers on the frame that just ended at current_sample.
            return VADEvent("start", int(out["start"]), self._iter.current_sample - FRAME_SAMPLES)
        # Silero reports end = temp_end + pad - frame, where temp_end is the end of the first silent frame.
        return VADEvent("end", int(out["end"]), int(out["end"]) - self._pad)
