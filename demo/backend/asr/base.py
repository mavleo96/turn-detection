"""Pluggable streaming ASR interface.

An engine holds the loaded model; each session opens one stream. The stream receives every audio
sample of the session (ASR is always on). `flush()` ends the current segment: the engine emits a
final for it, resets its state, and keeps transcribing the audio that follows as the next segment.

Sample positions are stream sample indices (16 kHz, counted from the session's first sample), so the
pipeline can map them to arrival times. Callbacks may be invoked from an engine worker thread.
"""

from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np


@dataclass
class ASRPartial:
    segment_id: int
    piece: str  # newly decoded text
    text: str  # segment text so far
    emit_ns: int
    chunk_end_sample: int  # last audio sample (exclusive) of the chunk being decoded


@dataclass
class TokenTiming:
    piece: str
    emit_ns: int
    frame_ready_sample: int  # audio needed for the token's encoder frame, excluding lookahead
    chunk_end_sample: int  # audio needed by the chunk that produced it, including lookahead


@dataclass
class ASRFinal:
    segment_id: int
    text: str
    start_sample: int
    end_sample: int  # end of real (non-padding) audio in the segment
    final_ns: int
    tokens: list[TokenTiming] = field(default_factory=list)


class ASRStream(Protocol):
    def push(self, samples: np.ndarray) -> None:
        """Queue float32 16 kHz mono samples. Must not block."""

    def flush(self) -> None:
        """End the current segment at the audio pushed so far. Must not block."""

    def close(self) -> None:
        """Flush the last segment, wait for its final, and stop."""


class StreamingASREngine(Protocol):
    name: str

    def info(self) -> dict:
        """Model and streaming settings, for logging."""

    def open_stream(
        self,
        on_partial: Callable[[ASRPartial], None],
        on_final: Callable[[ASRFinal], None],
    ) -> ASRStream: ...
