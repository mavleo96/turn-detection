"""Real model + real Silero through the pipeline, fed frame by frame (faster than real time)."""

import numpy as np
import pytest

from backend.config import Config
from backend.pipeline import Session
from backend.timing import FRAME_SAMPLES, SAMPLE_RATE

pytestmark = pytest.mark.gpu


def frames(audio: np.ndarray):
    n = len(audio) // FRAME_SAMPLES
    for i in range(n):
        yield audio[i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES]


def test_two_utterances_give_two_finals(asr_engine, silero_vad, speech_wav):
    clip = speech_wav[: 8 * SAMPLE_RATE]
    silence = np.zeros(SAMPLE_RATE, np.float32)
    audio = np.concatenate([silence, clip, silence, clip, silence])

    events = []
    s = Session("smoke", Config(), silero_vad, asr_engine, events.append)
    for seq, frame in enumerate(frames(audio)):
        s.on_frame(seq, frame)
    s.close()

    finals = [e for e in events if e["type"] == "asr_final" and e["reason"] == "vad_end"]
    texts = [f["text"] for f in finals if f["text"]]
    assert len(texts) >= 2, f"expected a final per utterance, got {[f['text'] for f in finals]}"
    assert any(e["type"] == "asr_partial" for e in events)

    for f in finals:
        assert f["vad_fire_to_final_ms"] >= 0
        assert f["speech_end_to_final_ms"] >= f["vad_end_latency_ms"]
        for tok in f["token_timings"]:
            assert tok["perceived_ms"] >= 0 and tok["compute_ms"] >= 0
            assert tok["perceived_ms"] >= tok["compute_ms"] - 0.01  # lookahead wait only adds

    # Segment audio must tile the stream: each segment starts where the previous ended.
    all_finals = [e for e in events if e["type"] == "asr_final"]
    for a, b in zip(all_finals, all_finals[1:]):
        assert a["segment_end_s"] == b["segment_start_s"]
    assert all_finals[-1]["segment_end_s"] == pytest.approx(len(audio) // FRAME_SAMPLES * FRAME_SAMPLES / SAMPLE_RATE)
