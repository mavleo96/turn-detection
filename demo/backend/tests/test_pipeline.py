"""Pipeline state machine with a scripted VAD, a fake ASR and a controlled clock."""

import numpy as np

from backend.config import Config
from backend.pipeline import Session
from backend.timing import FRAME_SAMPLES
from backend.vad.silero import VADEvent

from .fakes import MS, Clock, FakeASR, ScriptedVAD


def run_session(n_frames=20):
    clock = Clock()
    # speech starts in frame 1, ends after frame 4 (end sample = 5 frames), VAD end fires at frame 12
    vad = ScriptedVAD(
        {1: VADEvent("start", 0, FRAME_SAMPLES), 12: VADEvent("end", 5 * FRAME_SAMPLES + 480, 5 * FRAME_SAMPLES)}
    )
    asr = FakeASR(clock)
    events = []
    s = Session("test", Config(), vad, asr, events.append, now=clock)
    for seq in range(n_frames):
        clock.t = seq * 32 * MS  # frames arrive every 32 ms
        s.on_frame(seq, np.zeros(FRAME_SAMPLES, np.float32))
        clock.t += 1 * MS  # 1 ms of processing after each arrival
    s.close()
    return s, asr, events


def test_vad_end_flushes_asr_and_close_flushes_rest():
    s, asr, events = run_session()
    assert asr.flushes == 2  # one VAD end + one at close
    assert asr.pushed == 20 * FRAME_SAMPLES  # ASR got every frame
    finals = [e for e in events if e["type"] == "asr_final"]
    assert [f["reason"] for f in finals] == ["vad_end", "close"]
    assert finals[0]["text"] == "hi"
    # the empty close segment is logged but not counted as a segment row
    assert len(s.segment_rows()) == 1
    assert events[0]["type"] == "session_start" and events[-1]["type"] == "session_end"


def test_latency_metrics():
    s, _, events = run_session()
    by_type = {}
    for e in events:
        by_type.setdefault(e["type"], []).append(e)

    # VAD start fired at frame 1's processing (32 ms + 0) vs. arrival of frame 1 (32 ms)
    assert by_type["vad_start"][0]["latency_ms"] == 0.0
    # VAD end fired at 12*32 ms vs arrival of the last speech frame (frame 4, 128 ms)
    assert by_type["vad_end"][0]["latency_ms"] == 12 * 32 - 4 * 32

    final = by_type["asr_final"][0]
    assert final["vad_fire_to_final_ms"] == 5.0
    assert final["speech_end_to_final_ms"] == 12 * 32 + 5 - 4 * 32
    # token emitted at frame 12; its frame needed frame 0, its chunk needed frame 2
    tok = final["token_timings"][0]
    assert tok["perceived_ms"] == 12 * 32 and tok["compute_ms"] == 12 * 32 - 2 * 32

    partial = by_type["asr_partial"][0]
    assert partial["compute_latency_ms"] == 0.0  # emitted when frame 2 (its chunk end) arrived

    summary = by_type["session_end"][0]["summary"]
    assert summary["speech_end_to_final"]["count"] == 1
    assert summary["vad_fire_to_final"]["count"] == 1  # the close flush is not a VAD flush


def test_rtt_logged_as_one_way():
    clock = Clock()
    events = []
    s = Session("t", Config(), ScriptedVAD({}), FakeASR(clock), events.append, now=clock)
    s.on_rtt(10.0)
    s.close()
    assert [e["one_way_ms"] for e in events if e["type"] == "network"] == [5.0]


def test_close_while_speaking_keeps_vad_start():
    clock = Clock()
    events = []
    vad = ScriptedVAD({1: VADEvent("start", 0, FRAME_SAMPLES)})
    s = Session("t", Config(), vad, FakeASR(clock), events.append, now=clock)
    for seq in range(4):
        clock.t = seq * 32 * MS
        s.on_frame(seq, np.zeros(FRAME_SAMPLES, np.float32))
    s.close()
    (row,) = s.segment_rows()
    assert row["reason"] == "close"
    assert row["speech_start_s"] == FRAME_SAMPLES / 16000 and row["vad_start_latency_ms"] == 0.0


class FailingASR(FakeASR):
    """Like the real engine after a worker crash: close() raises with the original error as the cause."""

    def open_stream(self, on_partial, on_final):
        stream = super().open_stream(on_partial, on_final)

        def close():
            raise RuntimeError("ASR stream failed") from ValueError("boom")

        stream.close = close
        return stream


def test_asr_failure_reported_and_session_still_ends():
    events = []
    s = Session("t", Config(), ScriptedVAD({}), FailingASR(Clock()), events.append, now=Clock())
    s.close()
    assert [e["type"] for e in events[-2:]] == ["error", "session_end"]
    assert events[-2]["message"] == "ASR failed: ValueError('boom')"
