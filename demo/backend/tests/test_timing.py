import pytest

from backend.config import Config, load_config
from backend.timing import FRAME_SAMPLES, FrameClock, summarize


def test_arrival_at_maps_sample_end_to_containing_frame():
    clock = FrameClock()
    for seq in range(4):
        clock.record(seq, 1000 * (seq + 1))
    assert clock.arrival_at(0) == 1000  # nothing needed yet -> first frame
    assert clock.arrival_at(1) == 1000  # sample 0 is in frame 0
    assert clock.arrival_at(FRAME_SAMPLES) == 1000  # samples [0, 512) all in frame 0
    assert clock.arrival_at(FRAME_SAMPLES + 1) == 2000
    assert clock.arrival_at(4 * FRAME_SAMPLES) == 4000
    assert clock.arrival_at(100 * FRAME_SAMPLES) == 4000  # padding beyond received audio -> last frame


def test_frames_must_arrive_in_order():
    clock = FrameClock()
    clock.record(0, 1)
    with pytest.raises(ValueError):
        clock.record(2, 2)


def test_summarize():
    s = summarize([10.0, 20.0, 30.0, 40.0])
    assert s["count"] == 4 and s["mean"] == 25.0 and s["p50"] == 25.0
    assert s["min"] == 10.0 and s["max"] == 40.0
    assert summarize([])["count"] == 0


def test_config_rejects_80ms_chunks(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("backend:\n  asr:\n    chunk_latency_ms: 80\n")
    with pytest.raises(ValueError, match="not supported"):
        load_config(p)


def test_default_config_file_loads():
    cfg = load_config("configs/demo.yaml")
    assert cfg.asr.chunk_latency_ms == 160 and cfg.asr.num_lookahead_tokens == 1
    assert cfg.vad.min_silence_ms == 200
    assert Config().asr.model_id == cfg.asr.model_id
