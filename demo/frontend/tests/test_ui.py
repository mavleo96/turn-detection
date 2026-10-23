from frontend.config import FrontendConfig, load_config
from frontend.ui import SEGMENT_COLUMNS, config_markdown, session_tables

SESSION = {
    "session_id": "20261023-000000-abcd",
    "active": False,
    "closed": True,
    "audio_s": 12.5,
    "summary": {
        "vad_end_latency": {"count": 1, "mean": 258.0, "p50": 258.0, "p95": 258.0, "min": 258.0, "max": 258.0},
        "speech_end_to_final": {"count": 0, "mean": None, "p50": None, "p95": None, "min": None, "max": None},
    },
    "segments": [{"segment_id": 0, "reason": "vad_end", "text": "hello", "speech_end_to_final_ms": 400.0}],
}


def test_session_tables():
    line, summary_df, segments_df = session_tables(SESSION)
    assert "20261023-000000-abcd" in line and "finished" in line and "12.5 s" in line
    assert summary_df["metric"].tolist() == ["vad_end_latency", "speech_end_to_final"]
    assert summary_df.loc[0, "p50"] == 258.0
    assert segments_df.columns.tolist() == SEGMENT_COLUMNS
    assert segments_df.loc[0, "text"] == "hello"


def test_config_markdown():
    info = {
        "config": {
            "asr": {"model_id": "m", "chunk_latency_ms": 160, "dtype": "float32"},
            "vad": {"threshold": 0.5, "min_silence_ms": 200, "speech_pad_ms": 30},
        }
    }
    md = config_markdown(info)
    assert "160 ms" in md and "200 ms" in md


def test_backend_ws_url():
    assert FrontendConfig(backend_url="http://h:8000/").backend_ws_url == "ws://h:8000/ws/audio"
    assert FrontendConfig(backend_url="https://h").backend_ws_url == "wss://h/ws/audio"


def test_shared_config_file_loads():
    cfg = load_config("configs/demo.yaml")
    assert cfg.port == 7860 and cfg.backend_url == "http://127.0.0.1:8000"
