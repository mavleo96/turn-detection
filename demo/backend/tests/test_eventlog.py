from backend.eventlog import EventLog, read_events


def test_jsonl_round_trip(tmp_path):
    path = tmp_path / "runs" / "s.jsonl"
    log = EventLog(path)
    events = [{"type": "vad_start", "t_ns": 1, "latency_ms": 1.5}, {"type": "asr_final", "text": "héllo"}]
    for e in events:
        log.write(e)
    log.close()
    log.write({"type": "ignored after close"})
    assert read_events(path) == events
