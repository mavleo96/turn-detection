"""One live session: every frame goes to the VAD and the ASR; a VAD end-of-speech flushes the ASR.

Events are passed to `sink(event: dict)`, possibly from the ASR worker thread. Each event has a
`type`, `t_ns` (server monotonic clock) and `t_ms` (ms since session start).

Latency metrics (ms), all against frame arrival times on the server clock:
  vad_start_latency      vad_start fire − arrival of the frame holding VAD's (unpadded) start sample
  vad_end_latency        vad_end fire − arrival of the last speech sample (unpadded end; includes min_silence)
  token_perceived        token emit − arrival of the audio its encoder frame needs (excl. lookahead)
  token_compute          token emit − arrival of the last sample of the chunk that produced it
  vad_fire_to_final      asr_final emit − vad_end fire (flush cost)
  speech_end_to_final    asr_final emit − arrival of the last speech sample (user-perceived)
  network_one_way        browser WebSocket round-trip / 2
"""

import threading
import time
from typing import Callable

import numpy as np

from .asr.base import ASRFinal, ASRPartial, StreamingASREngine
from .config import Config
from .timing import FRAME_SAMPLES, SAMPLE_RATE, FrameClock, ns_to_ms, summarize

METRICS = (
    "vad_start_latency",
    "vad_end_latency",
    "token_perceived",
    "token_compute",
    "vad_fire_to_final",
    "speech_end_to_final",
    "network_one_way",
)


def _ms(a_ns: int, b_ns: int) -> float:
    return round(ns_to_ms(a_ns - b_ns), 2)


def _p50(values: list[float]) -> float | None:
    return round(float(np.median(values)), 1) if values else None


class Session:
    def __init__(
        self,
        session_id: str,
        cfg: Config,
        vad,
        asr_engine: StreamingASREngine,
        sink: Callable[[dict], None],
        now: Callable[[], int] = time.monotonic_ns,
    ):
        self.session_id = session_id
        self.vad = vad
        self.sink = sink
        self.now = now
        self.clock = FrameClock()
        self._lock = threading.Lock()
        self._t0 = now()
        self._values: dict[str, list[float]] = {m: [] for m in METRICS}
        self.segments: list[dict] = []  # one row per finished segment
        self._flushes: list[dict] = []  # index == ASR segment id
        self._speech: dict | None = None  # current VAD speech region, until its end flushes the ASR
        self.closed = False

        vad.reset()
        self._emit(
            {
                "type": "session_start",
                "session_id": session_id,
                "config": cfg.to_dict(),
                "asr": asr_engine.info(),
            }
        )
        self.asr = asr_engine.open_stream(self._on_partial, self._on_final)

    def _emit(self, event: dict, t_ns: int | None = None) -> None:
        t_ns = self.now() if t_ns is None else t_ns
        event = {**event, "t_ns": t_ns, "t_ms": round(ns_to_ms(t_ns - self._t0), 2)}
        self.sink(event)

    def _add(self, metric: str, value: float) -> None:
        with self._lock:
            self._values[metric].append(value)

    # --- input (event loop thread) ---

    def on_frame(self, seq: int, pcm: np.ndarray, client_ts_ms: float | None = None) -> None:
        t = self.now()
        self.clock.record(seq, t)
        self._emit({"type": "frame", "seq": seq, "client_ts_ms": client_ts_ms}, t)
        self.asr.push(pcm)
        event = self.vad.process(pcm)
        if event is not None:
            self._on_vad(event)

    def on_rtt(self, rtt_ms: float) -> None:
        one_way = round(rtt_ms / 2, 2)
        self._add("network_one_way", one_way)
        self._emit({"type": "network", "rtt_ms": round(rtt_ms, 2), "one_way_ms": one_way})

    def _on_vad(self, event) -> None:
        fire = self.now()
        if event.kind == "start":
            latency = _ms(fire, self.clock.arrival_at(event.raw_sample + 1))
            self._add("vad_start_latency", latency)
            self._speech = {"start_sample": event.raw_sample, "start_latency_ms": latency}
            self._emit(
                {
                    "type": "vad_start",
                    "sample": event.raw_sample,
                    "padded_sample": event.sample,
                    "latency_ms": latency,
                },
                fire,
            )
            return

        latency = _ms(fire, self.clock.arrival_at(event.raw_sample))
        self._add("vad_end_latency", latency)
        speech = self._speech or {}
        self._speech = None
        with self._lock:
            segment_id = len(self._flushes)
            self._flushes.append(
                {
                    "reason": "vad_end",
                    "fire_ns": fire,
                    "speech_start_sample": speech.get("start_sample"),
                    "speech_end_sample": event.raw_sample,
                    "vad_start_latency_ms": speech.get("start_latency_ms"),
                    "vad_end_latency_ms": latency,
                }
            )
        self._emit(
            {
                "type": "vad_end",
                "sample": event.raw_sample,
                "padded_sample": event.sample,
                "latency_ms": latency,
                "flush_segment_id": segment_id,
            },
            fire,
        )
        self.asr.flush()

    # --- ASR callbacks (ASR worker thread) ---

    def _on_partial(self, p: ASRPartial) -> None:
        self._emit(
            {
                "type": "asr_partial",
                "segment_id": p.segment_id,
                "piece": p.piece,
                "text": p.text,
                "compute_latency_ms": _ms(p.emit_ns, self.clock.arrival_at(p.chunk_end_sample)),
            },
            p.emit_ns,
        )

    def _on_final(self, f: ASRFinal) -> None:
        with self._lock:
            flush = self._flushes[f.segment_id]
        tokens = []
        for tok in f.tokens:
            perceived = _ms(tok.emit_ns, self.clock.arrival_at(tok.frame_ready_sample))
            compute = _ms(tok.emit_ns, self.clock.arrival_at(tok.chunk_end_sample))
            tokens.append({"piece": tok.piece, "perceived_ms": perceived, "compute_ms": compute})
            self._add("token_perceived", perceived)
            self._add("token_compute", compute)

        fire_to_final = _ms(f.final_ns, flush["fire_ns"])
        speech_end_to_final = None
        if flush["reason"] == "vad_end":
            speech_end_to_final = _ms(f.final_ns, self.clock.arrival_at(flush["speech_end_sample"]))
            self._add("vad_fire_to_final", fire_to_final)
            self._add("speech_end_to_final", speech_end_to_final)

        def sec(sample):
            return None if sample is None else round(sample / SAMPLE_RATE, 3)

        row = {
            "segment_id": f.segment_id,
            "reason": flush["reason"],
            "text": f.text,
            "speech_start_s": sec(flush.get("speech_start_sample")),
            "speech_end_s": sec(flush.get("speech_end_sample")),
            "vad_start_latency_ms": flush.get("vad_start_latency_ms"),
            "vad_end_latency_ms": flush.get("vad_end_latency_ms"),
            "vad_fire_to_final_ms": fire_to_final,
            "speech_end_to_final_ms": speech_end_to_final,
            "tokens": len(tokens),
            "token_perceived_p50_ms": _p50([t["perceived_ms"] for t in tokens]),
            "token_compute_p50_ms": _p50([t["compute_ms"] for t in tokens]),
        }
        if flush["reason"] == "vad_end" or f.text:
            with self._lock:
                self.segments.append(row)
        self._emit(
            {
                "type": "asr_final",
                **row,
                "segment_start_s": sec(f.start_sample),
                "segment_end_s": sec(f.end_sample),
                "token_timings": tokens,
            },
            f.final_ns,
        )

    # --- results ---

    def summary(self) -> dict:
        with self._lock:
            return {m: summarize(v) for m, v in self._values.items()}

    def segment_rows(self) -> list[dict]:
        with self._lock:
            return list(self.segments)

    def close(self) -> None:
        """Flush the last segment, wait for the ASR, and emit session_end. Blocking.

        An ASR failure is reported as an `error` event; session_end is still emitted.
        """
        if self.closed:
            return
        self.closed = True
        speech = self._speech or {}  # stopped mid-speech: keep the segment's VAD start
        self._speech = None
        with self._lock:
            self._flushes.append(
                {
                    "reason": "close",
                    "fire_ns": self.now(),
                    "speech_start_sample": speech.get("start_sample"),
                    "vad_start_latency_ms": speech.get("start_latency_ms"),
                }
            )
        try:
            self.asr.close()
        except Exception as e:
            self._emit({"type": "error", "message": f"ASR failed: {e.__cause__ or e!r}"})
        self._emit(
            {
                "type": "session_end",
                "frames": self.clock.num_frames,
                "audio_s": round(self.clock.num_frames * FRAME_SAMPLES / SAMPLE_RATE, 3),
                "summary": self.summary(),
            }
        )
