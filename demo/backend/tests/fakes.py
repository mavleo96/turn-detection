"""Test doubles: a controllable clock, a scripted VAD and a synchronous fake ASR engine."""

from backend.asr.base import ASRFinal, ASRPartial, TokenTiming
from backend.timing import FRAME_SAMPLES
from backend.vad.silero import VADEvent

MS = 1_000_000  # ns


class Clock:
    def __init__(self):
        self.t = 0

    def __call__(self):
        return self.t


class ScriptedVAD:
    """Returns the given event when the frame with that index is processed."""

    def __init__(self, script: dict[int, VADEvent]):
        self.script = script
        self.n = 0

    def reset(self):
        self.n = 0

    def process(self, frame):
        ev = self.script.get(self.n)
        self.n += 1
        return ev


class FakeASR:
    """Synchronous engine: a flush immediately produces a final 5 ms later on the clock."""

    name = "fake"

    def __init__(self, clock):
        self.clock = clock
        self.pushed = 0
        self.flushes = 0

    def info(self):
        return {"engine": "fake"}

    def open_stream(self, on_partial, on_final):
        engine = self
        engine.segment_start = 0

        class Stream:
            def push(self, samples):
                engine.pushed += len(samples)
                if engine.pushed == 3 * FRAME_SAMPLES:  # a partial while frame 2 is decoded
                    on_partial(ASRPartial(engine.flushes, "hi", "hi", engine.clock(), engine.pushed))

            def flush(self):
                seg = engine.flushes
                engine.flushes += 1
                tokens = []
                if engine.pushed > engine.segment_start:  # like the real engine: no audio, no tokens
                    tokens = [
                        TokenTiming(
                            "hi", engine.clock(), frame_ready_sample=FRAME_SAMPLES, chunk_end_sample=3 * FRAME_SAMPLES
                        )
                    ]
                final = ASRFinal(
                    seg, "hi" if seg == 0 else "", engine.segment_start, engine.pushed, engine.clock() + 5 * MS, tokens
                )
                engine.segment_start = engine.pushed
                on_final(final)

            def close(self):
                self.flush()

        return Stream()
