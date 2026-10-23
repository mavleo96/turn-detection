"""nvidia/nemotron-speech-streaming-en-0.6b (cache-aware FastConformer RNN-T) via HF Transformers.

Each segment is one streaming `model.generate()` call in the stream's worker thread. Its input is a
generator that cuts the live audio into exactly-sized mel chunks and blocks until enough audio has
arrived. `flush()` zero-pads the last chunk and ends the generator, so `generate()` returns the
segment's final; the next `generate()` starts with fresh caches (reset).

Token timing: the streamer is called once per decoding step (blanks included), so step i lines up
with `sequences[i + 1]` and `durations[i + 1]`; the durations give each token's encoder frame.
"""

import queue
import threading
import time
import traceback
from typing import Callable

import numpy as np
import torch
import transformers
from tokenizers.decoders import DecodeStream
from transformers import AutoModelForRNNT, AutoProcessor
from transformers.generation.streamers import BaseStreamer

from ..config import ASRConfig
from ..timing import SAMPLE_RATE
from .base import ASRFinal, ASRPartial, TokenTiming

_FLUSH = object()
_STOP = object()


class NemotronHFEngine:
    name = "nemotron-hf"

    def __init__(self, cfg: ASRConfig, warmup: bool = True):
        self.cfg = cfg
        self.processor = AutoProcessor.from_pretrained(cfg.model_id)
        self.processor.set_num_lookahead_tokens(cfg.num_lookahead_tokens)
        self.model = AutoModelForRNNT.from_pretrained(
            cfg.model_id, device_map=cfg.device, dtype=getattr(torch, cfg.dtype)
        ).eval()
        fe = self.processor.feature_extractor
        self.hop, self.n_fft, self.win = fe.hop_length, fe.n_fft, fe.win_length
        self.subsampling = self.model.config.encoder_config.subsampling_factor
        self.blank_id = self.model.config.blank_token_id
        self.pad_id = self.processor.tokenizer.pad_token_id
        self._open = False
        if warmup:
            self._warmup()

    def info(self) -> dict:
        return {
            "engine": self.name,
            "model_id": self.cfg.model_id,
            "chunk_latency_ms": self.cfg.chunk_latency_ms,
            "num_lookahead_tokens": self.cfg.num_lookahead_tokens,
            "streaming_latency_ms": self.processor.streaming_latency_ms,
            "dtype": self.cfg.dtype,
            "device": self.cfg.device,
            "transformers": transformers.__version__,
            "torch": torch.__version__,
        }

    def frame_ready_sample(self, frame_idx: int) -> int:
        """Segment-relative audio end needed for an encoder frame, excluding lookahead.

        Encoder frame f is aligned with mel frame f * subsampling, whose window ends at
        f * subsampling * hop + win_length // 2.
        """
        return frame_idx * self.subsampling * self.hop + self.win // 2

    def open_stream(
        self,
        on_partial: Callable[[ASRPartial], None],
        on_final: Callable[[ASRFinal], None],
    ) -> "NemotronStream":
        if self._open:
            raise RuntimeError("only one ASR stream can be open at a time")
        self._open = True
        return NemotronStream(self, on_partial, on_final)

    def _warmup(self) -> None:
        stream = self.open_stream(lambda p: None, lambda f: None)
        stream.push(np.zeros(SAMPLE_RATE, dtype=np.float32))
        stream.close()


class _StepStreamer(BaseStreamer):
    """Called by generate() once with the initial decoder input, then once per decoding step."""

    def __init__(self, segment: "_Segment"):
        self.segment = segment
        self.started = False
        self.step = 0

    def put(self, value):
        if not self.started:
            self.started = True
            return
        self.segment.on_step(self.step, int(value.reshape(-1)[0]))
        self.step += 1

    def end(self):
        pass


class _Segment:
    """Audio and decoding state of one segment (one generate() call)."""

    def __init__(self, stream: "NemotronStream", segment_id: int, start_sample: int):
        self.stream = stream
        self.engine = stream.engine
        self.segment_id = segment_id
        self.start_sample = start_sample
        self._buf = np.zeros(0, dtype=np.float32)
        self._buf_offset = 0  # segment-relative sample index of _buf[0]
        self.received = 0  # real samples received for this segment
        self.flushed = False
        self.chunk_end = 0  # segment-relative end of the chunk being decoded
        self.text = ""
        self._decode = DecodeStream(skip_special_tokens=True)
        self.tokens: list[tuple[int, str, int, int]] = []  # (step, piece, emit_ns, chunk_end)

    def _need(self, end: int) -> None:
        """Block until segment audio up to `end` is buffered; after a flush, zero-pad instead."""
        while self._buf_offset + len(self._buf) < end:
            if self.flushed:
                pad = end - (self._buf_offset + len(self._buf))
                self._buf = np.concatenate([self._buf, np.zeros(pad, dtype=np.float32)])
                return
            item = self.stream.get()
            if item is _FLUSH:
                self.flushed = True
            elif item is _STOP:  # close() always flushes first; treat defensively as a flush
                self.flushed = True
                self.stream.unget(_STOP)
            else:
                self._buf = np.concatenate([self._buf, item])
                self.received += len(item)

    def _slice(self, start: int, end: int) -> np.ndarray:
        self._need(end)
        # Earlier audio is never needed again: chunks only move forward.
        drop = max(0, start - self._buf_offset)
        if drop:
            self._buf = self._buf[drop:]
            self._buf_offset += drop
        return self._buf[start - self._buf_offset : end - self._buf_offset]

    def _features(self, audio: np.ndarray, first: bool) -> torch.Tensor:
        return self.engine.processor(
            audio,
            sampling_rate=SAMPLE_RATE,
            is_streaming=True,
            is_first_audio_chunk=first,
            return_tensors="pt",
        ).input_features

    def has_audio(self) -> bool:
        """Wait for the first chunk's audio; False if the segment was flushed with no audio at all."""
        self._need(self.engine.processor.num_samples_first_audio_chunk)
        return self.received > 0

    def chunks(self):
        p = self.engine.processor
        hop, n_fft = self.engine.hop, self.engine.n_fft
        first_end = p.num_samples_first_audio_chunk
        feats = self._features(self._slice(0, first_end), first=True)
        self.chunk_end = first_end
        yield feats[:, : p.num_mel_frames_first_audio_chunk, :]

        mel_idx = p.num_mel_frames_first_audio_chunk
        while True:
            start = mel_idx * hop - n_fft // 2
            end = start + p.num_samples_per_audio_chunk
            audio = self._slice(start, end)
            # Stop once the chunk's new mel frames would start after the real audio.
            if self.flushed and mel_idx * hop >= self.received:
                return
            feats = self._features(audio, first=False)
            self.chunk_end = end
            yield feats
            mel_idx += p.num_mel_frames_per_audio_chunk

    def on_step(self, step: int, token_id: int) -> None:
        if token_id in (self.engine.blank_id, self.engine.pad_id):
            return
        emit_ns = time.monotonic_ns()
        piece = self._decode.step(self.engine.processor.tokenizer._tokenizer, token_id) or ""
        self.tokens.append((step, piece, emit_ns, self.chunk_end))
        if piece:
            self.text += piece
            self.stream.on_partial(
                ASRPartial(
                    segment_id=self.segment_id,
                    piece=piece,
                    text=self.text,
                    emit_ns=emit_ns,
                    chunk_end_sample=self.start_sample + self.chunk_end,
                )
            )

    def run(self) -> ASRFinal:
        if not self.has_audio():
            return ASRFinal(self.segment_id, "", self.start_sample, self.start_sample, time.monotonic_ns())
        engine = self.engine
        out = engine.model.generate(
            input_features=self.chunks(),
            num_lookahead_tokens=engine.cfg.num_lookahead_tokens,
            streamer=_StepStreamer(self),
            return_dict_in_generate=True,
        )
        text = engine.processor.batch_decode(out.sequences, skip_special_tokens=True)[0].strip()
        final_ns = time.monotonic_ns()

        durations = out.durations[0]
        frame_idx = (durations.cumsum(-1) - durations).tolist()
        tokens = [
            TokenTiming(
                piece=piece,
                emit_ns=emit_ns,
                frame_ready_sample=self.start_sample + engine.frame_ready_sample(int(frame_idx[step + 1])),
                chunk_end_sample=self.start_sample + chunk_end,
            )
            for step, piece, emit_ns, chunk_end in self.tokens
        ]
        return ASRFinal(
            segment_id=self.segment_id,
            text=text,
            start_sample=self.start_sample,
            end_sample=self.start_sample + self.received,
            final_ns=final_ns,
            tokens=tokens,
        )


class NemotronStream:
    def __init__(self, engine: NemotronHFEngine, on_partial, on_final):
        self.engine = engine
        self.on_partial = on_partial
        self.on_final = on_final
        self.q: queue.Queue = queue.Queue()
        self.error: BaseException | None = None
        self._pending = None  # one item read ahead by the worker
        self._thread = threading.Thread(target=self._run, name="asr-stream", daemon=True)
        self._thread.start()

    def push(self, samples: np.ndarray) -> None:
        self.q.put(np.asarray(samples, dtype=np.float32))

    def flush(self) -> None:
        self.q.put(_FLUSH)

    def close(self) -> None:
        self.q.put(_FLUSH)
        self.q.put(_STOP)
        self._thread.join()
        self.engine._open = False
        if self.error is not None:
            raise RuntimeError("ASR stream failed") from self.error

    def _run(self) -> None:
        segment_id, start_sample = 0, 0
        with torch.inference_mode():
            while True:
                try:
                    item = self.get()
                    if item is _STOP:
                        return
                    self.unget(item)
                    segment = _Segment(self, segment_id, start_sample)
                    final = segment.run()
                except Exception as e:  # keep draining so close() returns
                    traceback.print_exc()
                    self.error = e
                    self._drain()
                    return
                self.on_final(final)
                segment_id += 1
                start_sample = final.end_sample

    def get(self):
        """Next queued item (worker thread only)."""
        if self._pending is not None:
            item, self._pending = self._pending, None
            return item
        return self.q.get()

    def unget(self, item) -> None:
        self._pending = item

    def _drain(self) -> None:
        while self.get() is not _STOP:
            pass
