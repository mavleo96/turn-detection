"""FastAPI app: POST /predict (audio -> P(turn complete)) and GET /health.

/predict takes the audio as the request body, in one of two forms (by Content-Type):
  - audio/wav, audio/flac, ... : an audio file, any sample rate, stereo is mixed down
  - application/octet-stream   : raw little-endian float32 mono samples; rate from ?sample_rate= (default 16000)

    curl -X POST --data-binary @clip.wav -H "Content-Type: audio/wav" http://127.0.0.1:8000/predict
    -> {"probability": 0.87, "complete": true, "threshold": 0.43}

Decoding and feature extraction run per request in a thread pool; the model runs through MicroBatcher.
When too many requests are already waiting for the model (`max_queue`), /predict answers 503 at once
(with Retry-After) rather than late. Each response carries `Server-Timing: total;dur=<ms>`, the time
spent in the server.
"""

import asyncio
import io
import time
from contextlib import asynccontextmanager

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, Query, Request, Response
from pydantic import BaseModel

from .batcher import MicroBatcher, QueueFull
from .inference import EoTInference

MAX_AUDIO_S = 60  # longer requests are rejected; only the model's last window_s seconds are used anyway


class Prediction(BaseModel):
    probability: float  # P(turn complete)
    complete: bool  # probability >= threshold
    threshold: float  # the model's decision threshold (chosen on val in training)


def read_audio(body: bytes, content_type: str, sample_rate: int) -> tuple[np.ndarray, int]:
    """Request body -> (mono float32 samples, sample rate); HTTPException for bad input."""
    if not body:
        raise HTTPException(400, "empty request body")

    if content_type.startswith("audio/"):
        # Note: soundfile, not training's torchcodec (FFmpeg, which needs torch in the image); for 16 kHz mono
        # (all training clips) both give the same samples, and the 3 dB a stereo mean differs by from
        # torchcodec's (L + R) / sqrt 2 is removed by loudness normalization
        try:
            x, sample_rate = sf.read(io.BytesIO(body), dtype="float32", always_2d=True)  # (T, channels)
        except (sf.LibsndfileError, RuntimeError) as error:
            raise HTTPException(400, f"could not decode audio: {error}") from error
        x = x.mean(axis=1)  # (T,)
    elif content_type == "application/octet-stream":
        if len(body) % 4:
            raise HTTPException(400, "raw audio must be float32 samples (a multiple of 4 bytes)")
        x = np.frombuffer(body, dtype="<f4")
    else:
        raise HTTPException(415, "Content-Type must be audio/* (wav, flac, ...) or application/octet-stream")

    if len(x) == 0:
        raise HTTPException(400, "no audio samples")
    if len(x) > MAX_AUDIO_S * sample_rate:
        raise HTTPException(400, f"audio longer than {MAX_AUDIO_S} s")
    if not np.isfinite(x).all():
        raise HTTPException(400, "audio contains NaN or infinite samples")

    return x, sample_rate


def create_app(
    inference: EoTInference,
    max_batch: int = 4,
    max_wait_ms: float = 2.0,
    workers: int = 1,
    max_queue: int = 0,
) -> FastAPI:
    batcher = MicroBatcher(
        inference.probabilities, max_batch=max_batch, max_wait_ms=max_wait_ms, workers=workers, max_queue=max_queue
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await asyncio.to_thread(inference.warmup, max_batch)  # first passes are slow: not on a request
        batcher.start()
        yield
        await batcher.stop()

    app = FastAPI(title="EoT inference", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict:
        mean_batch = batcher.requests / batcher.batches if batcher.batches else None
        return {
            "status": "ok",
            "threads": inference.threads,
            "threshold": inference.threshold,
            "max_batch": batcher.max_batch,
            "max_wait_ms": batcher.max_wait_s * 1000,
            "workers": batcher.workers,
            "max_queue": batcher.max_queue,
            "queued": batcher.queue.qsize(),
            "requests": batcher.requests,
            "batches": batcher.batches,
            "rejected": batcher.rejected,
            "mean_batch_size": mean_batch,
        }

    @app.post("/predict")
    async def predict(
        request: Request, response: Response, sample_rate: int = Query(16000, ge=8000, le=192000)
    ) -> Prediction:
        start = time.perf_counter()
        body = await request.body()
        content_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
        audio, sample_rate = read_audio(body, content_type, sample_rate)

        features = await asyncio.to_thread(inference.features, audio, sample_rate)
        try:
            probability = await batcher.submit(features)
        except QueueFull:
            raise HTTPException(503, "server busy, retry shortly", headers={"Retry-After": "1"}) from None

        response.headers["Server-Timing"] = f"total;dur={(time.perf_counter() - start) * 1000:.1f}"

        return Prediction(
            probability=probability,
            complete=probability >= inference.threshold,
            threshold=inference.threshold,
        )

    return app
