"""FastAPI app: POST /predict (audio -> P(turn complete)) and GET /health.

/predict takes the audio as the request body, in one of two forms (by Content-Type):
  - audio/wav, audio/flac, ... : an audio file, any sample rate, stereo is mixed down
  - application/octet-stream   : raw little-endian float32 mono samples; rate from ?sample_rate= (default 16000)

    curl -X POST --data-binary @clip.wav -H "Content-Type: audio/wav" http://127.0.0.1:8000/predict
    -> {"probability": 0.87, "complete": true, "threshold": 0.43}

Decoding and feature extraction run per request in a thread pool; the model runs through MicroBatcher.
"""

import asyncio
from contextlib import asynccontextmanager

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel

from turn_detection.data.audio import decode

from .batcher import MicroBatcher
from .inference import EoTInference

MAX_AUDIO_S = 60  # longer requests are rejected; only the last 8 s are used anyway


class Prediction(BaseModel):
    probability: float  # P(turn complete)
    complete: bool  # probability >= threshold
    threshold: float  # the model's decision threshold (chosen on val in training)


def read_audio(body: bytes, content_type: str, sample_rate: int) -> tuple[np.ndarray, int]:
    """Request body -> (mono float32 samples, sample rate); HTTPException for bad input."""
    if not body:
        raise HTTPException(400, "empty request body")

    if content_type.startswith("audio/"):
        try:
            x, sample_rate = decode(body)
        except (sf.LibsndfileError, RuntimeError) as error:
            raise HTTPException(400, f"could not decode audio: {error}") from error
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


def create_app(inference: EoTInference, max_batch: int = 32, max_wait_ms: float = 5.0) -> FastAPI:
    batcher = MicroBatcher(inference.probabilities, max_batch=max_batch, max_wait_ms=max_wait_ms)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        batcher.start()
        yield
        await batcher.stop()

    app = FastAPI(title="EoT inference", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict:
        mean_batch = batcher.requests / batcher.batches if batcher.batches else None
        return {
            "status": "ok",
            "device": inference.device,
            "threshold": inference.threshold,
            "max_batch": batcher.max_batch,
            "max_wait_ms": batcher.max_wait_s * 1000,
            "requests": batcher.requests,
            "batches": batcher.batches,
            "mean_batch_size": mean_batch,
        }

    @app.post("/predict")
    async def predict(request: Request, sample_rate: int = Query(16000, ge=8000, le=192000)) -> Prediction:
        body = await request.body()
        content_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
        audio, sample_rate = read_audio(body, content_type, sample_rate)

        features = await asyncio.to_thread(inference.features, audio, sample_rate)
        probability = await batcher.submit(features)

        return Prediction(
            probability=probability,
            complete=probability >= inference.threshold,
            threshold=inference.threshold,
        )

    return app
