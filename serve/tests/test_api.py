"""The FastAPI app end to end with the tiny model: inputs, errors, timing header, load shedding."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from serve.app import create_app

from .conftest import save_tiny_model, wav_bytes


@pytest.fixture
def client(inference):
    with TestClient(create_app(inference, workers=2)) as c:
        yield c


def test_predict_wav(client):
    r = client.post("/predict", content=wav_bytes(), headers={"Content-Type": "audio/wav"})
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["probability"] <= 1
    assert body["complete"] == (body["probability"] >= body["threshold"])
    assert r.headers["server-timing"].startswith("total;dur=")


def test_predict_raw_float32_at_8k(client):
    x = (np.random.default_rng(1).standard_normal(8000 * 3) * 0.1).astype("<f4")
    r = client.post(
        "/predict?sample_rate=8000", content=x.tobytes(), headers={"Content-Type": "application/octet-stream"}
    )
    assert r.status_code == 200


@pytest.mark.parametrize(
    ("body", "content_type", "status"),
    [
        (b"", "audio/wav", 400),
        (b"not audio", "audio/wav", 400),
        (b"abc", "application/octet-stream", 400),
        (np.array([np.nan], dtype="<f4").tobytes(), "application/octet-stream", 400),
        (b"x", "text/plain", 415),
    ],
)
def test_bad_input(client, body, content_type, status):
    assert client.post("/predict", content=body, headers={"Content-Type": content_type}).status_code == status


def test_window_follows_model_config(tmp_path):
    from serve.inference import EoTInference

    save_tiny_model(tmp_path, window_s=4)
    inference = EoTInference(tmp_path)
    features = inference.features(np.zeros(16000 * 6, dtype=np.float32), 16000)
    assert features.shape == (80, 400)
    assert inference.probabilities(features[None]).shape == (1,)


def test_too_long(client):
    r = client.post("/predict", content=wav_bytes(seconds=61), headers={"Content-Type": "audio/wav"})
    assert r.status_code == 400


def test_health_reports_settings_and_counters(client):
    client.post("/predict", content=wav_bytes(), headers={"Content-Type": "audio/wav"})
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["workers"] == 2 and h["requests"] >= 1 and h["rejected"] == 0


def test_busy_server_answers_503(inference, monkeypatch):
    from serve import batcher

    async def full(self, item):
        self.rejected += 1
        raise batcher.QueueFull

    monkeypatch.setattr(batcher.MicroBatcher, "submit", full)
    with TestClient(create_app(inference, max_queue=1)) as c:
        r = c.post("/predict", content=wav_bytes(), headers={"Content-Type": "audio/wav"})
        assert r.status_code == 503 and r.headers["retry-after"] == "1"
        assert c.get("/health").json()["rejected"] == 1
