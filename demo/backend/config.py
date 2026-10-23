from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

# chunk latency (ms) -> num_lookahead_tokens for nemotron-speech-streaming (80 ms encoder frames)
SUPPORTED_CHUNK_LATENCIES_MS = {160: 1, 560: 6, 1120: 13}


@dataclass
class ASRConfig:
    model_id: str = "nvidia/nemotron-speech-streaming-en-0.6b"
    chunk_latency_ms: int = 160
    dtype: str = "float32"
    device: str = "cuda"

    @property
    def num_lookahead_tokens(self) -> int:
        return SUPPORTED_CHUNK_LATENCIES_MS[self.chunk_latency_ms]


@dataclass
class VADConfig:
    threshold: float = 0.5
    min_silence_ms: int = 200
    speech_pad_ms: int = 30


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8000


@dataclass
class Config:
    asr: ASRConfig = field(default_factory=ASRConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    runs_dir: str = "runs"

    def validate(self) -> "Config":
        if self.asr.chunk_latency_ms not in SUPPORTED_CHUNK_LATENCIES_MS:
            raise ValueError(
                f"asr.chunk_latency_ms={self.asr.chunk_latency_ms} is not supported; "
                f"choose one of {sorted(SUPPORTED_CHUNK_LATENCIES_MS)}"
            )
        if self.asr.dtype not in ("float32", "float16", "bfloat16"):
            raise ValueError(f"asr.dtype must be float32, float16 or bfloat16, got {self.asr.dtype!r}")
        if not 0.0 < self.vad.threshold < 1.0:
            raise ValueError(f"vad.threshold must be in (0, 1), got {self.vad.threshold}")
        if self.vad.min_silence_ms <= 0 or self.vad.speech_pad_ms < 0:
            raise ValueError("vad.min_silence_ms must be > 0 and vad.speech_pad_ms >= 0")
        return self

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: str | Path) -> Config:
    """Load the `backend:` section of the shared demo config file."""
    raw = (yaml.safe_load(Path(path).read_text()) or {}).get("backend", {}) or {}
    cfg = Config(
        asr=ASRConfig(**raw.get("asr", {})),
        vad=VADConfig(**raw.get("vad", {})),
        server=ServerConfig(**raw.get("server", {})),
        runs_dir=raw.get("runs_dir", "runs"),
    )
    return cfg.validate()
