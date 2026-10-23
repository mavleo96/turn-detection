from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class FrontendConfig:
    host: str = "127.0.0.1"
    port: int = 7860
    backend_url: str = "http://127.0.0.1:8000"

    @property
    def backend_ws_url(self) -> str:
        base = self.backend_url.rstrip("/")
        return base.replace("https://", "wss://", 1).replace("http://", "ws://", 1) + "/ws/audio"


def load_config(path: str | Path) -> FrontendConfig:
    """Load the `frontend:` section of the shared demo config file."""
    raw = (yaml.safe_load(Path(path).read_text()) or {}).get("frontend", {}) or {}
    return FrontendConfig(**raw)
