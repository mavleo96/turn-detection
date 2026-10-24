"""Constants and paths shared by the data code, training and the scripts."""

from pathlib import Path

REPOS = {"train": "pipecat-ai/smart-turn-data-v3.2-train", "test": "pipecat-ai/smart-turn-data-v3.2-test"}
META_COLUMNS = ["id", "language", "endpoint_bool", "midfiller", "endfiller", "synthetic", "spoken_text", "dataset"]
LANGUAGE = "eng"
SEED = 42

SAMPLE_RATE = 16000
WINDOW_S = 8  # the model sees the last 8 s of each clip

_ROOT = Path(__file__).resolve().parents[3]  # the repo root (src/turn_detection/data/)
MANIFESTS_DIR = _ROOT / "artifacts" / "manifests"  # artifacts/ is generated and gitignored
FOLDS_DIR = _ROOT / "artifacts" / "folds"
RUNS_DIR = _ROOT / "artifacts" / "runs"
RESULTS_DIR = _ROOT / "results"  # dataset exploration and listening samples
