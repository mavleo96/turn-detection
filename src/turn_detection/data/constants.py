"""Constants and paths shared by the data code, training and the scripts."""

from pathlib import Path

# The Hugging Face datasets. Per split: (repo, the repo's own split name). `label` is the column with 1 / True =
# turn complete; `source` says where a clip comes from (for per-source metrics).
DATASETS = {
    # pipecat-ai smart-turn v3.2: 23 languages, mostly TTS; we use English (LANGUAGE) with our own val split
    "smart-turn": {
        "splits": {
            "train": ("pipecat-ai/smart-turn-data-v3.2-train", "train"),
            "test": ("pipecat-ai/smart-turn-data-v3.2-test", "train"),
        },
        "label": "endpoint_bool",
        "source": "dataset",
    },
    # Our own: SpokenWOZ phone calls + Seamless Interaction (naturalistic), English, with its own splits
    "ttd": {
        "splits": {split: ("mavleo96/turn-detection-data", split) for split in ("train", "validation", "test")},
        "label": "label",
        "source": "data_source",
    },
}

META_COLUMNS = ["id", "language", "endpoint_bool", "midfiller", "endfiller", "synthetic", "spoken_text", "dataset"]
LANGUAGE = "eng"  # smart-turn
SEED = 42

SAMPLE_RATE = 16000
WINDOW_S = 8  # the model sees the last 8 s of each clip

_ROOT = Path(__file__).resolve().parents[3]  # the repo root (src/turn_detection/data/)
MANIFESTS_DIR = _ROOT / "artifacts" / "manifests"  # artifacts/ is generated and gitignored
RUNS_DIR = _ROOT / "artifacts" / "runs"
RESULTS_DIR = _ROOT / "results"  # dataset exploration and listening samples
