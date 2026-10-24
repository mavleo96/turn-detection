"""Data: the Hugging Face datasets, per-clip audio steps and training augmentation. Constants and paths
live here.

- audio.py: per-clip steps shared by training, evaluation and serving (numpy and soxr only)
- dataset.py: load_split (train / validation / test: smart-turn + ttd in one schema), EoTDataset (decodes,
  waveforms), EoTCollator (features)
- transforms.py: gain + colored noise with torch-audiomentations, seeded per clip

Every clip goes through the same steps; augmentation sits before padding, so the noise SNR follows the speech:

    x = ...                                                    # 16 kHz mono float32, decoded by torchcodec
                                                               # (datasets in EoTDataset, AudioDecoder elsewhere)
    x = prepare_clip(x, window_s=8, use_narrowband=True, loudness_dbfs=-30.0)  # last 8 s -> 4 kHz band -> level
    x = apply_augment(augment, x, (seed, epoch, item))         # training only
    x = pad_window(x, window_s=8)                              # left zero-pad to 8 s

    train = EoTDataset(load_split("train"), build_augment())
    item = train[0]                                            # {"input_values": (128000,) float32, "labels": bool}
    fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny", chunk_length=8)
    batch = EoTCollator(fe)([train[0], train[1]])              # {"input_features": (2, 80, 800), "labels": (2,)}
"""

from pathlib import Path

# The Hugging Face datasets. The smart-turn repos (pipecat-ai v3.2, 23 languages) each have one split, "train";
# ttd (SpokenWOZ phone calls + Seamless Interaction, English) has train / validation / test.
SMART_TURN_TRAIN = "pipecat-ai/smart-turn-data-v3.2-train"
SMART_TURN_TEST = "pipecat-ai/smart-turn-data-v3.2-test"
TTD = "mavleo96/turn-detection-data"
DATASETS = ("smart-turn", "ttd")  # what load_split can combine

# Note: smart-turn sources kept out of train and validation, so their test clips are from sources never trained on
HOLDOUT = ("human_5", "rime_2")
VAL_FRAC = 0.1  # of smart-turn train (English, without HOLDOUT) for validation
LANGUAGE = "eng"  # smart-turn
META_COLUMNS = ["id", "language", "endpoint_bool", "midfiller", "endfiller", "synthetic", "spoken_text", "dataset"]
SEED = 42

SAMPLE_RATE = 16000
# Defaults of the preprocessing, used by data, the model config and the trainer (all saved in the model's config)
WINDOW_S = 8  # the model sees the last 8 s of each clip (EoTConfig.window_s: 2, 4 or 8)
NARROWBAND = True  # band-limit to 4 kHz (8 kHz round trip)
LOUDNESS_DBFS = -30.0  # RMS level clips are normalized to (None: not normalized)

_ROOT = Path(__file__).resolve().parents[3]  # the repo root (src/turn_detection/data/__init__.py)
RUNS_DIR = _ROOT / "artifacts" / "runs"  # artifacts/ is generated and gitignored
RESULTS_DIR = _ROOT / "results"  # dataset exploration and listening samples
