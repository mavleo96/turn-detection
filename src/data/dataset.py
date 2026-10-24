"""The smart-turn dataset through Hugging Face `datasets`, EoTDataset (waveforms) and EoTCollator (features).

`load_split` loads a split from the HF cache (downloaded once into HF_HOME, then memory-mapped) with the
audio left as the original FLAC bytes: decoding with soundfile is fast, and `datasets`' own audio
decoding would need torchcodec.

    ds = load_split("train")                                        # all languages, all rows
    train = EoTDataset(ds, manifest["id"], build_augment(AugmentConfig()))
    item = train[0]                    # {"input_values": (128000,) float32, "labels": 0. or 1.}
    fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny", chunk_length=8)  # sized for 8 s, not 30 s
    batch = EoTCollator(fe)([train[0], train[1]])         # {"input_features": (2, 80, 800), "labels"}

Each item is decoded (16 kHz mono), cut to the last 8 s (zero-padded when shorter, on the left by default) and
optionally augmented (see data.transforms); the collator runs `WhisperFeatureExtractor` (sized for 8 s)
on the whole batch. Augmentation is seeded from (seed, epoch, item), so a run is reproducible
regardless of DataLoader worker count or scheduling.
"""

import io
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from datasets import Audio, load_dataset
from datasets import Dataset as HFDataset
from torch.utils.data import Dataset
from torch_audiomentations import Compose
from transformers import WhisperFeatureExtractor

from .constants import LANGUAGE, META_COLUMNS, REPOS, SAMPLE_RATE, WINDOW_S
from .transforms import apply_augment


def load_split(split: str, num_proc: int | None = None) -> HFDataset:
    """One split (`train` or `test`), every language, audio as undecoded FLAC bytes."""
    ds = load_dataset(REPOS[split], split="train", num_proc=num_proc)
    return ds.cast_column("audio", Audio(decode=False))


def load_metadata(split: str, ds: HFDataset | None = None) -> pd.DataFrame:
    """English metadata of one split (no audio), sorted by id."""
    ds = load_split(split) if ds is None else ds
    meta = ds.select_columns(META_COLUMNS).to_pandas()
    return meta[meta["language"] == LANGUAGE].sort_values("id", ignore_index=True)


def rows_for_ids(ds: HFDataset, ids: Sequence[str]) -> np.ndarray:
    """Row numbers of `ids` (in the given order) in `ds`; every id must be present."""
    ids = pd.Series(ids)
    row_of_id = pd.Series(np.arange(len(ds)), index=pd.Index(ds["id"]))

    missing = ~ids.isin(row_of_id.index)
    if missing.any():
        raise KeyError(f"{int(missing.sum())} ids are not in the dataset, e.g. {ids[missing].iloc[0]}")

    return row_of_id.loc[ids.to_numpy()].to_numpy()


def decode(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    """Encoded audio (FLAC, WAV, ...) -> mono float32 at its original sample rate."""
    x, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)  # (T, channels)
    return x.mean(axis=1), sr


def last_window(x: np.ndarray, padding: str = "left") -> np.ndarray:
    """The last 8 s (WINDOW_S). Shorter input is zero-padded on the `padding` side: "left" keeps the clip end
    at the window end, "right" keeps the clip start at the window start."""
    if padding not in ("left", "right"):
        raise ValueError(f"padding must be 'left' or 'right', got {padding!r}")

    n = WINDOW_S * SAMPLE_RATE
    x = np.asarray(x, dtype=np.float32)
    if len(x) >= n:
        return x[len(x) - n :].copy()

    zeros = np.zeros(n - len(x), dtype=np.float32)
    if padding == "left":
        return np.concatenate([zeros, x])
    return np.concatenate([x, zeros])


class EoTDataset(Dataset):
    """{"input_values": (augmented) last 8 s of 16 kHz audio, float32, "labels": float32 (1. = turn complete)}
    for the clips with `ids`. Features are computed per batch by EoTCollator."""

    def __init__(
        self,
        ds: HFDataset,
        ids: Sequence[str],
        augment: Compose | None = None,
        seed: int = 0,
        padding: str = "left",
    ):
        self.rows = rows_for_ids(ds, ids)

        endpoint = np.asarray(ds["endpoint_bool"], dtype=bool)
        self.labels = endpoint[self.rows].astype(np.float32)
        self.audio = ds.select_columns(["audio"])  # undecoded FLAC bytes

        self.augment = augment
        self.padding = padding
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        row = int(self.rows[i])
        audio_bytes = self.audio[row]["audio"]["bytes"]
        x, sr = decode(audio_bytes)
        if sr != SAMPLE_RATE:
            raise ValueError(f"expected {SAMPLE_RATE} Hz audio, got {sr} Hz (row {row})")

        x = last_window(x, padding=self.padding)
        if self.augment is not None:
            x = apply_augment(self.augment, x, (self.seed, self.epoch, i))

        return {"input_values": x, "labels": self.labels[i]}


@dataclass
class EoTCollator:
    """Batch of EoTDataset items -> the model's inputs: {"input_features": (B, 80, frames), "labels": (B,)}."""

    feature_extractor: WhisperFeatureExtractor

    def __call__(self, items: list[dict]) -> dict[str, torch.Tensor]:
        # One clip at a time: on a CPU worker thread the extractor's batched path is ~2.4x slower for
        # 64 clips than 64 single calls (measured), and the results are the same.
        features = []
        for item in items:
            out = self.feature_extractor(item["input_values"], sampling_rate=SAMPLE_RATE, return_tensors="np")
            features.append(out.input_features[0])  # (80, frames)

        labels = [item["labels"] for item in items]

        return {
            "input_features": torch.from_numpy(np.stack(features)),  # (B, 80, frames)
            "labels": torch.tensor(labels, dtype=torch.float32),  # (B,)
        }
