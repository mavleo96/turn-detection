"""The smart-turn dataset through Hugging Face `datasets`, EoTDataset (waveforms) and EoTCollator (features).

`load_split` loads a split from the HF cache (downloaded once into HF_HOME, then memory-mapped) with the
audio left as the original FLAC bytes: decoding with soundfile is fast, and `datasets`' own audio
decoding would need torchcodec.

    ds = load_split("train")                                        # all languages, all rows
    train = EoTDataset(ds, manifest["id"], build_augment(AugmentConfig()))
    item = train[0]                    # {"input_values": (128000,) float32, "labels": 0. or 1.}
    fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny", chunk_length=8)  # sized for 8 s, not 30 s
    batch = EoTCollator(fe)([train[0], train[1]])         # {"input_features": (2, 80, 800), "labels"}

Each item is decoded (16 kHz mono), cut to the last 8 s, band-limited to telephone bandwidth (`narrowband`, on
by default), optionally augmented (see data.transforms), and only then zero-padded to 8 s (on the left by
default; the per-clip audio steps are in data.audio), so the augmentation's noise level follows the speech,
not the padding. The collator runs `WhisperFeatureExtractor` (sized for 8 s) on the whole batch. Augmentation
is seeded from (seed, epoch, item), so a run is reproducible regardless of DataLoader worker count or
scheduling.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from datasets import Audio, load_dataset
from datasets import Dataset as HFDataset
from torch.utils.data import Dataset
from torch_audiomentations import Compose
from transformers import WhisperFeatureExtractor

from .audio import crop_window, decode, narrowband, pad_window
from .constants import LANGUAGE, META_COLUMNS, REPOS, SAMPLE_RATE
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
        narrowband: bool = True,
    ):
        self.rows = rows_for_ids(ds, ids)

        endpoint = np.asarray(ds["endpoint_bool"], dtype=bool)
        self.labels = endpoint[self.rows].astype(np.float32)
        self.audio = ds.select_columns(["audio"])  # undecoded FLAC bytes

        self.augment = augment
        self.padding = padding
        self.narrowband = narrowband
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

        x = crop_window(x)
        if self.narrowband:
            x = narrowband(x)
        if self.augment is not None:
            x = apply_augment(self.augment, x, (self.seed, self.epoch, i))  # before padding: SNR vs. the speech
        x = pad_window(x, padding=self.padding)

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
