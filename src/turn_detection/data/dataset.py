"""The datasets (data.constants.DATASETS) through Hugging Face `datasets`, EoTDataset (waveforms) and EoTCollator
(features).

`load_split` loads a split from the HF cache (downloaded once into HF_HOME, then memory-mapped) with the
audio left as the original FLAC bytes: decoding with soundfile is fast, and `datasets`' own audio
decoding would need torchcodec.

    ds = load_split("train", dataset="ttd")                         # every row of the split
    train = EoTDataset(ds, ds["id"], build_augment())
    st = load_split("test")                                         # smart-turn: all languages
    test = EoTDataset(st, manifest["id"], label_column="endpoint_bool")
    item = train[0]                    # {"input_values": (128000,) float32, "labels": bool}
    fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny", chunk_length=8)  # sized for 8 s, not 30 s
    batch = EoTCollator(fe)([train[0], train[1]])         # {"input_features": (2, 80, 800), "labels"}

Each item is decoded (16 kHz mono), cut to the last 8 s, band-limited to telephone bandwidth (`narrowband`, on
by default), optionally augmented (see data.transforms), and only then zero-padded to 8 s (on the left; the
per-clip audio steps are in data.audio), so the augmentation's noise level follows the speech, not the
padding. The collator runs `WhisperFeatureExtractor` (sized for 8 s) on the whole batch. Augmentation is
seeded from (seed, epoch, item), so a run is reproducible regardless of DataLoader worker count or scheduling.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from datasets import Audio, load_dataset
from datasets import Dataset as HFDataset
from torch.utils.data import ConcatDataset, Dataset
from torch_audiomentations import Compose
from transformers import WhisperFeatureExtractor

from .audio import decode, pad_window, prepare_clip
from .constants import DATASETS, LANGUAGE, META_COLUMNS, SAMPLE_RATE
from .transforms import apply_augment


def load_split(split: str, num_proc: int | None = None, dataset: str = "smart-turn") -> HFDataset:
    """One split of `dataset` (smart-turn: `train`, `test`; ttd: `train`, `validation`, `test`), every row,
    audio as undecoded FLAC bytes."""
    repo, repo_split = DATASETS[dataset]["splits"][split]
    ds = load_dataset(repo, split=repo_split, num_proc=num_proc)
    return ds.cast_column("audio", Audio(decode=False))


def load_metadata(split: str, ds: HFDataset | None = None) -> pd.DataFrame:
    """smart-turn: English metadata of one split (no audio), sorted by id."""
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
    """{"input_values": (augmented) last 8 s of 16 kHz audio, float32, "labels": bool (True = turn complete)}
    for the clips with `ids`; `label_column` holds 1 / True = complete (DATASETS[...]["label"]). Features are
    computed per batch by EoTCollator."""

    def __init__(
        self,
        ds: HFDataset,
        ids: Sequence[str],
        augment: Compose | None = None,
        seed: int = 0,
        narrowband: bool = True,
        loudness_dbfs: float | None = -30.0,
        label_column: str = "label",
    ):
        self.rows = rows_for_ids(ds, ids)

        self.labels = np.asarray(ds[label_column], dtype=bool)[self.rows]
        self.audio = ds.select_columns(["audio"])  # undecoded FLAC bytes

        self.augment = augment
        self.narrowband = narrowband
        self.loudness_dbfs = loudness_dbfs
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

        x = prepare_clip(x, self.narrowband, self.loudness_dbfs)
        if self.augment is not None:
            x = apply_augment(self.augment, x, (self.seed, self.epoch, i))  # before padding: SNR vs. the speech
        x = pad_window(x)

        return {"input_values": x, "labels": self.labels[i]}


class EoTConcat(ConcatDataset):
    """Several EoTDatasets as one (e.g. ttd + smart-turn), with their labels and the epoch passed through."""

    def __init__(self, datasets: list[EoTDataset]):
        super().__init__(datasets)
        self.labels = np.concatenate([d.labels for d in datasets])

    def set_epoch(self, epoch: int) -> None:
        for d in self.datasets:
            d.set_epoch(epoch)


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
            "labels": torch.tensor(labels),  # (B,) bool; the model casts
        }
