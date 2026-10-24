from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from datasets import Audio, ClassLabel, concatenate_datasets, load_dataset
from datasets import Dataset as HFDataset
from torch.utils.data import Dataset
from torch_audiomentations import Compose
from transformers import WhisperFeatureExtractor

from . import DATASETS, HOLDOUT, LANGUAGE, SAMPLE_RATE, SEED, SMART_TURN_TEST, SMART_TURN_TRAIN, TTD, VAL_FRAC
from .audio import pad_window, prepare_clip
from .transforms import apply_augment

COLUMNS = ["id", "audio", "dataset", "label"]
LABELS = ClassLabel(names=["incomplete", "complete"])
# Note: datasets decodes with torchcodec, resampling to 16 kHz and downmixing to mono ((L + R) / sqrt 2)
AUDIO = Audio(sampling_rate=SAMPLE_RATE, num_channels=1)


def load_split(split: str, datasets: Sequence[str] = DATASETS, num_proc: int | None = None) -> HFDataset:
    """train / validation / test of `datasets` (smart-turn English, ttd) as id, audio (16 kHz mono), dataset (source),
    label. smart-turn train without HOLDOUT is split VAL_FRAC at random into validation; test is its whole English
    test split."""
    unknown = set(datasets) - set(DATASETS)
    if unknown or not datasets:
        raise ValueError(f"datasets must be some of {DATASETS}, got {list(datasets)}")

    parts = []
    if "smart-turn" in datasets:
        holdout = () if split == "test" else HOLDOUT
        repo = SMART_TURN_TEST if split == "test" else SMART_TURN_TRAIN
        smart_turn = load_dataset(repo, split="train", num_proc=num_proc)  # both repos have one split, "train"
        smart_turn = smart_turn.filter(
            lambda language, source: language == LANGUAGE and source not in holdout,
            input_columns=["language", "dataset"],
            num_proc=num_proc,
        )
        if split != "test":
            train_val = smart_turn.train_test_split(test_size=VAL_FRAC, seed=SEED)
            smart_turn = train_val["train" if split == "train" else "test"]
        # Note: a ClassLabel cast rewrites the whole table (audio included) into the cache, so cast only this subset;
        # ttd's label is already LABELS
        smart_turn = smart_turn.rename_column("endpoint_bool", "label").select_columns(COLUMNS)
        parts.append(smart_turn.cast_column("label", LABELS))

    if "ttd" in datasets:
        parts.append(load_dataset(TTD, split=split, num_proc=num_proc).rename_column("data_source", "dataset"))

    # Note: the repos declare different audio features (ttd: sampling_rate 16000); an Audio cast is metadata only
    parts = [d.select_columns(COLUMNS).cast_column("audio", AUDIO) for d in parts]
    return concatenate_datasets(parts)


class EoTDataset(Dataset):
    """Clips of a split -> {"input_values": prepared (augmented) 8 s waveform, "labels": True = complete}."""

    def __init__(
        self,
        ds: HFDataset,
        augment: Compose | None = None,
        seed: int = 0,
        narrowband: bool = True,
        loudness_dbfs: float | None = -30.0,
    ):
        self.labels = np.asarray(ds["label"], dtype=bool)
        self.audio = ds.select_columns(["audio"]).cast_column("audio", AUDIO)

        self.augment = augment
        self.narrowband = narrowband
        self.loudness_dbfs = loudness_dbfs
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> dict:
        x = self.audio[i]["audio"].get_all_samples().data[0].numpy()  # (T,)

        x = prepare_clip(x, self.narrowband, self.loudness_dbfs)
        # Note: augment before padding, so the noise SNR is measured against the speech, not the zeros
        if self.augment is not None:
            x = apply_augment(self.augment, x, (self.seed, self.epoch, i))
        x = pad_window(x)  # (128000,)

        return {"input_values": x, "labels": self.labels[i]}


@dataclass
class EoTCollator:
    """Batch of EoTDataset items -> the model's inputs: {"input_features": (B, 80, frames), "labels": (B,)}."""

    feature_extractor: WhisperFeatureExtractor

    def __call__(self, items: list[dict]) -> dict[str, torch.Tensor]:
        # Note: one clip at a time: on a CPU worker the extractor's batched path is ~2.4x slower (measured)
        features = []
        for item in items:
            out = self.feature_extractor(item["input_values"], sampling_rate=SAMPLE_RATE, return_tensors="np")
            features.append(out.input_features[0])  # (80, frames)

        return {
            "input_features": torch.from_numpy(np.stack(features)),  # (B, 80, frames)
            "labels": torch.tensor([item["labels"] for item in items]),  # (B,) bool; the model casts
        }
