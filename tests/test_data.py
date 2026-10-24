import numpy as np
import pytest
import torch
from transformers import WhisperFeatureExtractor

from turn_detection.data.audio import crop_window, decode, narrowband, normalize_loudness, pad_window
from turn_detection.data.constants import LANGUAGE, SAMPLE_RATE, WINDOW_S
from turn_detection.data.dataset import EoTCollator, EoTDataset, load_metadata, load_split, rows_for_ids
from turn_detection.data.transforms import build_augment

from .helpers import fake_hf_split, flac_bytes


def smart_turn_dataset(*args, **kwargs) -> EoTDataset:
    return EoTDataset(*args, label_column="endpoint_bool", **kwargs)


N = WINDOW_S * SAMPLE_RATE


def window(x: np.ndarray) -> np.ndarray:
    return pad_window(crop_window(x))


def test_crop_and_pad_window():
    x = np.arange(N + 5000, dtype=np.float32)
    np.testing.assert_array_equal(crop_window(x), x[-N:])
    np.testing.assert_array_equal(window(x), x[-N:])
    short = window(np.ones(1000, dtype=np.float32))
    assert short.shape == (N,) and not short[:-1000].any() and (short[-1000:] == 1).all()
    assert crop_window(np.ones(1000, dtype=np.float32)).shape == (1000,)  # crop never pads
    exact = np.random.default_rng(0).standard_normal(N).astype(np.float32)
    np.testing.assert_array_equal(window(exact), exact)
    assert not window(np.zeros(0, dtype=np.float32)).any()


def test_narrowband_removes_content_above_4khz():
    t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    low = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
    high = np.sin(2 * np.pi * 6000 * t).astype(np.float32)
    for x in (low, high, low[:-1]):  # also an odd length
        assert narrowband(x).shape == x.shape and narrowband(x).dtype == np.float32
    middle = slice(1000, -1000)  # away from the resampler's edges
    assert np.abs(narrowband(low)[middle] - low[middle]).max() < 0.01
    assert np.abs(narrowband(high)[middle]).max() < 0.01
    assert narrowband(np.zeros(0, dtype=np.float32)).shape == (0,)


def test_decode_downmixes_to_mono():
    t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    stereo = np.stack([np.sin(2 * np.pi * 440 * t), np.zeros_like(t)], axis=1) * 0.5
    x, sr = decode(flac_bytes(stereo, SAMPLE_RATE))
    assert sr == SAMPLE_RATE and x.shape == (SAMPLE_RATE,) and x.dtype == np.float32
    assert 0.2 < np.abs(x).max() < 0.3  # mean of the two channels


def test_feature_extractor_is_sized_for_the_window():
    fe = WhisperFeatureExtractor(chunk_length=WINDOW_S)
    feats = fe(np.zeros(N, dtype=np.float32), sampling_rate=SAMPLE_RATE, return_tensors="np").input_features
    assert feats.shape == (1, 80, 800) and fe.n_samples == N


def test_load_metadata_english_sorted(hf_splits):
    meta = load_metadata("train")
    assert (meta["language"] == LANGUAGE).all() and len(meta) == 64 and meta["id"].is_monotonic_increasing


def test_rows_for_ids_and_dataset_items(hf_splits):
    source = hf_splits["train"]
    assert rows_for_ids(source, ["train-005", "train-002"]).tolist() == [5, 2]
    with pytest.raises(KeyError):
        rows_for_ids(source, ["nope"])
    ds = smart_turn_dataset(source, ["train-005", "train-002"], build_augment(noise_p=1.0), seed=0)
    assert ds.labels.tolist() == [0.0, 1.0]  # endpoint_bool = (i % 2 == 0)
    item = ds[1]
    x = item["input_values"]
    assert x.shape == (N,) and x.dtype == np.float32 and item["labels"] == 1.0
    assert np.array_equal(ds[1]["input_values"], x)  # same epoch, same augmentation
    ds.set_epoch(1)
    assert not np.array_equal(ds[1]["input_values"], x)  # new augmentation each epoch


def test_dataset_without_augment_is_the_last_window(hf_splits):
    # train-001: 1.5 s of speech + 250 ms of silence, left-padded to 8 s
    ds = smart_turn_dataset(hf_splits["train"], ["train-001"])
    x = ds[0]["input_values"]
    assert not x[: N - 28000].any() and np.abs(x[-28000:-4000]).max() > 0.1
    ds.set_epoch(3)
    assert np.array_equal(ds[0]["input_values"], x)


def test_collator_extracts_features_per_batch(hf_splits):
    ds = smart_turn_dataset(hf_splits["train"], ["train-001", "train-002", "train-003"])
    fe = WhisperFeatureExtractor(chunk_length=WINDOW_S)
    batch = EoTCollator(fe)([ds[0], ds[1], ds[2]])
    assert batch["input_features"].shape == (3, 80, 800) and batch["input_features"].dtype == torch.float32
    assert batch["labels"].tolist() == [0.0, 1.0, 0.0] and batch["labels"].dtype == torch.float32
    single = fe(ds[1]["input_values"], sampling_rate=SAMPLE_RATE, return_tensors="np").input_features[0]
    np.testing.assert_allclose(batch["input_features"][1].numpy(), single, atol=1e-5)  # same as one at a time


def test_dataset_rejects_other_sample_rates():
    ds = smart_turn_dataset(fake_hf_split(3, "x", sample_rate=8000), ["x-001"])
    with pytest.raises(ValueError, match="16000 Hz"):
        ds[0]


def test_dataset_noise_level_ignores_padding(hf_splits):
    # train-001 is 1.75 s, so 6.25 s of the window is padding: noise must be scaled to the 1.75 s of audio
    augment = build_augment(gain_db=0.0, noise_p=1.0, snr_db=(20.0, 20.0))
    clean = smart_turn_dataset(hf_splits["train"], ["train-001"])[0]["input_values"]
    noisy = smart_turn_dataset(hf_splits["train"], ["train-001"], augment)[0]["input_values"]
    assert not noisy[: N - 28000].any()  # padding is added after the noise, so it stays silent
    audio, noise = clean[-28000:], noisy[-28000:] - clean[-28000:]
    snr = 20 * np.log10(np.sqrt(np.mean(audio**2)) / np.sqrt(np.mean(noise**2)))
    assert abs(snr - 20) < 0.5


def test_dataset_narrowband_flag(hf_splits):
    full = smart_turn_dataset(hf_splits["train"], ["train-002"], narrowband=False)[0]["input_values"]
    band = smart_turn_dataset(hf_splits["train"], ["train-002"])[0]["input_values"]
    spectrum_full = np.abs(np.fft.rfft(full)) ** 2
    spectrum_band = np.abs(np.fft.rfft(band)) ** 2
    above_4k = np.fft.rfftfreq(N, 1 / SAMPLE_RATE) > 4200
    assert spectrum_band[above_4k].sum() < 1e-3 * spectrum_full[above_4k].sum()


def test_ttd_split_and_labels(ttd_splits):
    ds = load_split("train", dataset="ttd")
    assert len(ds) == 40 and set(ds["data_source"]) == {"spokenwoz", "seamless"}
    items = EoTDataset(ds, ["spokenwoz_train_001", "seamless_train_030"])  # label_column "label" (ClassLabel)
    assert items.labels.tolist() == [1.0, 0.0] and items[0]["input_values"].shape == (N,)


def test_normalize_loudness():
    rng = np.random.default_rng(0)
    for level in (0.001, 0.05, 0.5):  # -60 / -26 / -6 dBFS-ish inputs all end at the target
        x = (level * rng.standard_normal(SAMPLE_RATE)).astype(np.float32)
        y = normalize_loudness(x, -30.0)
        assert y.dtype == np.float32 and 20 * np.log10(np.sqrt(np.mean(y**2))) == pytest.approx(-30.0, abs=0.01)
    silent = np.full(1000, 1e-5, dtype=np.float32)  # -100 dBFS: left alone, not blown up
    np.testing.assert_array_equal(normalize_loudness(silent), silent)
    spike = np.zeros(SAMPLE_RATE, dtype=np.float32)
    spike[0] = 0.5  # reaching -30 dBFS RMS would need a peak far above 1: scaled to peak 1 instead
    assert np.abs(normalize_loudness(spike)).max() == pytest.approx(1.0)
    assert normalize_loudness(np.zeros(0, dtype=np.float32)).shape == (0,)


def test_dataset_items_are_loudness_normalized(hf_splits):
    for loudness in (-30.0, -20.0):
        x = smart_turn_dataset(hf_splits["train"], ["train-003"], loudness_dbfs=loudness)[0]["input_values"]
        audio = x[np.flatnonzero(x)[0] :]  # without the left padding
        assert 20 * np.log10(np.sqrt(np.mean(audio**2))) == pytest.approx(loudness, abs=0.05)
    raw = smart_turn_dataset(hf_splits["train"], ["train-003"], narrowband=False, loudness_dbfs=None)[0]
    assert np.abs(raw["input_values"]).max() > 0.5  # off: the fake clip's own level (0.3 std noise)
