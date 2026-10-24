import numpy as np
import pytest
import torch
from datasets import Dataset
from transformers import WhisperFeatureExtractor

from turn_detection.data import HOLDOUT, LANGUAGE, SAMPLE_RATE, SEED, VAL_FRAC, WINDOW_S
from turn_detection.data.audio import band_limit, normalize_loudness, pad_window, prepare_clip
from turn_detection.data.dataset import EoTCollator, EoTDataset, load_split
from turn_detection.data.transforms import build_augment

from .helpers import fake_hf_split, tiny_model


def smart_turn_dataset(ds, rows: list[int], *args, **kwargs) -> EoTDataset:
    """EoTDataset on `rows` of a raw fake smart-turn split, endpoint_bool as the label."""
    return EoTDataset(ds.select(rows).rename_column("endpoint_bool", "label"), *args, **kwargs)


N = WINDOW_S * SAMPLE_RATE


def crop(x: np.ndarray) -> np.ndarray:
    return prepare_clip(x, narrowband=False, loudness_dbfs=None)


def window(x: np.ndarray) -> np.ndarray:
    return pad_window(crop(x))


def test_crop_and_pad_window():
    x = np.arange(N + 5000, dtype=np.float32)
    np.testing.assert_array_equal(crop(x), x[-N:])
    np.testing.assert_array_equal(window(x), x[-N:])
    short = window(np.ones(1000, dtype=np.float32))
    assert short.shape == (N,) and not short[:-1000].any() and (short[-1000:] == 1).all()
    assert crop(np.ones(1000, dtype=np.float32)).shape == (1000,)  # crop never pads
    exact = np.random.default_rng(0).standard_normal(N).astype(np.float32)
    np.testing.assert_array_equal(window(exact), exact)
    assert not window(np.zeros(0, dtype=np.float32)).any()


def test_narrowband_removes_content_above_4khz():
    t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    low = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
    high = np.sin(2 * np.pi * 6000 * t).astype(np.float32)
    for x in (low, high, low[:-1]):  # also an odd length
        assert band_limit(x).shape == x.shape and band_limit(x).dtype == np.float32
    middle = slice(1000, -1000)  # away from the resampler's edges
    assert np.abs(band_limit(low)[middle] - low[middle]).max() < 0.01
    assert np.abs(band_limit(high)[middle]).max() < 0.01
    assert band_limit(np.zeros(0, dtype=np.float32)).shape == (0,)


def test_feature_extractor_is_sized_for_the_window():
    fe = WhisperFeatureExtractor(chunk_length=WINDOW_S)
    feats = fe(np.zeros(N, dtype=np.float32), sampling_rate=SAMPLE_RATE, return_tensors="np").input_features
    assert feats.shape == (1, 80, 800) and fe.n_samples == N


def test_dataset_items(hf_splits):
    ds = smart_turn_dataset(hf_splits["train"], [5, 2], build_augment(noise_p=1.0), seed=0)
    assert ds.labels.tolist() == [0.0, 1.0]  # endpoint_bool = (i % 2 == 0)
    item = ds[1]
    x = item["input_values"]
    assert x.shape == (N,) and x.dtype == np.float32 and item["labels"] == 1.0
    assert np.array_equal(ds[1]["input_values"], x)  # same epoch, same augmentation
    ds.set_epoch(1)
    assert not np.array_equal(ds[1]["input_values"], x)  # new augmentation each epoch


def test_dataset_without_augment_is_the_last_window(hf_splits):
    # train-001: 1.5 s of speech + 250 ms of silence, left-padded to 8 s
    ds = smart_turn_dataset(hf_splits["train"], [1])
    x = ds[0]["input_values"]
    assert not x[: N - 28000].any() and np.abs(x[-28000:-4000]).max() > 0.1
    ds.set_epoch(3)
    assert np.array_equal(ds[0]["input_values"], x)


@pytest.mark.parametrize("window_s", [2, 4, 8])
def test_window_sizes_agree(hf_splits, window_s):
    # items, features and the model all follow window_s
    ds = smart_turn_dataset(hf_splits["train"], [1, 2], window_s=window_s)
    fe = WhisperFeatureExtractor(chunk_length=window_s)
    batch = EoTCollator(fe)([ds[0], ds[1]])
    assert ds[0]["input_values"].shape == (window_s * SAMPLE_RATE,)
    assert batch["input_features"].shape == (2, 80, window_s * 100)
    assert tiny_model(window_s=window_s)(**batch).logits.shape == (2,)


def test_collator_rejects_audio_of_another_window(hf_splits):
    # the extractor would silently keep the first 2 s of an 8 s window, without the pause at its end
    ds = smart_turn_dataset(hf_splits["train"], [1])
    with pytest.raises(ValueError, match="expects 32000"):
        EoTCollator(WhisperFeatureExtractor(chunk_length=2))([ds[0]])


def test_collator_extracts_features_per_batch(hf_splits):
    ds = smart_turn_dataset(hf_splits["train"], [1, 2, 3])
    fe = WhisperFeatureExtractor(chunk_length=WINDOW_S)
    batch = EoTCollator(fe)([ds[0], ds[1], ds[2]])
    assert batch["input_features"].shape == (3, 80, 800) and batch["input_features"].dtype == torch.float32
    assert batch["labels"].tolist() == [0.0, 1.0, 0.0] and batch["labels"].dtype == torch.bool
    single = fe(ds[1]["input_values"], sampling_rate=SAMPLE_RATE, return_tensors="np").input_features[0]
    np.testing.assert_allclose(batch["input_features"][1].numpy(), single, atol=1e-5)  # same as one at a time


def test_dataset_resamples_to_16k():
    ds = smart_turn_dataset(fake_hf_split(3, "x", sample_rate=8000), [1], narrowband=False, loudness_dbfs=None)
    x = ds[0]["input_values"]  # x-001: 1.75 s, so 28000 samples at 16 kHz after the padding
    assert not x[: N - 28000].any() and x[N - 28000] != 0


def test_dataset_noise_level_ignores_padding(hf_splits):
    # train-001 is 1.75 s, so 6.25 s of the window is padding: noise must be scaled to the 1.75 s of audio
    augment = build_augment(gain_db=0.0, noise_p=1.0, snr_db=(20.0, 20.0))
    clean = smart_turn_dataset(hf_splits["train"], [1])[0]["input_values"]
    noisy = smart_turn_dataset(hf_splits["train"], [1], augment)[0]["input_values"]
    assert not noisy[: N - 28000].any()  # padding is added after the noise, so it stays silent
    audio, noise = clean[-28000:], noisy[-28000:] - clean[-28000:]
    snr = 20 * np.log10(np.sqrt(np.mean(audio**2)) / np.sqrt(np.mean(noise**2)))
    assert abs(snr - 20) < 0.5


def test_dataset_narrowband_flag(hf_splits):
    full = smart_turn_dataset(hf_splits["train"], [2], narrowband=False)[0]["input_values"]
    band = smart_turn_dataset(hf_splits["train"], [2])[0]["input_values"]
    spectrum_full = np.abs(np.fft.rfft(full)) ** 2
    spectrum_band = np.abs(np.fft.rfft(band)) ** 2
    above_4k = np.fft.rfftfreq(N, 1 / SAMPLE_RATE) > 4200
    assert spectrum_band[above_4k].sum() < 1e-3 * spectrum_full[above_4k].sum()


def test_load_split(hub):
    splits = {name: load_split(name) for name in ("train", "validation", "test")}
    for ds in splits.values():
        assert ds.column_names == ["id", "audio", "dataset", "label"]
        assert ds.features["label"].names == ["incomplete", "complete"]

    # smart-turn: English only; train / validation are a random 90/10 split without the holdout sources
    def smart_turn_ids(ds):
        return {i for i, source in zip(ds["id"], ds["dataset"]) if source not in ("spokenwoz", "seamless")}

    raw = hub["smart-turn"]["train"]
    kept = {
        i for i, lang, src in zip(raw["id"], raw["language"], raw["dataset"]) if lang == LANGUAGE and src not in HOLDOUT
    }
    train, val = smart_turn_ids(splits["train"]), smart_turn_ids(splits["validation"])
    assert train | val == kept and not train & val and len(val) == round(0.1 * len(kept))
    assert set(splits["test"]["dataset"]) & set(HOLDOUT) == {"rime_2"}  # holdout sources only in test
    raw_test = hub["smart-turn"]["test"]
    assert smart_turn_ids(splits["test"]) == {
        i for i, lang in zip(raw_test["id"], raw_test["language"]) if lang == LANGUAGE
    }

    # ttd: its own splits; labels follow endpoint_bool / ttd's ClassLabel
    assert [sum(s in ("spokenwoz", "seamless") for s in ds["dataset"]) for ds in splits.values()] == [40, 16, 20]
    label_of = dict(zip(raw["id"], raw["endpoint_bool"])) | dict(
        zip(hub["ttd"]["train"]["id"], hub["ttd"]["train"]["label"])
    )
    assert all(label == int(label_of[i]) for i, label in zip(splits["train"]["id"], splits["train"]["label"]))

    items = EoTDataset(splits["train"])
    assert items.labels.dtype == bool and len(items) == len(splits["train"]) and items[0]["input_values"].shape == (N,)


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
        x = smart_turn_dataset(hf_splits["train"], [3], loudness_dbfs=loudness)[0]["input_values"]
        audio = x[np.flatnonzero(x)[0] :]  # without the left padding
        assert 20 * np.log10(np.sqrt(np.mean(audio**2))) == pytest.approx(loudness, abs=0.05)
    raw = smart_turn_dataset(hf_splits["train"], [3], narrowband=False, loudness_dbfs=None)[0]
    assert np.abs(raw["input_values"]).max() > 0.5  # off: the fake clip's own level (0.3 std noise)


def test_load_split_selects_datasets(hub):
    assert set(load_split("test", ["ttd"])["dataset"]) == {"spokenwoz", "seamless"}
    assert not {"spokenwoz", "seamless"} & set(load_split("validation", ["smart-turn"])["dataset"])
    with pytest.raises(ValueError, match="datasets must be"):
        load_split("train", ["nope"])


def test_smart_turn_val_is_split_from_hub_order(hub):
    # Note: the only shuffle is train_test_split's, on the filtered rows in Hub order
    raw = hub["smart-turn"]["train"]
    kept = [
        i for i, lang, src in zip(raw["id"], raw["language"], raw["dataset"]) if lang == LANGUAGE and src not in HOLDOUT
    ]
    expected = Dataset.from_dict({"id": kept}).train_test_split(test_size=VAL_FRAC, seed=SEED)
    assert load_split("validation", ["smart-turn"])["id"] == expected["test"]["id"]
    assert load_split("train", ["smart-turn"])["id"] == expected["train"]["id"]
