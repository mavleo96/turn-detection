"""The scripts end to end on tiny in-memory splits (no network)."""

import numpy as np

from explore_dataset import load_all, trailing_silence, write_results
from samples import export_samples


def test_trailing_silence():
    sr = 16000
    speech = 0.3 * np.random.default_rng(0).standard_normal(sr)
    assert trailing_silence(np.concatenate([speech, np.zeros(sr // 2)]), sr) == 0.5
    assert trailing_silence(speech, sr) == 0.0
    assert trailing_silence(np.zeros(sr), sr) == 1.0


def test_explore_end_to_end(hf_splits, tmp_path):
    df = load_all(num_proc=None, with_audio=True)
    assert len(df) == 110 and df["error"].isna().all()
    assert set(df["sample_rate"]) == {16000}
    write_results(df, tmp_path)
    text = (tmp_path / "summary.md").read_text()
    assert "## Known limitations" in text and "## English: trailing silence by label and kind" in text


def test_samples(hf_splits, tmp_path):
    csv = export_samples(tmp_path / "samples", per_bucket=3, seed=0)
    assert len(csv) == 4 * 3 and csv["bucket"].nunique() == 4 and set(csv["split"]) <= {"train", "test"}
    assert all((tmp_path / "samples" / f).exists() for f in csv["file"])
