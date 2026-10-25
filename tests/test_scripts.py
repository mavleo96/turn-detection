"""The scripts end to end on tiny in-memory splits (no network)."""

import numpy as np
import pandas as pd

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
    splits = pd.read_csv(tmp_path / "tables" / "splits_by_corpus.csv")
    # fake hub: 64 English smart-turn train clips (9 from the holdout source rime_2) + 24 test; ttd 40 / 16 / 20
    counts = {(s, c): n for s, c, n in zip(splits["split"], splits["corpus"], splits["clips"])}
    assert counts[("test", "ttd")] == 20 and counts[("validation", "ttd")] == 16
    assert (
        counts[("train", "smart-turn")]
        + counts[("validation", "smart-turn")]
        + counts[("excluded (holdout)", "smart-turn")]
        == 64
    )


def test_samples(hf_splits, tmp_path):
    csv = export_samples(tmp_path / "samples", per_bucket=3, seed=0)
    assert len(csv) == 4 * 3 and csv["bucket"].nunique() == 4 and set(csv["split"]) <= {"train", "test"}
    assert all((tmp_path / "samples" / f).exists() for f in csv["file"])


def test_vad_timeout_endpointer():
    from run_eval_vad import vad_metrics

    # complete turns: the user stays silent (or never speaks again); incomplete: the user speaks again after 0.4 s
    clips = pd.DataFrame({"label": [1, 1, 0, 0], "silence_s": [2.0, np.inf, 0.4, 0.4]})
    assert vad_metrics(clips, 0.3)["fpr"] == 1.0  # fires before the user resumes: interrupts both
    m = vad_metrics(clips, 0.5)
    assert m["fpr"] == 0.0 and m["fnr"] == 0.0 and m["n"] == 4 and m["complete"] == 0.5
    assert vad_metrics(clips, 3.0)["fnr"] == 0.5  # waits past the 2 s silence of a complete turn


def test_model_plus_fallback_timeout():
    from run_eval_system import system, threshold_for_fpr

    # turn 0: an incomplete pause (user resumes after 0.4 s), then the complete one (never speaks again)
    # turn 1: one complete pause, the user's next speech 3 s later
    clips = pd.DataFrame({"label": [0, 1, 1], "silence_s": [0.4, np.inf, 3.0], "turn": [0, 0, 1]})
    model = pd.Series([False, False, True])
    alone = system(clips, model, None)  # no fallback: a user who never speaks again isn't "ended" by inf >= inf
    assert alone["fnr"] == 0.5 and alone["fpr"] == 0.0 and alone["delay_mean_s"] == 0.224
    m = system(clips, model, 0.3)  # the fallback fires on the incomplete pause (0.4 >= 0.3) and on the complete one
    assert m["fpr"] == 1.0 and m["fnr"] == 0.0 and m["turns_interrupted"] == 0.5
    assert m["delay_mean_s"] == np.mean([0.3, 0.224]) and m["turns_interrupted_of_those_with_a_pause"] == 1.0
    assert system(clips, model, 0.5)["fpr"] == 0.0  # the user resumes first

    labels, probs = np.array([0, 0, 0, 0, 1]), np.array([0.1, 0.2, 0.3, 0.9, 0.8])
    t = threshold_for_fpr(labels, probs, 0.25)  # at most 1 of 4 incomplete clips may score >= t
    assert (probs[labels == 0] >= t).mean() <= 0.25 and (probs[labels == 0] >= t).mean() == 0.25
