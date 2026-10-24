import json

import numpy as np
import pandas as pd

from explore_dataset import build_tables, id_checks, spec_check, write_results


def fake_df(with_audio: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            "id": [f"id-{i}" for i in range(n)],
            "language": np.where(np.arange(n) % 4 == 0, "eng", "spa"),
            "endpoint_bool": np.arange(n) % 2 == 0,
            "midfiller": [None if i % 5 == 0 else bool(i % 2) for i in range(n)],
            "endfiller": [None if i % 5 == 0 else bool(i % 3) for i in range(n)],
            "synthetic": np.arange(n) % 3 != 0,
            "spoken_text": [None] * n,
            "dataset": np.where(np.arange(n) % 3 != 0, "chirp", "human_1"),
            "split": np.where(np.arange(n) < 160, "train", "test"),
        }
    )
    df["label"] = df["endpoint_bool"].map({True: "complete", False: "incomplete"})
    df["kind"] = df["synthetic"].map({True: "synthetic", False: "human"})
    if with_audio:
        df["sample_rate"] = 16000
        df["channels"] = 1
        df["duration_s"] = rng.uniform(0.4, 20, n)
        df["peak"] = rng.uniform(0.005, 1.0, n)
        df["clipped_frac"] = 0.0
        df["trailing_silence_s"] = np.where(df["endpoint_bool"], 0.3, 0.05)
        df["error"] = None
    return df


def test_tables_and_checks():
    df = fake_df()
    t = build_tables(df)
    assert t["overview"].loc["train", "rows"] == 160 and t["overview"].loc["train", "english_share"] == 0.25
    assert t["languages"].loc["eng", "train"] + t["languages"].loc["eng", "test"] == 50
    assert "None" in t["endfiller_english"].index  # nulls are counted, not dropped
    assert t["trailing_silence_english"].loc[("complete", "human"), "50%"] == 0.3
    assert set(t["anomalies"].index) >= {"decode error", "near-silent (peak < 0.01)"}
    assert id_checks(df) == {
        "duplicate_ids_within_split": {"train": 0, "test": 0},
        "ids_in_both_splits": 0,
        "spoken_text_non_null": 0,
    }
    assert spec_check(df).loc["English share", "measured (all rows)"] == "25.0%"


def test_metadata_only_tables():
    t = build_tables(fake_df(with_audio=False))
    assert "overview" in t and "duration_by_language" not in t and "anomalies" not in t


def test_write_results(tmp_path):
    write_results(fake_df(), tmp_path)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["rows"] == 200
    assert (tmp_path / "tables" / "languages.csv").exists() and (tmp_path / "duration_hist.png").exists()
    text = (tmp_path / "summary.md").read_text()
    assert "## Task spec vs. measured" in text and "## Anomalies" in text
    assert len(pd.read_parquet(tmp_path / "clip_stats.parquet")) == 200
