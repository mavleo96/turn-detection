"""Export random English clips per (label x synthetic) bucket as WAV files, plus a metadata CSV, for listening.

The data-task spec calls this step inspect.py; it is named samples.py so it can't shadow Python's
`inspect`. Clips are drawn from train + test and written at their original sample rate, untouched.

Run (from the repo root):  python scripts/samples.py [--per-bucket 15] [--seed 42]
  writes results/samples/<bucket>/<nn>_<id>.wav and results/samples/samples.csv
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from datasets import load_dataset

from turn_detection.data import LANGUAGE, META_COLUMNS, RESULTS_DIR, SEED, SMART_TURN_TEST, SMART_TURN_TRAIN


def load_metadata(ds) -> pd.DataFrame:
    """English metadata of a raw smart-turn split (no audio), sorted by id."""
    meta = ds.select_columns(META_COLUMNS).to_pandas()
    return meta[meta["language"] == LANGUAGE].sort_values("id", ignore_index=True)


def bucket_name(endpoint: bool, synthetic: bool) -> str:
    return f"{'complete' if endpoint else 'incomplete'}_{'synthetic' if synthetic else 'human'}"


def pick_samples(meta: pd.DataFrame, per_bucket: int = 15, seed: int = SEED) -> pd.DataFrame:
    """Up to `per_bucket` random rows per (endpoint_bool, synthetic) bucket, deterministic for a seed."""
    meta = meta.sort_values("id", ignore_index=True)
    rng = np.random.default_rng(seed)
    picks = []
    for (endpoint, synthetic), g in meta.groupby(["endpoint_bool", "synthetic"], sort=True):
        take = np.sort(rng.permutation(len(g))[:per_bucket])
        picks.append(g.iloc[take].assign(bucket=bucket_name(endpoint, synthetic)))
    return pd.concat(picks, ignore_index=True)


def export_samples(out_dir: Path, per_bucket: int, seed: int) -> pd.DataFrame:
    splits = {
        "train": load_dataset(SMART_TURN_TRAIN, split="train"),
        "test": load_dataset(SMART_TURN_TEST, split="train"),
    }
    meta = pd.concat([load_metadata(ds).assign(split=split) for split, ds in splits.items()], ignore_index=True)
    picks = pick_samples(meta, per_bucket, seed)
    picks["n"] = picks.groupby("bucket").cumcount()
    for bucket in picks["bucket"].unique():  # regenerate buckets so an earlier run's files can't linger
        shutil.rmtree(out_dir / bucket, ignore_errors=True)
        (out_dir / bucket).mkdir(parents=True)

    files, durations, rates = {}, {}, {}
    for split, g in picks.groupby("split", sort=True):
        ds = splits[split]
        for i, row in zip(g.index, pd.Index(ds["id"]).get_indexer(g["id"])):
            samples = ds[int(row)]["audio"].get_all_samples()  # the file's own rate and channels
            x, sr = samples.data.numpy().T, samples.sample_rate  # (T, channels)
            rel = Path(picks.at[i, "bucket"]) / f"{picks.at[i, 'n']:02d}_{picks.at[i, 'id']}.wav"
            sf.write(out_dir / rel, x, sr)
            files[i], durations[i], rates[i] = str(rel), round(len(x) / sr, 2), sr

    picks["file"] = pd.Series(files)
    picks["duration_s"] = pd.Series(durations)
    picks["sample_rate"] = pd.Series(rates)
    columns = ["bucket", "file", "id", "split", "dataset", "endpoint_bool", "synthetic", "midfiller", "endfiller"]
    csv = picks.sort_values(["bucket", "n"])[columns + ["duration_s", "sample_rate"]]
    csv.to_csv(out_dir / "samples.csv", index=False)
    return csv


def main() -> None:
    ap = argparse.ArgumentParser(description="Export listening samples per (label x synthetic) bucket")
    ap.add_argument("--out", type=Path, default=RESULTS_DIR / "samples")
    ap.add_argument("--per-bucket", type=int, default=15)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    csv = export_samples(args.out, args.per_bucket, args.seed)
    print(f"wrote {len(csv)} clips to {args.out}")


if __name__ == "__main__":
    main()
