"""smart-turn train / val / test manifests of the English clips (ids + metadata, never audio).

  test        the official test split (English), untouched
  train/val   the official train split sorted by id, last --val-frac (10%) to val. Ids are random UUIDs, so this
              is a random split that doesn't depend on the Hub's row order. Ids also in test are dropped.

Run (from the repo root):  python scripts/split.py [--val-frac 0.1]
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from turn_detection.data.constants import MANIFESTS_DIR
from turn_detection.data.dataset import load_metadata

MANIFEST_COLUMNS = ["id", "endpoint_bool", "synthetic", "midfiller", "endfiller", "dataset"]


def build_splits(full_train: pd.DataFrame, test: pd.DataFrame, val_frac: float = 0.1) -> dict[str, pd.DataFrame]:
    """{train, val, test}; train/val is a cut of the official train split sorted by id."""
    test = test.sort_values("id", ignore_index=True)
    full_train = full_train[~full_train["id"].isin(test["id"])].sort_values("id", ignore_index=True)

    n_val = round(len(full_train) * val_frac)
    n_train = len(full_train) - n_val

    return {
        "train": full_train.iloc[:n_train].reset_index(drop=True),
        "val": full_train.iloc[n_train:].reset_index(drop=True),
        "test": test,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Write the smart-turn train/val/test manifests")
    ap.add_argument("--out", type=Path, default=MANIFESTS_DIR)
    ap.add_argument("--val-frac", type=float, default=0.1)
    args = ap.parse_args()

    splits = build_splits(load_metadata("train"), load_metadata("test"), args.val_frac)

    args.out.mkdir(parents=True, exist_ok=True)
    for name, df in splits.items():
        df[MANIFEST_COLUMNS].to_parquet(args.out / f"{name}.parquet", index=False)

    summary = {
        name: {"clips": len(df), "complete_share": round(float(df["endpoint_bool"].mean()), 4)}
        for name, df in splits.items()
    }
    (args.out / "split_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
