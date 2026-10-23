"""Train / val / test manifests and leave-one-source-out folds of the English clips (ids + metadata, never audio).

  test        the official test split (English), untouched
  train/val   90/10 of the official train split, stratified by (endpoint_bool, synthetic, dataset);
              strata with fewer than --min-stratum clips go entirely to train
  --subset N  stratified subsample of train (same strata, proportional allocation)
  folds       for each human source with >= --fold-min-clips English clips in the train split:
              folds/<source>/heldout.parquet = all of that source's train-split clips,
              folds/<source>/{train,val}.parquet = the train/val manifests without that source;
              --holdout A B adds a fold holding out sources A and B together (folds/A+B/)

Everything is deterministic: rows are sorted by id, strata are visited in sorted order, and one
seeded generator is used throughout.

Run (from the repo root):  python scripts/split.py [--seed 42] [--subset N] [--holdout A B ...]
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from data.constants import FOLDS_DIR, MANIFESTS_DIR, SEED
from data.dataset import load_metadata

MANIFEST_COLUMNS = ["id", "endpoint_bool", "synthetic", "midfiller", "endfiller", "dataset"]


def human_sources(meta: pd.DataFrame) -> list[str]:
    """Sources whose clips are mostly real human speech (synthetic == False)."""
    human_share = (meta["synthetic"] == False).groupby(meta["dataset"]).mean()  # noqa: E712 (nullable column)
    return sorted(human_share[human_share > 0.5].index)


def strata_key(df: pd.DataFrame) -> pd.Series:
    # map(str), not astype(str): pandas 3 keeps missing values missing under astype(str), and groupby
    # would silently drop those rows; map(str) turns them into "None".
    return df["endpoint_bool"].map(str) + "|" + df["synthetic"].map(str) + "|" + df["dataset"].map(str)


def _strata(df: pd.DataFrame) -> list[np.ndarray]:
    """Row positions of each stratum, strata in sorted key order."""
    groups = df.groupby(strata_key(df), sort=True).indices
    return [groups[k] for k in sorted(groups)]


def stratified_split(
    df: pd.DataFrame, val_frac: float = 0.1, seed: int = SEED, min_stratum: int = 10
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(train, val): round(val_frac * n) random clips of every stratum with >= min_stratum clips go to val."""
    df = df.sort_values("id", ignore_index=True)
    rng = np.random.default_rng(seed)
    is_val = np.zeros(len(df), dtype=bool)
    for idx in _strata(df):
        if len(idx) >= min_stratum:
            is_val[rng.permutation(idx)[: int(round(len(idx) * val_frac))]] = True
    return df[~is_val].reset_index(drop=True), df[is_val].reset_index(drop=True)


def stratified_subsample(df: pd.DataFrame, n: int, seed: int = SEED) -> pd.DataFrame:
    """Exactly `n` clips, allocated to strata in proportion to their size (largest remainder)."""
    df = df.sort_values("id", ignore_index=True)
    if n >= len(df):
        return df
    strata = _strata(df)
    sizes = np.array([len(idx) for idx in strata])
    quota = sizes * n / sizes.sum()
    take = np.floor(quota).astype(int)
    take[np.argsort(-(quota - take), kind="stable")[: n - take.sum()]] += 1
    rng = np.random.default_rng(seed)
    picked = np.concatenate([rng.permutation(idx)[:k] for idx, k in zip(strata, take)])
    return df.iloc[np.sort(picked)].reset_index(drop=True)


def loso_folds(
    train: pd.DataFrame,
    val: pd.DataFrame,
    full_train: pd.DataFrame,
    min_clips: int = 300,
    holdouts: list[list[str]] | None = None,
) -> dict[str, dict[str, pd.DataFrame]]:
    """One fold per human source with >= min_clips clips in the official train split, plus one fold per
    extra group of sources in `holdouts` (held out together; fold name "<a>+<b>")."""
    counts = full_train["dataset"].value_counts()
    groups = [[source] for source in human_sources(full_train) if counts.get(source, 0) >= min_clips]
    for group in holdouts or []:
        unknown = set(group) - set(counts.index)
        if unknown:
            raise ValueError(f"unknown sources in --holdout: {sorted(unknown)}")
        groups.append(sorted(group))
    folds = {}
    for group in groups:
        folds["+".join(group)] = {
            "train": train[~train["dataset"].isin(group)].reset_index(drop=True),
            "val": val[~val["dataset"].isin(group)].reset_index(drop=True),
            "heldout": full_train[full_train["dataset"].isin(group)].reset_index(drop=True),
        }
    return folds


def build_splits(
    full_train: pd.DataFrame,
    test: pd.DataFrame,
    val_frac: float = 0.1,
    seed: int = SEED,
    subset: int | None = None,
    min_stratum: int = 10,
    fold_min_clips: int = 300,
    holdouts: list[list[str]] | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, pd.DataFrame]], dict]:
    """(splits, folds, summary). Ids that also occur in the test split are dropped from the train side."""
    overlap = set(full_train["id"]) & set(test["id"])
    full_train = full_train[~full_train["id"].isin(overlap)].sort_values("id", ignore_index=True)
    test = test.sort_values("id", ignore_index=True)

    train, val = stratified_split(full_train, val_frac, seed, min_stratum)
    if subset is not None:
        train = stratified_subsample(train, subset, seed)
    splits = {"train": train, "val": val, "test": test}
    folds = loso_folds(train, val, full_train, fold_min_clips, holdouts)

    summary = {
        "seed": seed,
        "val_frac": val_frac,
        "min_stratum": min_stratum,
        "subset": subset,
        "train_test_id_overlap_dropped": len(overlap),
        "clips": {name: len(df) for name, df in splits.items()},
        "complete_share": {name: round(float(df["endpoint_bool"].mean()), 4) for name, df in splits.items()},
        "folds": {src: {part: len(df) for part, df in f.items()} for src, f in folds.items()},
    }
    return splits, folds, summary


def write_outputs(splits: dict[str, pd.DataFrame], folds: dict, summary: dict, out: Path, folds_out: Path) -> None:
    """Manifests (ids + metadata) to `out`, fold id lists to `folds_out/<source>/`, and the summary."""
    out.mkdir(parents=True, exist_ok=True)
    for name, df in splits.items():
        df[MANIFEST_COLUMNS].to_parquet(out / f"{name}.parquet", index=False)
    # Folds are fully regenerated, so folds from an earlier run (other seed/subset) can't linger.
    shutil.rmtree(folds_out, ignore_errors=True)
    for source, parts in folds.items():
        (folds_out / source).mkdir(parents=True)
        for part, df in parts.items():
            df[["id"]].to_parquet(folds_out / source / f"{part}.parquet", index=False)
    (out / "split_summary.json").write_text(json.dumps(summary, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description="Write train/val/test manifests and leave-one-source-out folds")
    ap.add_argument("--out", type=Path, default=MANIFESTS_DIR)
    ap.add_argument("--folds-out", type=Path, default=FOLDS_DIR)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--subset", type=int, default=None, help="stratified subsample of train (default: all)")
    ap.add_argument("--min-stratum", type=int, default=10)
    ap.add_argument("--fold-min-clips", type=int, default=300)
    ap.add_argument(
        "--holdout",
        nargs="+",
        action="append",
        metavar="SOURCE",
        help="extra fold holding out these sources together (repeatable), e.g. --holdout human_5 rime_2",
    )
    args = ap.parse_args()

    full_train = load_metadata("train")  # English rows, from the Hugging Face cache
    test = load_metadata("test")
    splits, folds, summary = build_splits(
        full_train, test, args.val_frac, args.seed, args.subset, args.min_stratum, args.fold_min_clips, args.holdout
    )

    write_outputs(splits, folds, summary, args.out, args.folds_out)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
