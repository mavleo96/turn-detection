"""Explore the whole smart-turn-data-v3.2 dataset (both splits, every language) with Hugging Face `datasets`.

Downloads both repos into the Hugging Face cache (HF_HOME): ~46 GB of parquet, plus the Arrow copy
`datasets` prepares next to it (~90 GB in total). Reruns reuse the cache, including the audio pass.

  metadata  every row: languages, sources, labels, synthetic flag, fillers, nulls, id checks
  audio     every clip decoded once (Dataset.map, parallel, cached): duration, sample rate,
            channels, peak / clipping, trailing silence

Writes to --out (default results/):
  summary.md          readable summary, including the task spec's rough numbers vs. measured
  summary.json        headline numbers and checks
  tables/*.csv        every table in full
  clip_stats.parquet  one row per clip: metadata + audio stats, for further exploration
  *.png               languages, duration and trailing-silence plots

Run (from the repo root):  python scripts/explore_dataset.py [--num-proc 16] [--no-audio]
"""

import argparse
import json
import os
from datetime import date
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from datasets import Features, Value, load_dataset

from turn_detection.data import LANGUAGE, META_COLUMNS, RESULTS_DIR, SMART_TURN_TEST, SMART_TURN_TRAIN, WINDOW_S

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402 (after selecting the backend)

REPOS = {"train": SMART_TURN_TRAIN, "test": SMART_TURN_TEST}

FRAME_MS = 20
SILENCE_DB = -40.0  # trailing silence: frames this far below the clip's loudest frame
CLIP_LEVEL = 0.999  # |sample| at or above this counts as clipped
CLIPPED_FRAC = 0.001  # a clip is flagged when more than this share of its samples is clipped
SHORT_S = 1.0
QUIET_PEAK = 0.01  # peak below -40 dBFS: near-silent clip
AUDIO_FEATURES = Features(
    {
        "id": Value("string"),
        "sample_rate": Value("int32"),
        "channels": Value("int32"),
        "duration_s": Value("float64"),
        "peak": Value("float64"),
        "clipped_frac": Value("float64"),
        "trailing_silence_s": Value("float64"),
        "error": Value("string"),
    }
)
STAT_COLUMNS = [c for c in AUDIO_FEATURES if c != "id"]
# Rough numbers from the task spec (from a partial sample), checked against the full data.
SPEC_CLAIMS = {
    "English share": "~24%",
    "complete share (labels)": "~50%",
    "synthetic share": "~83%",
    "endfiller null share": "~20%",
    "clip length range": "0.4-33 s",
    "median clip length": "~7 s",
}


# --- per-clip audio measures ---


def trailing_silence(x: np.ndarray, sr: int, frame_ms: int = FRAME_MS, rel_db: float = SILENCE_DB) -> float:
    """Seconds of trailing audio more than `rel_db` below the loudest frame (RMS over `frame_ms` frames
    aligned to the clip end). Relative, so it measures "quiet compared to the speech"; a clip with no
    audible frame at all counts as entirely silent."""
    frame = max(1, sr * frame_ms // 1000)
    n_frames = len(x) // frame
    if n_frames == 0:
        return len(x) / sr
    frames = x[len(x) - n_frames * frame :].reshape(n_frames, frame)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    if rms.max() < 1e-6:
        return len(x) / sr
    db = 20 * np.log10(rms + 1e-12)
    last_loud = np.flatnonzero(db > db.max() + rel_db)[-1]
    return (n_frames - 1 - last_loud) * frame / sr


def clip_stats(audio) -> dict:
    """Stats of one clip from its datasets AudioDecoder (the file's own rate and channels)."""
    try:
        samples = audio.get_all_samples()
    except Exception as e:
        return {"error": repr(e)[:200]}
    sr = samples.sample_rate
    x = samples.data.numpy()  # (channels, T)
    channels = x.shape[0]
    x = x.mean(axis=0)
    absx = np.abs(x)
    n = len(x)
    return {
        "sample_rate": sr,
        "channels": channels,
        "duration_s": n / sr,
        "peak": float(absx.max()) if n else 0.0,
        "clipped_frac": float(np.mean(absx >= CLIP_LEVEL)) if n else 0.0,
        "trailing_silence_s": trailing_silence(x, sr),
        "error": None,
    }


# --- loading ---


def _audio_stats(batch: dict) -> dict:
    rows = [clip_stats(audio) for audio in batch["audio"]]
    return {"id": batch["id"], **{c: [r.get(c) for r in rows] for c in STAT_COLUMNS}}


def audio_stats(ds, num_proc: int) -> pd.DataFrame:
    """Per-clip audio stats in dataset order, decoded by `datasets` (torchcodec)."""
    mapped = ds.select_columns(["id", "audio"]).map(
        _audio_stats,
        batched=True,
        batch_size=64,
        num_proc=num_proc,
        remove_columns=["audio"],
        features=AUDIO_FEATURES,
        desc="audio stats",
    )
    return mapped.to_pandas()


def load_all(num_proc: int | None, with_audio: bool) -> pd.DataFrame:
    frames = []
    for split, repo in REPOS.items():
        ds = load_dataset(repo, split="train", num_proc=num_proc)
        meta = ds.select_columns(META_COLUMNS).to_pandas()
        if with_audio:
            stats = audio_stats(ds, num_proc)
            if not stats["id"].equals(meta["id"]):
                raise RuntimeError("audio stats are not in dataset order")
            meta = pd.concat([meta, stats.drop(columns="id")], axis=1)
        frames.append(meta.assign(split=split))
    df = pd.concat(frames, ignore_index=True)
    df["label"] = df["endpoint_bool"].map({True: "complete", False: "incomplete"}).fillna("null")
    df["kind"] = df["synthetic"].map({True: "synthetic", False: "human"}).fillna("null")
    return df


# --- tables ---


def share(s: pd.Series) -> float:
    return round(float(s.mean()), 4) if len(s) else float("nan")


def true_share(s: pd.Series) -> float:
    """Share of values that are True (nulls count as not True)."""
    return share(s.eq(True))


def duration_table(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    g = df.groupby(by)["duration_s"]
    t = g.describe(percentiles=[0.05, 0.5, 0.95])[["count", "mean", "min", "5%", "50%", "95%", "max"]]
    t[f"> {WINDOW_S} s"] = g.apply(lambda v: share(v > WINDOW_S))
    return t.round(3)


def build_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """All tables, keyed by name. `df`: one row per clip (META_COLUMNS + split, label, kind [+ audio stats])."""
    has_audio = "duration_s" in df
    eng = df[df["language"] == LANGUAGE]
    splits = list(dict.fromkeys(df["split"]))
    t = {}

    overview = df.groupby("split", sort=False).agg(
        rows=("id", "size"),
        languages=("language", "nunique"),
        sources=("dataset", "nunique"),
        english_share=("language", lambda s: share(s == LANGUAGE)),
        complete_share=("endpoint_bool", true_share),
        synthetic_share=("synthetic", true_share),
    )
    if has_audio:
        overview["hours"] = (df.groupby("split", sort=False)["duration_s"].sum() / 3600).round(1)
    t["overview"] = overview

    lang = pd.crosstab(df["language"], df["split"]).reindex(columns=splits, fill_value=0)
    lang["share"] = (lang.sum(axis=1) / len(df)).round(4)
    lang["complete_share"] = df.groupby("language")["endpoint_bool"].apply(true_share)
    lang["synthetic_share"] = df.groupby("language")["synthetic"].apply(true_share)
    lang["sources"] = df.groupby("language")["dataset"].nunique()
    if has_audio:
        lang["median_duration_s"] = df.groupby("language")["duration_s"].median().round(2)
    t["languages"] = lang.sort_values("share", ascending=False)

    src = pd.crosstab(df["dataset"], df["split"]).reindex(columns=splits, fill_value=0)
    src["languages"] = df.groupby("dataset")["language"].nunique()
    src["english"] = eng.groupby("dataset").size().reindex(src.index, fill_value=0)
    src["synthetic_share"] = df.groupby("dataset")["synthetic"].apply(true_share)
    src["complete_share"] = df.groupby("dataset")["endpoint_bool"].apply(true_share)
    t["sources"] = src.sort_values(splits[0], ascending=False)
    t["source_by_language"] = pd.crosstab(df["dataset"], df["language"])

    eng_src = pd.crosstab(eng["dataset"], eng["split"]).reindex(columns=splits, fill_value=0)
    eng_src["synthetic_share"] = eng.groupby("dataset")["synthetic"].apply(true_share)
    for split in splits:
        g = eng[eng["split"] == split]
        eng_src[f"complete_share_{split}"] = g.groupby("dataset")["endpoint_bool"].apply(true_share)
    t["english_sources"] = eng_src.sort_values(splits[0], ascending=False)
    t["english_label_by_kind"] = pd.crosstab([eng["split"], eng["label"]], eng["kind"], margins=True)

    split_label = (df["split"] + " / " + df["label"]).rename("split / label")
    eng_split_label = (eng["split"] + " / " + eng["label"]).rename("split / label")
    for col in ("midfiller", "endfiller"):
        t[f"{col}_all"] = pd.crosstab(df[col].map(str), split_label)
        t[f"{col}_english"] = pd.crosstab(eng[col].map(str), eng_split_label)

    t["nulls"] = pd.DataFrame(
        {split: df[df["split"] == split][META_COLUMNS].isna().sum() for split in splits}
    ).rename_axis("column")

    if has_audio:
        ok = df[df["error"].isna()]
        ok_eng = ok[ok["language"] == LANGUAGE]
        t["duration_by_language"] = duration_table(ok, ["language"]).sort_values("count", ascending=False)
        t["duration_english"] = duration_table(ok_eng, ["split", "label", "kind"])
        t["duration_english_by_source"] = duration_table(ok_eng, ["dataset"]).sort_values("count", ascending=False)

        g = ok_eng.groupby(["label", "kind"])["trailing_silence_s"]
        trail = g.describe(percentiles=[0.1, 0.5, 0.9])[["count", "10%", "50%", "90%"]].round(3)
        for ms in (100, 200, 500):
            trail[f">= {ms} ms"] = g.apply(lambda v, s=ms / 1000: share(v >= s))
        t["trailing_silence_english"] = trail
        t["trailing_silence_english_by_source"] = ok_eng.pivot_table(
            "trailing_silence_s", "dataset", "label", aggfunc="median"
        ).round(3)
        t["trailing_silence_by_language"] = ok.pivot_table(
            "trailing_silence_s", "language", "label", aggfunc="median"
        ).round(3)

        checks = {
            "decode error": df["error"].notna(),
            "empty audio": df["duration_s"] == 0,
            f"very short (< {SHORT_S:g} s)": df["duration_s"] < SHORT_S,
            f"clipping (> {100 * CLIPPED_FRAC:g}% of samples)": df["clipped_frac"] > CLIPPED_FRAC,
            f"near-silent (peak < {QUIET_PEAK})": df["peak"] < QUIET_PEAK,
            "not mono": df["channels"] > 1,
        }
        rows = []
        for name, mask in checks.items():
            mask = mask.fillna(False).astype(bool)
            row = {"check": name}
            for split in splits:
                in_split = df["split"] == split
                row[f"{split} (all)"] = int((mask & in_split).sum())
                row[f"{split} (eng)"] = int((mask & in_split & (df["language"] == LANGUAGE)).sum())
            row["example ids"] = ", ".join(df.loc[mask, "id"].head(3))
            rows.append(row)
        t["anomalies"] = pd.DataFrame(rows).set_index("check")
        t["sample_rates"] = pd.crosstab(df["sample_rate"].astype("Int64"), df["split"])
        t["channels"] = pd.crosstab(df["channels"].astype("Int64"), df["split"])
    return t


def id_checks(df: pd.DataFrame) -> dict:
    by_split = {split: g["id"] for split, g in df.groupby("split", sort=False)}
    names = list(by_split)
    return {
        "duplicate_ids_within_split": {s: int(ids.duplicated().sum()) for s, ids in by_split.items()},
        "ids_in_both_splits": int(len(set(by_split[names[0]]) & set(by_split[names[-1]]))) if len(names) > 1 else 0,
        "spoken_text_non_null": int(df["spoken_text"].notna().sum()),
    }


def spec_check(df: pd.DataFrame) -> pd.DataFrame:
    measured = {
        "English share": f"{100 * share(df['language'] == LANGUAGE):.1f}%",
        "complete share (labels)": f"{100 * true_share(df['endpoint_bool']):.1f}%",
        "synthetic share": f"{100 * true_share(df['synthetic']):.1f}%",
        "endfiller null share": f"{100 * share(df['endfiller'].isna()):.1f}%",
    }
    if "duration_s" in df:
        d = df["duration_s"].dropna()
        measured["clip length range"] = f"{d.min():.2f}-{d.max():.1f} s"
        measured["median clip length"] = f"{d.median():.2f} s"
    return pd.DataFrame(
        [{"claim": k, "spec": v, "measured (all rows)": measured.get(k, "–")} for k, v in SPEC_CLAIMS.items()]
    ).set_index("claim")


# --- output ---


def md_table(df: pd.DataFrame, index: bool = True) -> str:
    df = df.reset_index() if index else df

    def cell(v):
        if isinstance(v, (bool, np.bool_)):
            return str(v)
        if isinstance(v, float):
            return "–" if np.isnan(v) else f"{v:,.3g}" if abs(v) < 100 else f"{v:,.0f}"
        return f"{v:,}" if isinstance(v, (int, np.integer)) else str(v)

    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def plots(df: pd.DataFrame, out: Path) -> list[str]:
    names = []
    fig, ax = plt.subplots(figsize=(11, 4.5))
    counts = pd.crosstab(df["language"], df["split"])
    counts.loc[counts.sum(axis=1).sort_values(ascending=False).index].plot.bar(stacked=True, ax=ax)
    ax.set_ylabel("clips")
    ax.set_title("Clips per language")
    names.append(_save(fig, out, "languages.png"))

    if "duration_s" not in df:
        return names
    ok = df[df["error"].isna()]
    eng = ok[ok["language"] == LANGUAGE]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    bins = np.arange(0, min(ok["duration_s"].max(), 35) + 0.5, 0.5)
    for ax, (title, d) in zip(axes, (("all languages", ok), ("English", eng))):
        for label, g in d.groupby("label"):
            ax.hist(g["duration_s"].clip(upper=bins[-1]), bins=bins, alpha=0.6, label=label)
        ax.axvline(WINDOW_S, color="k", ls="--", lw=1, label=f"{WINDOW_S} s model window")
        ax.set_title(f"Clip duration by label ({title})")
        ax.set_xlabel("duration (s)")
        ax.legend()
    names.append(_save(fig, out, "duration_hist.png"))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    bins = np.arange(0, 2.02, 0.02)
    for ax, (kind, g) in zip(axes, eng.groupby("kind")):
        for label, gg in g.groupby("label"):
            ax.hist(gg["trailing_silence_s"].clip(upper=bins[-1]), bins=bins, alpha=0.6, label=label)
        ax.set_title(f"Trailing silence by label (English, {kind})")
        ax.set_xlabel("trailing silence (s, clipped at 2 s)")
        ax.legend()
    names.append(_save(fig, out, "trailing_silence_hist.png"))
    return names


def _save(fig, out: Path, name: str) -> str:
    fig.tight_layout()
    fig.savefig(out / name, dpi=110)
    plt.close(fig)
    return name


SUMMARY_SECTIONS = [
    ("Overview", "overview", None),
    ("Languages (top 25)", "languages", 25),
    ("Sources", "sources", None),
    ("English: sources", "english_sources", None),
    ("English: labels by human / synthetic", "english_label_by_kind", None),
    ("English: midfiller (incl. nulls)", "midfiller_english", None),
    ("English: endfiller (incl. nulls)", "endfiller_english", None),
    ("Null counts", "nulls", None),
    ("Duration by language (top 25)", "duration_by_language", 25),
    ("English: duration by split, label, kind", "duration_english", None),
    ("English: duration by source", "duration_english_by_source", None),
    ("English: trailing silence by label and kind", "trailing_silence_english", None),
    ("English: median trailing silence by source", "trailing_silence_english_by_source", None),
    ("Anomalies", "anomalies", None),
    ("Sample rates", "sample_rates", None),
    ("Channels", "channels", None),
]


def write_results(df: pd.DataFrame, out: Path) -> None:
    (out / "tables").mkdir(parents=True, exist_ok=True)
    tables = build_tables(df)
    for name, table in tables.items():
        table.to_csv(out / "tables" / f"{name}.csv")
    checks = id_checks(df)
    claims = spec_check(df)
    images = plots(df, out)
    df.to_parquet(out / "clip_stats.parquet", index=False)

    summary = {
        "generated": str(date.today()),
        "repos": REPOS,
        "rows": len(df),
        "overview": tables["overview"].reset_index().to_dict("records"),
        "checks": checks,
        "spec_vs_measured": claims["measured (all rows)"].to_dict(),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))

    s = ["# smart-turn-data-v3.2: dataset exploration\n"]
    s.append(
        f"Generated by `python scripts/explore_dataset.py` on {date.today()} from {len(df):,} rows "
        f"({', '.join(REPOS.values())}). Full tables: `tables/*.csv`; per clip: `clip_stats.parquet`.\n"
    )
    s.append("## Task spec vs. measured\n")
    s.append(md_table(claims) + "\n")
    s.append("## Id and text checks\n")
    s.append(
        f"- Duplicate ids within a split: {checks['duplicate_ids_within_split']}\n"
        f"- Ids present in both splits: {checks['ids_in_both_splits']}\n"
        f"- Rows with non-empty `spoken_text`: {checks['spoken_text_non_null']}\n"
    )
    for title, name, top in SUMMARY_SECTIONS:
        if name in tables:
            table = tables[name] if top is None else tables[name].head(top)
            s.append(f"## {title}\n")
            s.append(md_table(table) + "\n")
    s += [f"![{name}]({name})\n" for name in images]
    s.append("## Known limitations\n")
    s.append(
        "- **Possible leakage across splits:** `spoken_text` is empty, so complete and truncated versions of the "
        "same sentence cannot be grouped and may land in different splits (train vs val, or train vs test). "
        "Val/test scores may be optimistic.\n"
        "- **Domain gap:** the data is clean recorded or TTS speech, not phone calls.\n"
        f"- **Trailing silence** is energy-based and relative to each clip's loudest frame ({SILENCE_DB:g} dB, "
        f"{FRAME_MS} ms frames), so noisy recordings read as less silent than clean TTS.\n"
    )
    (out / "summary.md").write_text("\n".join(s))


def main() -> None:
    ap = argparse.ArgumentParser(description="Explore smart-turn-data-v3.2 with Hugging Face datasets -> results/")
    ap.add_argument("--out", type=Path, default=RESULTS_DIR)
    ap.add_argument("--num-proc", type=int, default=min(16, os.cpu_count() or 1), help="download / map processes")
    ap.add_argument("--no-audio", dest="audio", action="store_false", help="metadata only (skip decoding)")
    args = ap.parse_args()

    df = load_all(args.num_proc, args.audio)
    args.out.mkdir(parents=True, exist_ok=True)
    write_results(df, args.out)
    print(f"wrote {args.out / 'summary.md'} ({len(df):,} rows)")


if __name__ == "__main__":
    main()
