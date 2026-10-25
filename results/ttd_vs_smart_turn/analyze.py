"""Every number in FINDINGS.md, from collect.py's output (features.parquet, layer_weights.json), the test evaluations
(results/eval) and the exploration tables. Writes tables.md next to this script; with --previous <dir>, also checks
the clip sample and the data-only measures against an earlier collect.py output (features.parquet, vad_check.parquet,
smart_turn_asr.parquet).

Run (from the repo root):  python results/ttd_vs_smart_turn/analyze.py [--previous artifacts/ttd_vs_st_2026-10-24]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import roc_auc_score

from turn_detection.data import RESULTS_DIR, TTD

OUT = Path(__file__).parent
GROUPS = ["smart-turn", "spokenwoz", "seamless"]
MODELS = {"ttd": "TTD only", "st": "smart-turn only", "both": "both", "smart_turn_v32": "Smart Turn v3.2"}
RUNS = {
    "ttd": "ttd-8s-5ep-20261024-232232/best",
    "st": "smart-turn-8s-5ep-20261024-233637/best",
    "both": "both-8s-5ep-20261024-225310/best",
    "smart_turn_v32": "smart-turn-v3.2-gpu",
}
CUES = {
    "duration_s": "clip length",
    "vad_trailing_s": "VAD trailing non-speech",
    "energy_end_slope_db_per_s": "energy slope, last 300 ms of speech",
    "f0_end_vs_median_st": "final pitch vs the speaker's median",
    "vad_snr_db": "speech vs non-speech level",
}


def auc(labels: pd.Series, values: pd.Series) -> float:
    ok = values.notna()
    return roc_auc_score(labels[ok], values[ok]) if labels[ok].nunique() == 2 else np.nan


def finished(text) -> float:
    """1 if a transcript ends a sentence (. ? !), 0 if not, NaN without a transcript."""
    if not isinstance(text, str) or not text.strip():
        return np.nan
    return float(text.strip()[-1] in ".?!")


def md(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    df = df.reset_index()
    cell = lambda v: floatfmt.format(v) if isinstance(v, float) else str(v)  # noqa: E731
    rows = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    return "\n".join(rows + ["| " + " | ".join(cell(v) for v in r) + " |" for r in df.itertuples(index=False)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--previous", type=Path, default=None, help="an earlier collect.py output to compare against")
    args = ap.parse_args()

    f = pd.read_parquet(OUT / "features.parquet")
    stats = f[~f.showcase]  # the 300 random clips per group; the showcase clips are for reading
    by = {g: stats[stats.group == g] for g in GROUPS}
    s = []

    # Test ROC AUC of the models, by corpus and source (whole test split)
    auc_test = pd.DataFrame(
        {
            MODELS[m]: pd.read_csv(RESULTS_DIR / "eval" / r / "metrics.csv").set_index("name")["roc_auc"]
            for m, r in RUNS.items()
        }
    )
    rows = [
        "all",
        "ttd",
        "spokenwoz",
        "seamless",
        "smart-turn",
        "liva_1",
        "chirp3_1",
        "midcentury_1",
        "human_5",
        "rime_2",
    ]
    s += ["## Test ROC AUC (whole test split)\n", md(auc_test.loc[rows]), ""]

    kinds = pd.read_csv(RESULTS_DIR / "data_exploration" / "tables" / "english_label_by_kind.csv")
    test = kinds[kinds.split == "test"]
    human = test.human.sum() / (test.human.sum() + test.synthetic.sum())
    s += [f"English smart-turn test: {human:.1%} human speech.\n"]

    # 1. Clip endings and length
    rows = {}
    for g, d in by.items():
        t = d.vad_trailing_s
        rows[g] = {
            "VAD trailing p10": t.quantile(0.1),
            "p50": t.median(),
            "p90": t.quantile(0.9),
            "no speech found": d.vad_trailing_s.isna().mean(),
            "length p50": d.duration_s.median(),
        }
    s += ["## 1. Clip endings (s; 300 clips per group)\n", md(pd.DataFrame(rows).T), ""]

    # 2. Single cues
    rows = {g: {CUES[c]: auc(d.label, d[c]) for c in CUES} for g, d in by.items()}
    s += ["## 2. Single-cue label AUC (0.5 = none, SE ~0.03)\n", md(pd.DataFrame(rows)), ""]
    sw = by["spokenwoz"]
    pitch = sw.groupby("label").f0_end_vs_median_st.agg(["median", "mean"])
    yes, no = pitch.loc[1], pitch.loc[0]
    s += [
        f"SpokenWOZ final pitch vs the speaker's median (semitones): complete median {yes['median']:.2f} "
        f"(mean {yes['mean']:.2f}), incomplete median {no['median']:.2f} (mean {no['mean']:.2f}).\n"
    ]

    # 3. Transcripts: final punctuation (smart-turn: Whisper Tiny; Seamless: dataset; SpokenWOZ: turn-level, skipped)
    rows = {}
    for g in ("smart-turn", "seamless"):
        d = by[g].assign(finished=by[g].text.map(finished)).dropna(subset=["finished"])
        unfinished, done = d[d.finished == 0], d[d.finished == 1]
        rows[g] = {
            "clips": len(d),
            "AUC": auc(d.label, d.finished),
            "unfinished": len(unfinished),
            "unfinished but complete": unfinished.label.mean(),
            "finished but incomplete": 1 - done.label.mean(),
        }
    # Note: Seamless has a transcript for every clip, so also measure it on its whole test split (the sample is noisy)
    meta = load_dataset(TTD, split="test").select_columns(["data_source", "label", "meta"]).to_pandas()
    seamless = meta[meta.data_source == "seamless"].assign(
        text=lambda d: d.meta.map(lambda m: json.loads(m).get("text"))
    )
    d = seamless.assign(finished=seamless.text.map(finished)).dropna(subset=["finished"])
    rows["seamless, whole test"] = {
        "clips": len(d),
        "AUC": auc(d.label, d.finished),
        "unfinished": int((d.finished == 0).sum()),
        "unfinished but complete": d[d.finished == 0].label.mean(),
        "finished but incomplete": 1 - d[d.finished == 1].label.mean(),
    }
    s += ["## 3. Final punctuation as a cue\n", md(pd.DataFrame(rows).T), ""]

    # 4. The models
    weights = json.loads((OUT / "layer_weights.json").read_text())
    s += [
        "## 4. Models\n",
        md(pd.DataFrame(weights, index=[f"layer {i}" for i in range(1, 5)]).T.rename(index=MODELS)),
        "",
    ]
    peak = {
        MODELS[m]: {g: d[f"attn_peak_s_before_end_{m}"].median() for g, d in by.items()} for m in ("ttd", "st", "both")
    }
    s += ["Attention peak, s before the clip end (median)\n", md(pd.DataFrame(peak)), ""]

    st = by["smart-turn"]
    rows = {
        MODELS[m]: {
            "rho with VAD trailing": st[f"prob_{m}"].corr(st.vad_trailing_s, method="spearman"),
            "rho with length": st[f"prob_{m}"].corr(st.duration_s, method="spearman"),
        }
        for m in MODELS
    }
    s += [
        "On smart-turn: Spearman rho of P(complete) with cues that don't predict its labels\n",
        md(pd.DataFrame(rows).T),
        "",
    ]

    rows = {MODELS[m]: {g: d[f"prob_{m}"].mean() for g, d in by.items()} for m in MODELS}
    calib = pd.DataFrame(rows).T
    calib.loc["actual complete share"] = {g: d.label.mean() for g, d in by.items()}
    s += ["Mean P(complete) vs the actual complete share (300 clips per group)\n", md(calib), ""]

    # Reproduction check against an earlier run of collect.py
    if args.previous:
        old = pd.read_parquet(args.previous / "features.parquet")
        same = set(old.id) == set(f.id) and len(old) == len(f)
        m = f.merge(old, on="id", suffixes=("", "_old"))
        cols = [
            "duration_s",
            "trailing_silence_s",
            "snr_db",
            "hf_share",
            "energy_end_slope_db_per_s",
            "f0_end_vs_median_st",
        ]
        diff = {c: float(np.nanmax(np.abs(m[c] - m[f"{c}_old"]))) for c in cols}
        vad = pd.read_parquet(args.previous / "vad_check.parquet")[
            ["id", "vad_trailing_s", "vad_snr_db", "nonspeech_s"]
        ]
        v = f.merge(vad, on="id", suffixes=("", "_old"))
        vad_cmp = {
            c: {
                "rho": v[c].corr(v[f"{c}_old"], method="spearman"),
                "median |diff|": (v[c] - v[f"{c}_old"]).abs().median(),
                "new p50": v[c].median(),
                "old p50": v[f"{c}_old"].median(),
            }
            for c in ["vad_trailing_s", "vad_snr_db", "nonspeech_s"]
        }
        asr = pd.read_parquet(args.previous / "smart_turn_asr.parquet")[["id", "asr"]].merge(f[["id", "text"]], on="id")
        same_text = (asr.asr.str.strip() == asr.text.str.strip()).mean()
        same_end = (asr.asr.map(finished) == asr.text.map(finished)).mean()
        s += [
            f"## Check against {args.previous}\n",
            f"Same clips: {same} ({len(f)}). Acoustic features, max |new - old|: "
            + ", ".join(f"{k} {v:.2g}" for k, v in diff.items())
            + ".\n",
            "VAD measures, new (Silero 6.2.3, threshold 0.5, min silence 100 ms) vs old (settings not kept)\n",
            md(pd.DataFrame(vad_cmp).T),
            "",
            f"Whisper Tiny transcripts of the {len(asr)} smart-turn clips: identical text {same_text:.1%}, same "
            f"finished / unfinished call {same_end:.1%}.\n",
        ]

    (OUT / "tables.md").write_text("# TTD vs smart-turn: tables (analyze.py)\n\n" + "\n".join(s) + "\n")
    print("\n".join(s))


if __name__ == "__main__":
    main()
