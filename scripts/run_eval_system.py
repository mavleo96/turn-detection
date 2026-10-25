"""The full turn-taking system on the TTD test split: the model at the VAD stop, plus a fallback silence timeout.

At each pause (a TTD clip, 0.224 s after the end of speech) the model decides with its threshold: complete -> the
turn ends there. Otherwise the turn ends once the user has been silent for T (gap_to_next_same_s >= T, as in
run_eval_vad.py), unless the user speaks again first. Per pause: FPR = incomplete pauses where the system ends the
turn (an interruption); FNR = complete pauses it never ends; response delay = time from the end of speech to the end
of turn on complete pauses it ends (0.224 s or T; inference time not included). Per turn (clips sharing conversation,
channel and start): the share of turns with at least one interruption. Rows: our model + T, Smart Turn v3.2 + T, and T
alone, for T in 0.75 / 1 / 1.5 / 2 s. Then the matched operating point: our threshold for Smart Turn's TTD FPR, chosen
on test (comparison only) and on TTD validation (needs our validation predictions: computed once, cached).

Writes to --out (default results/eval/system/): system.csv, matched.csv, summary.md (+ the cached validation
predictions). Run (from the repo root):  python scripts/run_eval_system.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset

from turn_detection.data import RESULTS_DIR, RUNS_DIR, TTD

STOP_S = 0.224  # the model decides this long after the end of speech (TTD clip end)
TIMEOUTS = [0.75, 1.0, 1.5, 2.0]
OURS = "both-8s-5ep-20261024-225310"
SMART_TURN = RESULTS_DIR / "eval" / "smart-turn-v3.2-gpu" / "predictions.parquet"


def ttd_test() -> pd.DataFrame:
    """TTD test clips: id, source, label, silence_s (end of speech to the user's next speech; inf if none), turn."""
    t = load_dataset(TTD, split="test").select_columns(
        ["id", "data_source", "label", "original_data_id", "channel", "timestamp_start", "meta"]
    )
    d = t.to_pandas().rename(columns={"data_source": "source"})
    gap = d.pop("meta").map(lambda m: json.loads(m).get("gap_to_next_same_s"))
    d["silence_s"] = gap.astype(float).fillna(np.inf)  # Note: missing = the user never speaks again (all complete)
    d["turn"] = d.groupby(["original_data_id", "channel", "timestamp_start"]).ngroup()
    return d[["id", "source", "label", "silence_s", "turn"]]


def system(clips: pd.DataFrame, model_complete: pd.Series, timeout: float | None) -> dict:
    """Per-pause and per-turn metrics of model + fallback timeout (timeout None: the model alone; model_complete all
    False: the timeout alone)."""
    # Note: None, not inf: a user who never speaks again has silence_s = inf, and inf >= inf would fire the fallback
    fallback = ~model_complete & (clips["silence_s"] >= timeout) if timeout is not None else model_complete & False
    ended = model_complete | fallback
    complete, incomplete = clips["label"] == 1, clips["label"] == 0
    delay = np.where(model_complete, STOP_S, timeout)[complete & ended]  # (complete pauses ended,)
    interrupted = (ended & incomplete).groupby(clips["turn"]).any()
    has_incomplete = incomplete.groupby(clips["turn"]).any()
    return {
        "fpr": ended[incomplete].mean(),
        "fnr": (~ended[complete]).mean(),
        "delay_mean_s": delay.mean(),
        "delay_p90_s": np.percentile(delay, 90),
        "ended_by_model": model_complete[complete & ended].mean(),
        "turns_interrupted": interrupted.mean(),
        "turns_interrupted_of_those_with_a_pause": interrupted[has_incomplete].mean(),
    }


def threshold_for_fpr(labels: np.ndarray, probs: np.ndarray, fpr: float) -> float:
    """The lowest threshold whose FPR (prob >= threshold on incomplete clips) does not exceed `fpr`."""
    neg = np.sort(probs[labels == 0])[::-1]  # (n_incomplete,) descending
    k = int(np.floor(fpr * len(neg)))  # at most k incomplete clips may be called complete
    return float(np.nextafter(neg[k], np.inf)) if k < len(neg) else float(neg[-1])


def main() -> None:
    ap = argparse.ArgumentParser(description="Model + fallback timeout on the TTD test split")
    ap.add_argument("--out", type=Path, default=RESULTS_DIR / "eval" / "system")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    clips = ttd_test()
    ours = pd.read_parquet(RESULTS_DIR / "eval" / OURS / "best" / "predictions.parquet", columns=["id", "prob"])
    st = pd.read_parquet(SMART_TURN, columns=["id", "prob"]).rename(columns={"prob": "prob_st"})
    clips = clips.merge(ours, on="id", validate="one_to_one").merge(st, on="id", validate="one_to_one")
    assert len(clips) == 60574, len(clips)
    threshold = json.loads((RUNS_DIR / OURS / "best" / "config.json").read_text())["threshold"]

    # Step 2: the system per pause and per turn
    deciders = {
        f"ours ({threshold:.4f}) + T": clips["prob"] >= threshold,
        "Smart Turn v3.2 (0.5) + T": clips["prob_st"] >= 0.5,
        "T alone": pd.Series(False, index=clips.index),
    }
    rows = []
    for name, model_complete in deciders.items():
        for t in TIMEOUTS:
            rows.append({"system": name, "T_s": t, **system(clips, model_complete, t)})
        if name != "T alone":
            rows.append({"system": name.replace(" + T", " alone"), "T_s": "-", **system(clips, model_complete, None)})
    table = pd.DataFrame(rows)
    table.to_csv(args.out / "system.csv", index=False)

    # Step 3: our threshold for Smart Turn's TTD FPR, on test and on TTD validation
    st_fpr = (clips["prob_st"] >= 0.5)[clips["label"] == 0].mean()
    labels, probs = clips["label"].to_numpy(), clips["prob"].to_numpy()
    val_file = args.out / "validation_predictions.parquet"
    if not val_file.exists():
        import sys

        from transformers import WhisperFeatureExtractor

        from turn_detection.data.dataset import load_split
        from turn_detection.model import EoTModel

        sys.path.insert(0, str(Path(__file__).parent))
        from run_eval import predict

        best = RUNS_DIR / OURS / "best"
        model, fe = EoTModel.from_pretrained(best), WhisperFeatureExtractor.from_pretrained(best)
        predict(model, fe, load_split("validation"), batch_size=128, workers=16, use_cpu=False).to_parquet(val_file)
    val = pd.read_parquet(val_file)
    val = val[val["corpus"] == "ttd"]
    rows = []
    for chosen_on, (y, p) in {
        "test (comparison only)": (labels, probs),
        "TTD validation": (val["label"].to_numpy(), val["prob"].to_numpy()),
    }.items():
        t = threshold_for_fpr(y, p, st_fpr)
        pred = probs >= t
        rows.append(
            {
                "threshold chosen on": chosen_on,
                "threshold": f"{t:.4f}",
                "test fpr": pred[labels == 0].mean(),
                "test fnr": (~pred[labels == 1]).mean(),
            }
        )
    rows.append(
        {
            "threshold chosen on": "Smart Turn v3.2 at 0.5",
            "threshold": "0.5",
            "test fpr": st_fpr,
            "test fnr": (clips["prob_st"] < 0.5)[clips["label"] == 1].mean(),
        }
    )
    rows.append(
        {
            "threshold chosen on": "ours at its validation threshold",
            "threshold": f"{threshold:.4f}",
            "test fpr": (probs >= threshold)[labels == 0].mean(),
            "test fnr": (probs < threshold)[labels == 1].mean(),
        }
    )
    matched = pd.DataFrame(rows)
    matched.to_csv(args.out / "matched.csv", index=False)

    def md(df: pd.DataFrame) -> str:
        cell = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)  # noqa: E731
        lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
        return "\n".join(lines + ["| " + " | ".join(cell(v) for v in r) + " |" for r in df.itertuples(index=False)])

    text = "\n".join(
        [
            "# Model + fallback timeout on the TTD test split\n",
            md(table),
            "\n# Matched operating point\n",
            md(matched),
            "",
        ]
    )
    (args.out / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
