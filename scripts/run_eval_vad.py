"""A plain VAD endpointer on the TTD test split, as a baseline: end of turn once the user has been silent for T seconds.

TTD clips stop where the dataset's VAD fired, 0.224 s after the end of speech, and their metadata has the time from the
end of speech to the user's next speech (gap_to_next_same_s; none if the user doesn't speak again). A silence-timeout
endpointer with timeout T calls the turn complete iff the user stays silent for T (gap >= T), and decides T after the
end of speech; our models decide at the clip end, 0.224 s after it (plus inference). TTD's label is defined by who
speaks next, so a long enough T approaches it by construction: the baseline is about the latency a VAD needs for a
given error rate, not about whether it can be accurate. smart-turn has no timing metadata, so this is TTD only.

Writes to --out (default results/eval/vad-timeout/):
  metrics.csv  per timeout and group (ttd, spokenwoz, seamless): n, complete share, accuracy, f1, fnr, fpr
  metrics.md   the same, plus the timeout a VAD needs to interrupt (FPR) as rarely as each evaluated model

Run (from the repo root):  python scripts/run_eval_vad.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset

from turn_detection.data import RESULTS_DIR, TTD
from turn_detection.train.metrics import eot_metrics

STOP_S = 0.224  # TTD clips end this long after the end of speech
TIMEOUTS = [STOP_S, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
COLUMNS = ["accuracy", "f1", "fnr", "fpr"]
# Our models and Smart Turn, from run_eval.py / run_eval_baseline.py: their FPR / FNR on TTD at their thresholds
EVALS = {
    "both-8s": "both-8s-5ep-20261024-225310/best",
    "ttd-8s": "ttd-8s-5ep-20261024-232232/best",
    "smart-turn-8s": "smart-turn-8s-5ep-20261024-233637/best",
    "Smart Turn v3.2": "smart-turn-v3.2-gpu",
}


def vad_metrics(clips: pd.DataFrame, timeout: float) -> dict:
    """Metrics of a silence-timeout endpointer on `clips` (columns: label, silence_s)."""
    pred = (clips["silence_s"] >= timeout).astype(float)  # 1 = complete: the user stayed silent for `timeout`
    m = eot_metrics(clips["label"].to_numpy(), pred.to_numpy(), threshold=0.5)
    return {"n": len(clips), "complete": clips["label"].mean(), **{k: m[k] for k in COLUMNS}}


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a silence-timeout VAD endpointer on the TTD test split")
    ap.add_argument("--out", type=Path, default=RESULTS_DIR / "eval" / "vad-timeout")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    test = load_dataset(TTD, split="test").select_columns(["data_source", "label", "meta"]).to_pandas()
    gap = test.pop("meta").map(lambda m: json.loads(m).get("gap_to_next_same_s"))
    clips = test.rename(columns={"data_source": "source"}).assign(silence_s=gap.astype(float).fillna(np.inf))
    groups = {"ttd": clips} | dict(list(clips.groupby("source")))

    rows = [
        {"timeout_s": t, "added_latency_s": t - STOP_S, "name": name, **vad_metrics(g, t)}
        for t in TIMEOUTS
        for name, g in groups.items()
    ]
    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.out / "metrics.csv", index=False)

    # The timeout at which the VAD interrupts (FPR) no more often than each model, on all of TTD
    fine = np.round(np.arange(STOP_S, 10.0, 0.01), 3)
    curve = pd.DataFrame([{"timeout_s": t, **vad_metrics(clips, t)} for t in fine])
    matches = []
    for model, run in EVALS.items():
        m = pd.read_csv(RESULTS_DIR / "eval" / run / "metrics.csv").set_index("name").loc["ttd"]
        vad = curve[curve["fpr"] <= m["fpr"]].iloc[0]
        matches.append(
            {
                "model": model,
                "model fpr": m["fpr"],
                "model fnr": m["fnr"],
                "VAD timeout for that fpr (s)": vad["timeout_s"],
                "VAD added latency (s)": vad["timeout_s"] - STOP_S,
                "VAD fnr there": vad["fnr"],
            }
        )

    def table(df: pd.DataFrame) -> list[str]:
        lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
        cell = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)  # noqa: E731
        return lines + ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]

    lines = [
        "# Silence-timeout VAD endpointer on the TTD test split\n",
        f"Turn complete iff the user stays silent for the timeout; TTD clips (and our models' decisions) come "
        f"{STOP_S} s after the end of speech. Added latency = timeout - {STOP_S} s.\n",
        "## By timeout\n",
        *table(metrics),
        "\n## The timeout a VAD needs to interrupt as rarely as each model (all of TTD)\n",
        *table(pd.DataFrame(matches)),
    ]
    (args.out / "metrics.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
