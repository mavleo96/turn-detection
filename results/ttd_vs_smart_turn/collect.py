"""Collect per-clip features for the TTD vs smart-turn comparison: 20 showcase clips per dataset plus a larger
random sample per group (smart-turn / spokenwoz / seamless; fixed seeds, so always the same clips). Per clip: acoustic
features, Silero VAD, the models' predictions (from results/eval) and attention, transcripts (TTD: the dataset's;
smart-turn: Whisper Tiny). Writes features.parquet, layer_weights.json and spec_*.png next to this script.

Run (from the repo root):  python results/ttd_vs_smart_turn/collect.py"""

import io
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from datasets import Audio, load_dataset
from scipy.signal import stft

from turn_detection.data import LANGUAGE, RESULTS_DIR, SMART_TURN_TEST, TTD
from turn_detection.data.audio import pad_window, prepare_clip
from turn_detection.model import EoTModel

OUT = Path(__file__).parent
RUNS = Path("/workspace/turn-detection/artifacts/runs")
# Our 8 s models (frozen encoder, 5 epochs) by training data; checkpoints in artifacts/, predictions in results/eval/
MODELS = {
    "ttd": "ttd-8s-5ep-20261024-232232",
    "st": "smart-turn-8s-5ep-20261024-233637",
    "both": "both-8s-5ep-20261024-225310",
}
BASELINE = RESULTS_DIR / "eval" / "smart-turn-v3.2-gpu" / "predictions.parquet"  # Pipecat's Smart Turn v3.2
SR = 16000
BIG_N = 300


def frame_db(x: np.ndarray, frame: int = 320) -> np.ndarray:
    """20 ms frame energies in dB."""
    n = len(x) // frame
    frames = x[: n * frame].reshape(n, frame)
    return 10 * np.log10(np.mean(frames**2, axis=1) + 1e-12)


def pitch_track(x: np.ndarray, fmin: float = 60, fmax: float = 400, frame: int = 640, hop: int = 160):
    """F0 per 10 ms (40 ms frames) by normalized autocorrelation; NaN where unvoiced (peak < 0.5)."""
    lags = np.arange(int(SR / fmax), int(SR / fmin) + 1)
    f0 = []
    for start in range(0, len(x) - frame, hop):
        w = x[start : start + frame] * np.hanning(frame)
        w = w - w.mean()
        energy = np.dot(w, w)
        if energy < 1e-8:
            f0.append(np.nan)
            continue
        ac = np.correlate(w, w, mode="full")[frame - 1 :]
        r = ac[lags] / energy
        best = int(np.argmax(r))
        f0.append(SR / lags[best] if r[best] > 0.5 else np.nan)
    return np.array(f0)


def acoustic_features(x: np.ndarray) -> dict:
    db = frame_db(x)
    loud = np.percentile(db, 95)
    speech = db > loud - 30  # frames within 30 dB of the loud part count as speech

    # Trailing silence: frames at the end below the speech gate
    last_speech = np.flatnonzero(speech)[-1] if speech.any() else 0
    trailing_s = (len(db) - 1 - last_speech) * 0.02

    feats = {
        "duration_s": len(x) / SR,
        "trailing_silence_s": trailing_s,
        "speech_share": speech.mean(),
        "snr_db": loud - np.percentile(db, 10),
    }

    # Bandwidth: energy share above 4 kHz
    spec = np.abs(np.fft.rfft(x[-8 * SR :])) ** 2
    freqs = np.fft.rfftfreq(len(x[-8 * SR :]), 1 / SR)
    feats["hf_share"] = spec[freqs > 4000].sum() / (spec.sum() + 1e-12)

    # Ending prosody on the last 1.5 s of speech (before the trailing silence)
    end = int((last_speech + 1) * 0.02 * SR)
    seg = x[max(0, end - int(1.5 * SR)) : end]
    if len(seg) > 0.3 * SR:
        f0 = pitch_track(seg)
        t = np.arange(len(f0)) * 0.01
        v = np.isfinite(f0)
        if v.sum() >= 10:
            semis = 12 * np.log2(f0[v] / np.median(f0[v]))
            tv = t[v]
            last = tv >= tv[-1] - 0.4  # the last 400 ms of voiced speech
            feats["f0_end_slope_st_per_s"] = np.polyfit(tv[last], semis[last], 1)[0] if last.sum() >= 5 else np.nan
            feats["f0_end_vs_median_st"] = np.mean(semis[last][-5:])
        seg_db = frame_db(seg, 160)
        tail = seg_db[-30:]  # last 300 ms of speech
        feats["energy_end_slope_db_per_s"] = np.polyfit(np.arange(len(tail)) * 0.01, tail, 1)[0]
    return feats


_vad = {}


def vad_features(x: np.ndarray) -> dict:
    """Silero VAD (threshold 0.5, min silence 100 ms, no padding): non-speech after the last speech, total non-speech,
    speech vs non-speech level."""
    from silero_vad import get_speech_timestamps, load_silero_vad

    if "model" not in _vad:
        torch.set_num_threads(1)
        _vad["model"] = load_silero_vad()
    ts = get_speech_timestamps(
        torch.from_numpy(x),
        _vad["model"],
        sampling_rate=SR,
        threshold=0.5,
        min_silence_duration_ms=100,
        speech_pad_ms=0,
    )
    speech = np.zeros(len(x), dtype=bool)
    for t in ts:
        speech[t["start"] : t["end"]] = True
    feats = {"nonspeech_s": (~speech).sum() / SR, "vad_trailing_s": np.nan, "vad_snr_db": np.nan}
    if ts:
        feats["vad_trailing_s"] = (len(x) - ts[-1]["end"]) / SR
    if speech.any() and (~speech).any():
        level = lambda v: 10 * np.log10(np.mean(np.square(v, dtype=np.float64)) + 1e-12)  # noqa: E731
        feats["vad_snr_db"] = level(x[speech]) - level(x[~speech])
    return feats


def decode(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    """FLAC bytes -> mono float32 (channel mean) at the file's rate, with soundfile, as when FINDINGS.md was written."""
    x, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)  # (T, channels)
    return x.mean(axis=1), sr


def features_for(item: tuple[str, bytes]) -> dict:
    clip_id, audio_bytes = item
    x, sr = decode(audio_bytes)
    assert sr == SR
    return {"id": clip_id, **acoustic_features(x), **vad_features(x)}


@torch.no_grad()
def attention(models: dict, clips: list[bytes], feature_extractor) -> list[dict]:
    """Where each model's attention pooling puts its weight: share on the last 1 s and on the zero padding."""
    rows = []
    for audio_bytes in clips:
        x, _ = decode(audio_bytes)
        row = {}
        for name, model in models.items():
            c = model.config
            cropped = prepare_clip(x, window_s=c.window_s, narrowband=c.narrowband, loudness_dbfs=c.loudness_dbfs)
            window = pad_window(cropped, c.window_s)
            feats = feature_extractor(window, sampling_rate=SR, return_tensors="pt").input_features.cuda()
            n_pad = int((len(window) - len(cropped)) / SR * 50)  # 50 encoder positions per second
            captured = {}
            hook = model.pool.score.register_forward_hook(lambda m, i, o: captured.__setitem__("s", o))
            model(input_features=feats)
            hook.remove()
            w = torch.softmax(captured["s"].squeeze(-1).float(), dim=-1)[0].cpu().numpy()  # (400,)
            row[f"attn_last1s_{name}"] = w[-50:].sum()
            row[f"attn_padding_{name}"] = w[:n_pad].sum()
            row[f"attn_peak_s_before_end_{name}"] = (len(w) - 1 - int(np.argmax(w))) / 50
        rows.append(row)
    return rows


def main() -> None:
    # smart-turn: English test clips with both models' predictions
    # Note: FLAC bytes, decoded with soundfile (decode above)
    st_ds = load_dataset(SMART_TURN_TEST, split="train").cast_column("audio", Audio(decode=False))
    st_meta = st_ds.select_columns(["id", "language", "dataset", "synthetic"]).to_pandas()
    st_meta = st_meta[st_meta.language == LANGUAGE].drop(columns="language").sort_values("id", ignore_index=True)
    st = st_meta.rename(columns={"dataset": "source"}).assign(group="smart-turn")

    ttd_ds = load_dataset(TTD, split="test").cast_column("audio", Audio(decode=False))
    ttd_meta = ttd_ds.select_columns(["id", "data_source", "meta"]).to_pandas()
    ttd = ttd_meta.rename(columns={"data_source": "source"}).assign(group=lambda d: d.source)

    # Predictions on the whole test split (run_eval.py, run_eval_baseline.py), joined by id
    files = {name: RESULTS_DIR / "eval" / run / "best" / "predictions.parquet" for name, run in MODELS.items()}
    preds = pd.read_parquet(files["ttd"], columns=["id", "label"])
    for name, f in {**files, "smart_turn_v32": BASELINE}.items():
        preds = preds.merge(
            pd.read_parquet(f, columns=["id", "prob"]).rename(columns={"prob": f"prob_{name}"}), on="id"
        )
    st = st.merge(preds, on="id")
    ttd = ttd.merge(preds, on="id")

    # Samples: 20 showcase clips per dataset (TTD: 10 per source), BIG_N per group for statistics
    show = pd.concat(
        [
            st.sample(20, random_state=0),
            ttd[ttd.source == "spokenwoz"].sample(10, random_state=0),
            ttd[ttd.source == "seamless"].sample(10, random_state=0),
        ]
    ).assign(showcase=True)
    big = pd.concat(
        [
            st[~st.id.isin(show.id)].sample(BIG_N, random_state=1),
            ttd[(ttd.source == "spokenwoz") & ~ttd.id.isin(show.id)].sample(BIG_N, random_state=1),
            ttd[(ttd.source == "seamless") & ~ttd.id.isin(show.id)].sample(BIG_N, random_state=1),
        ]
    ).assign(showcase=False)
    sample = pd.concat([show, big], ignore_index=True)

    # Audio bytes for the sampled clips
    def audio_of(ds, ids):
        rows = pd.Series(np.arange(len(ds)), index=ds["id"]).loc[list(ids)].to_numpy()
        sub = ds.select(rows).select_columns(["id", "audio"])
        return {r["id"]: r["audio"]["bytes"] for r in sub}

    audio = audio_of(st_ds, sample.id[sample.group == "smart-turn"]) | audio_of(
        ttd_ds, sample.id[sample.group != "smart-turn"]
    )
    print(f"{len(audio)} clips", file=sys.stderr)

    import multiprocessing as mp

    with ProcessPoolExecutor(12, mp_context=mp.get_context("spawn")) as pool:
        acoustic = pd.DataFrame(list(pool.map(features_for, audio.items(), chunksize=8)))
    sample = sample.merge(acoustic, on="id")

    # Attention of both models
    from transformers import WhisperFeatureExtractor

    models = {name: EoTModel.from_pretrained(RUNS / run / "best").cuda().eval() for name, run in MODELS.items()}
    fe = WhisperFeatureExtractor.from_pretrained(RUNS / MODELS["ttd"] / "best")  # all three are 8 s
    attn = pd.DataFrame(attention(models, [audio[i] for i in sample.id], fe))
    sample = pd.concat([sample.reset_index(drop=True), attn], axis=1)
    layer_weights = {name: torch.softmax(m.layer_weights, 0).tolist() for name, m in models.items()}
    (OUT / "layer_weights.json").write_text(json.dumps(layer_weights))

    # Transcripts: TTD from meta; smart-turn with Whisper Tiny (showcase clips only)
    meta = sample.meta.map(lambda m: json.loads(m) if isinstance(m, str) else {})
    sample["text"] = meta.map(lambda m: m.get("text"))
    sample["other_words"] = meta.map(lambda m: m.get("other_words"))
    sample["gap_to_next_same_s"] = meta.map(lambda m: m.get("gap_to_next_same_s"))
    sample["gap_to_next_other_s"] = meta.map(lambda m: m.get("gap_to_next_other_s"))
    sample["label_a"] = meta.map(lambda m: m.get("label_a"))
    sample = sample.drop(columns="meta")

    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    proc = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    asr = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny").cuda().eval()
    todo = sample.index[sample.group == "smart-turn"]  # every sampled smart-turn clip (showcase + statistics)
    for i in todo:
        x, _ = decode(audio[sample.id[i]])
        inputs = proc(x[-30 * SR :], sampling_rate=SR, return_tensors="pt").input_features.cuda()
        with torch.no_grad():
            ids = asr.generate(inputs, language="en", task="transcribe")
        sample.loc[i, "text"] = proc.batch_decode(ids, skip_special_tokens=True)[0].strip()

    sample.to_parquet(OUT / "features.parquet", index=False)

    # Spectrograms of the showcase clips' last 3 s, for viewing
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shows = sample[sample.showcase].reset_index(drop=True)
    for group, g in shows.groupby("group"):
        g = g.head(10)
        fig, axes = plt.subplots(len(g), 1, figsize=(10, 1.6 * len(g)))
        for ax, (_, r) in zip(np.atleast_1d(axes), g.iterrows()):
            x, _ = decode(audio[r.id])
            seg = x[-3 * SR :]
            magnitude = np.abs(stft(seg, fs=SR, nperseg=512, noverlap=512 - 128)[2])  # (257, frames)
            S = 20 * np.log10(np.maximum(magnitude, 1e-10) / magnitude.max())  # dB re the loudest bin
            ax.imshow(S, origin="lower", aspect="auto", vmin=-70, vmax=0, extent=[-len(seg) / SR, 0, 0, 8])
            ax.set_ylabel("kHz")
            lab = "complete" if r.label == 1 else "incomplete"
            title = f"{r.source} | {lab} | P_ttd {r.prob_ttd:.2f} P_st {r.prob_st:.2f} P_both {r.prob_both:.2f}"
            ax.set_title(f"{title} | trail {r.trailing_silence_s:.2f}s", fontsize=8)
            ax.tick_params(labelsize=6)
        fig.tight_layout()
        fig.savefig(OUT / f"spec_{group}.png", dpi=80)
        plt.close(fig)


if __name__ == "__main__":
    main()
