# Why TTD and smart-turn behave differently (rewritten 2026-10-25 for the current models and splits)

Models (8 s window, frozen Whisper Tiny encoder, 5 epochs, `load_split` splits; checkpoints on W&B as
`<run>-best:best`): **TTD only** `ttd-8s-5ep-20261024-232232`, **smart-turn only** `smart-turn-8s-5ep-20261024-233637`
(English, without the holdout sources `human_5` / `rime_2`), **both** `both-8s-5ep-20261024-225310`; baseline **Smart
Turn v3.2** (Pipecat, ONNX; trained on all of smart-turn v3.2 train, holdout sources included, never on TTD).

| test ROC AUC (whole test split) | TTD only | smart-turn only | both | Smart Turn v3.2 |
|---|---|---|---|---|
| all (68,394) | 0.878 | 0.811 | **0.898** | 0.791 |
| TTD (60,574): SpokenWOZ / Seamless | 0.886: 0.927 / 0.742 | 0.781: 0.814 / 0.688 | **0.887**: 0.928 / 0.744 | 0.750: 0.784 / 0.667 |
| smart-turn (7,820) | 0.815 | 0.974 | 0.965 | **0.986** |
| smart-turn holdout: human_5 / rime_2 | 0.840 / 0.594 | 0.978 / 0.944 | 0.967 / 0.898 | 0.992 / 0.995 (trained on them) |

**Method.** `collect.py` draws 20 showcase clips per dataset (smart-turn English; TTD 10 SpokenWOZ + 10 Seamless, read
in detail with transcripts and spectrograms of the last 3 s) and 300 random test clips per group (smart-turn,
SpokenWOZ, Seamless) for statistics; fixed seeds, so the same 940 clips as the first version. Per clip: acoustic
features, Silero VAD (threshold 0.5, min silence 100 ms, no padding), the models' predictions (`results/eval`),
attention and layer weights, transcripts (TTD: the dataset's; smart-turn: Whisper Tiny). `analyze.py` computes every
number below into `tables.md`. Files: `features.parquet`, `layer_weights.json`, `spec_*.png`, `tables.md`.

**Reproduction.** Rerun on the same clips, the data-only measures match the first version: same 940 clips,
identical acoustic features and Whisper Tiny transcripts, identical VAD trailing and non-speech times, and the same VAD
level wherever the first version computed one (max difference 0.0006 dB; it left 177 more clips undefined). The first
version's VAD settings were not recorded; these give the same values. Findings 1–3 are therefore reproduced, not
re-estimated; finding 4 is re-measured on the new models. The English smart-turn test set is 68.6% human speech
(liva_1, midcentury_1, human_5, convcollector), not mostly TTS.

## Findings (measured unless marked *interpretation*)

**1. Clip endings differ (the best-supported difference).** Silero VAD, time from the last speech to the clip end
(300 clips per group): smart-turn median 0.68 s (10th–90th percentile 0–1.08 s); SpokenWOZ 0.14 s (0–0.17 s);
Seamless 0 s (0–0.17 s). TTD clips end where the dataset's VAD fired, ~0.2 s after speech. Clip length also differs:
smart-turn median 8.7 s (fills the window), SpokenWOZ 3.0 s, Seamless 5.3 s.

**2. No simple acoustic shortcut.** Single-cue label AUC (0.5 = none; SE ≈ 0.03):

| cue | smart-turn | SpokenWOZ | Seamless |
|---|---|---|---|
| clip length | 0.49 | 0.51 | 0.45 |
| VAD trailing non-speech | 0.51 | 0.57 | 0.57 |
| energy slope, last 300 ms of speech | 0.57 | 0.53 | 0.57 |
| final pitch vs the speaker's median | 0.48 | **0.61** | 0.51 |
| speech vs non-speech level | 0.51 | 0.54 | 0.50 |

Only SpokenWOZ has a clear prosodic cue: complete turns end a median 3.2 semitones above the speaker's median pitch, vs
0.45 for incomplete (means 2.9 vs 1.3). *Interpretation:* closings and questions end high. The speech-to-non-speech
level (VAD) is a median 18.5–19.5 dB in all three groups, so noise level does not separate the datasets.

**3. Transcripts predict the label weakly in both datasets; smart-turn is clean in one direction.** Final punctuation
alone (`.` `?` `!` = finished):

| | clips | AUC | unfinished | unfinished but complete | finished but incomplete |
|---|---|---|---|---|---|
| smart-turn (Whisper Tiny, 300) | 300 | 0.741 | 78 | **0%** | 37.8% |
| Seamless (dataset transcripts, whole test) | 19,988 | 0.738 | 10,022 | 17.6% | 36.6% |

In smart-turn an unfinished transcript is never complete; in Seamless about one in six is (hand-offs,
interruptions). In both, about 37% of finished-sentence clips are incomplete. The two transcript sources differ, so
the comparison is rough. SpokenWOZ transcripts are per annotated turn, not per clip, so they can't be used this way.

**4. The models use the ending differently.**
- *Layer weights* (layers 1–4): TTD only 0.26 / 0.20 / 0.21 / 0.32; smart-turn only 0.15 / 0.16 / 0.27 / **0.42**;
  both 0.22 / 0.18 / 0.22 / 0.39. The smart-turn model leans most on the top layer (*interpretation:* more
  content-like features).
- *Attention* (median peak before the clip end): smart-turn only **0.88 s** on smart-turn clips (about where speech
  ends, given finding 1), 0.20–0.24 s on TTD; TTD only 0.02–0.04 s on every dataset (the last syllable and pause);
  both 0.22 s on smart-turn, 0–0.08 s on TTD.
- *Sensitivity to cues that don't predict smart-turn labels* (Spearman ρ of P(complete) on smart-turn clips): TTD only
  **0.26 with VAD trailing non-speech, 0.22 with clip length**; smart-turn only, both and Smart Turn v3.2 all
  |ρ| < 0.09. The TTD model, which never saw ~0.7 s endings, reads a longer pause as "complete"; trained with
  smart-turn too, the model doesn't.
- *Calibration* (mean P(complete) vs the actual complete share, 300 clips per group):

| | smart-turn (0.46) | SpokenWOZ (0.58) | Seamless (0.45) |
|---|---|---|---|
| TTD only | 0.55 | 0.58 | 0.50 |
| smart-turn only | 0.46 | **0.47** | **0.32** |
| both | 0.48 | 0.58 | 0.49 |
| Smart Turn v3.2 | 0.46 | **0.43** | **0.33** |

  Models trained without TTD under-predict "complete" on TTD's short endings: smart-turn only and Smart Turn v3.2
  alike. That fits Smart Turn's FNR of 0.42 on TTD at its 0.5 threshold. *Interpretation:* both learned that a turn
  ends with ~0.7 s of silence and treat TTD's ~0.2 s as "not yet".

**5. Training on both mostly removes the mismatch.** The combined model matches TTD only on TTD (0.887 vs 0.886,
calibrated on both TTD sources), shows no trailing-silence sensitivity on smart-turn (ρ 0.02), and loses 0.009 on
smart-turn test against smart-turn only (0.965 vs 0.974), most of it on the holdout TTS source rime_2 (0.898 vs 0.944,
394 clips, so suggestive only).

## Corrected in review (do not reuse)
- From the first version: "smart-turn is mostly TTS, truncated by construction" (wrong: 69% human); "smart-turn audio
  is 15–20 dB noisier" and "the TTD model follows noise level" (a measurement error of the percentile-energy metric;
  VAD gives ~19 dB everywhere); "smart-turn trailing silence 0 to ~0.9 s" (energy gate; VAD gives median 0.68 s);
  "smart-turn is lexical, TTD is interactional" and "0.98 mostly measures truncated-sentence detection" (overstated,
  finding 3); "phone-band speech breaks word recognition" (one clip, speculation).
- In this rewrite: the first version's Seamless punctuation numbers (0.733 / 18.5% / ~37%) do not reproduce on the
  300-clip sample (0.778 / 16.1% / 28.9%; the code that made them was not kept) but match the whole Seamless test
  (0.738 / 17.6% / 36.6%), which finding 3 now uses. Its SpokenWOZ pitch numbers (3.2 vs 0.5) were medians.

## Implications
1. TTD is the realistic test: production clips end where a VAD fires, like TTD; smart-turn clips end ~0.7 s later.
   Smart-turn scores (ours ~0.97, Smart Turn's 0.986) say little about a VAD-triggered pipeline, where Smart Turn
   v3.2 drops to 0.750 and misses 42% of completed turns at its default threshold.
2. Train on both: the measured cost on smart-turn is small and the TTD model's pause sensitivity goes away (finding 5).
   Re-cutting smart-turn clips to VAD-style endings (~0.2 s after the last speech) remains worth testing, to stop the
   model learning that a turn ends with 0.7 s of silence at all.
3. Seamless stays at ~0.74 for every model, ours or Smart Turn's: its labels depend on who speaks next, and neither
   acoustic nor lexical cues separate them well (findings 2–3). *Hypothesis:* it needs more capacity (unfrozen
   encoder) or context beyond the user's own audio.

## Caveats
300 clips per group (SE ≈ 0.03 on AUCs); the 20-clip showcases are illustrative. Whisper Tiny transcripts are rough;
the pitch tracker is a simple autocorrelation one (librosa's `pyin` segfaults in the `turn` env). Our models' test
numbers are at frozen-encoder, 5-epoch training; Smart Turn's architecture and training differ (linear head; its
encoder training was not checked), so the baseline comparison is of systems, not of data alone.
