# Why TTD and smart-turn behave differently (2026-10-24, peer-reviewed and corrected the same day)

Models compared (5 epochs each, frozen Whisper Tiny encoder, same settings; narrowband + −30 dBFS loudness):
`artifacts/runs/ttd-5ep-20261024-052319` (trained on TTD = mavleo96/turn-detection-data) and
`artifacts/runs/st-5ep-20261024-055918` (trained on English smart-turn).

| test ROC AUC | TTD-trained | smart-turn-trained |
|---|---|---|
| TTD test | **0.885** (SpokenWOZ 0.927, Seamless 0.740) | 0.781 (SpokenWOZ 0.818, Seamless 0.684) |
| smart-turn test (English) | 0.807 (liva_1 0.762, rime_2 0.585) | **0.978** |

Method: 20 random test clips per dataset (smart-turn English; TTD 10 SpokenWOZ + 10 Seamless) read in detail
(transcripts, spectrograms of the last 3 s, both models' P(complete)); every pattern checked on 300 more random clips
per group. Files: `collect.py` → `features.parquet` (940 clips: acoustic features, predictions, attention, transcripts),
`vad_check.parquet` (Silero-VAD trailing silence and speech-to-non-speech level, review step),
`smart_turn_asr.parquet` (Whisper Tiny transcripts of the 300 smart-turn clips), `spec_*.png`. Run with the `turn` env.
Note: the English smart-turn test set is **69% human speech** (liva_1, midcentury_1, human_5, convcollector), not TTS.

## Findings (verified)

**1. Clip endings differ (the best-supported difference).** Silero VAD, time from the last speech to the clip end:
smart-turn median ~0.7 s (10th–90th percentile 0–1.1 s); TTD 0–0.17 s (the dataset's VAD fires ~0.22 s after speech
ends). Clip length also differs: smart-turn 8–9 s (fills the window), SpokenWOZ ~3 s, Seamless ~5 s.

**2. No simple acoustic shortcut.** Single-cue label AUC on 300 clips per group (0.5 = none; SE ≈ 0.03): duration
0.45–0.51, VAD trailing silence 0.51 (smart-turn) / 0.56–0.57 (TTD), energy decay 0.53–0.57, final pitch vs the
speaker's median 0.48 / **0.61 SpokenWOZ** / 0.51. Only SpokenWOZ has a clear prosodic cue: complete turns end ~3.2
semitones above the speaker's median vs ~0.5 for incomplete (why: interpretation, e.g. closings and questions).

**3. Transcripts predict the label about equally weakly in both datasets, but smart-turn is cleaner in one direction.**
Final punctuation alone: smart-turn AUC 0.741 (Whisper Tiny transcripts), Seamless 0.733 (dataset transcripts; the two
transcript sources differ, so this is rough). In smart-turn an unfinished transcript is never complete (0 of 78); in
Seamless 18.5% are (hand-offs, interruptions). In both, ~37–38% of finished-sentence clips are incomplete. TTD examples:
SpokenWOZ dictation pauses ("…it's . tr5015 .") labelled incomplete, Seamless "…the same thing. So" labelled complete.

**4. The two models work differently (measured; roles of layers are interpretation).**
- smart-turn model: layer weights 0.14 / 0.15 / 0.22 / **0.49** (leans on the top layer); attention peaks ~0.9 s
  before the clip end on smart-turn, i.e. around where speech ends (given finding 1).
- TTD model: layer weights 0.27 / 0.20 / 0.21 / 0.32; attention peaks ~0.06 s before the end on every dataset (the
  last syllable and pause). On smart-turn its output follows trailing silence (ρ 0.28 with the VAD measure) and clip
  length (ρ 0.27), which don't predict smart-turn labels: it never saw ~0.7 s endings in training.
- smart-turn model under-predicts "complete" on Seamless (mean P 0.36 vs actual 0.45).

## Corrected in review (do not reuse)
- "smart-turn is mostly TTS, truncated by construction": wrong (69% human; 7 of 11 sampled incomplete clips end mid-phrase).
- "smart-turn audio is 15–20 dB noisier (~29 vs ~45–48 dB)" and "TTD model follows noise level": measurement error
  (the percentile-energy metric read speech as noise in speech-dense clips). VAD-based speech-to-non-speech level is
  ~18–20 dB in all three groups. The "stronger noise augmentation" recommendation is withdrawn.
- "smart-turn trailing silence 0 to ~0.9 s": the energy gate was unreliable; VAD gives median ~0.7 s (finding 1).
- "smart-turn is lexical, TTD is interactional" / "smart-turn's 0.98 mostly measures truncated-sentence detection":
  overstated (finding 3). Why the smart-turn model is so strong in-domain is not explained by any single measured cue.
- "Phone-band speech breaks the smart-turn model's word recognition" (one clip): speculation, dropped.

## Implications
1. TTD is the realistic test: production clips end where the VAD fires, like TTD; smart-turn clips end ~0.7 s later.
2. Train on both datasets, with smart-turn re-cut to VAD-style endings (~0.2 s after the last speech), so the model
   can't key on how clips end and the TTD model's trailing-silence sensitivity goes away.
3. Hypothesis: Seamless (weak acoustic and lexical signal, label depends on the other speaker) needs more capacity
   (unfrozen encoder).

## Caveats
300 clips per group (SE ≈ 0.03 on AUCs); 20-clip examples are illustrative. Whisper Tiny transcripts are rough; the
pitch tracker is a simple autocorrelation one (librosa `pyin` segfaults in the `turn` env, numba 0.68).
