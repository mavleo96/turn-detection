# TTD vs smart-turn: tables (analyze.py)

## Test ROC AUC (whole test split)

| name | TTD only | smart-turn only | both | Smart Turn v3.2 |
|---|---|---|---|---|
| all | 0.878 | 0.811 | 0.898 | 0.791 |
| ttd | 0.886 | 0.781 | 0.887 | 0.750 |
| spokenwoz | 0.927 | 0.814 | 0.928 | 0.784 |
| seamless | 0.742 | 0.688 | 0.744 | 0.667 |
| smart-turn | 0.815 | 0.974 | 0.965 | 0.986 |
| liva_1 | 0.768 | 0.974 | 0.966 | 0.986 |
| chirp3_1 | 0.887 | 0.965 | 0.961 | 0.968 |
| midcentury_1 | 0.938 | 0.997 | 0.995 | 0.999 |
| human_5 | 0.840 | 0.978 | 0.967 | 0.992 |
| rime_2 | 0.594 | 0.944 | 0.898 | 0.995 |

English smart-turn test: 68.6% human speech.

## 1. Clip endings (s; 300 clips per group)

| index | VAD trailing p10 | p50 | p90 | no speech found | length p50 |
|---|---|---|---|---|---|
| smart-turn | 0.000 | 0.684 | 1.083 | 0.000 | 8.680 |
| spokenwoz | 0.000 | 0.140 | 0.172 | 0.007 | 3.020 |
| seamless | 0.000 | 0.000 | 0.172 | 0.000 | 5.276 |

## 2. Single-cue label AUC (0.5 = none, SE ~0.03)

| index | smart-turn | spokenwoz | seamless |
|---|---|---|---|
| clip length | 0.493 | 0.514 | 0.451 |
| VAD trailing non-speech | 0.511 | 0.567 | 0.573 |
| energy slope, last 300 ms of speech | 0.565 | 0.533 | 0.571 |
| final pitch vs the speaker's median | 0.483 | 0.608 | 0.508 |
| speech vs non-speech level | 0.509 | 0.537 | 0.501 |

SpokenWOZ final pitch vs the speaker's median (semitones): complete median 3.17 (mean 2.90), incomplete median 0.45 (mean 1.34).

## 3. Final punctuation as a cue

| index | clips | AUC | unfinished | unfinished but complete | finished but incomplete |
|---|---|---|---|---|---|
| smart-turn | 300.000 | 0.741 | 78.000 | 0.000 | 0.378 |
| seamless | 295.000 | 0.778 | 143.000 | 0.161 | 0.289 |
| seamless, whole test | 19988.000 | 0.738 | 10022.000 | 0.176 | 0.366 |

## 4. Models

| index | layer 1 | layer 2 | layer 3 | layer 4 |
|---|---|---|---|---|
| TTD only | 0.264 | 0.202 | 0.212 | 0.322 |
| smart-turn only | 0.152 | 0.164 | 0.265 | 0.419 |
| both | 0.216 | 0.179 | 0.219 | 0.386 |

Attention peak, s before the clip end (median)

| index | TTD only | smart-turn only | both |
|---|---|---|---|
| smart-turn | 0.020 | 0.880 | 0.220 |
| spokenwoz | 0.030 | 0.200 | 0.000 |
| seamless | 0.040 | 0.240 | 0.080 |

On smart-turn: Spearman rho of P(complete) with cues that don't predict its labels

| index | rho with VAD trailing | rho with length |
|---|---|---|
| TTD only | 0.257 | 0.219 |
| smart-turn only | 0.027 | -0.058 |
| both | 0.022 | -0.081 |
| Smart Turn v3.2 | 0.071 | -0.032 |

Mean P(complete) vs the actual complete share (300 clips per group)

| index | smart-turn | spokenwoz | seamless |
|---|---|---|---|
| TTD only | 0.551 | 0.576 | 0.495 |
| smart-turn only | 0.462 | 0.467 | 0.318 |
| both | 0.478 | 0.578 | 0.490 |
| Smart Turn v3.2 | 0.462 | 0.426 | 0.328 |
| actual complete share | 0.460 | 0.577 | 0.450 |

## Check against artifacts/ttd_vs_st_2026-10-24

Same clips: True (940). Acoustic features, max |new - old|: duration_s 0, trailing_silence_s 0, snr_db 0, hf_share 0, energy_end_slope_db_per_s 0, f0_end_vs_median_st 0.

VAD measures, new (Silero 6.2.3, threshold 0.5, min silence 100 ms) vs old (settings not kept)

| index | rho | median |diff| | new p50 | old p50 |
|---|---|---|---|---|
| vad_trailing_s | 1.000 | 0.000 | 0.140 | 0.140 |
| vad_snr_db | 1.000 | 0.000 | 19.114 | 19.483 |
| nonspeech_s | 1.000 | 0.000 | 0.896 | 0.896 |

Whisper Tiny transcripts of the 300 smart-turn clips: identical text 100.0%, same finished / unfinished call 100.0%.

