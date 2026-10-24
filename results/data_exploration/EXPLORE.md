# Split exploration: counts, label ratios, clip lengths (2026-10-24)

What the `load_split` splits contain (`--datasets` both, smart-turn or ttd): clips, hours, label ratio and clip
length, per corpus and source. Tables: the "Splits" sections of `summary.md` and `tables/splits*.csv` next to this
file, written by `scripts/explore_dataset.py` (which also explores the raw smart-turn repos, all languages).

## Method

- **Clips:** all English smart-turn clips (73,622: train 65,802, test 7,820) and all TTD clips (431,458), each
  assigned to the split `load_split` puts it in. smart-turn train clips of the holdout sources (`human_5`, `rime_2`)
  are in no split ("excluded (holdout)", 7,032 clips).
- **Clip length:** smart-turn from `explore_dataset.py`'s audio pass (every clip decoded); TTD from
  `timestamp_end - timestamp_start`, without decoding. These equal the audio length: on 3,000 random TTD clips (1,000
  per split) the FLAC header length and the timestamps differ by 0.000 s.
- **Label:** 1 = complete. **Duration AUC:** ROC AUC of clip length alone for the label, over every clip of a source
  (all splits). 0.5 = no signal; above 0.5 = complete clips are longer, below = shorter.
- Measurements are stated as such; interpretations are marked *Interpretation*.

## Splits

Since this write-up, `load_split` leaves out 4 SpokenWOZ train clips torchcodec can't decode (`UNDECODABLE`), so
TTD train is 333,280 clips and train 386,173; the shares below change by less than 0.01 percentage points.

| split | clips | hours | complete | smart-turn share (clips / hours) |
|---|---|---|---|---|
| train | 386,177 | 684.1 | 50.2% | 13.7% / 19.6% |
| validation | 43,477 | 78.5 | 49.9% | 13.5% / 18.8% |
| test | 68,394 | 108.9 | 51.6% | 11.4% / 17.6% |
| excluded (holdout) | 7,032 | 10.1 | 48.7% | 100% |

| | smart-turn train / validation / test | TTD train / validation / test |
|---|---|---|
| clips | 52,893 / 5,877 / 7,820 | 333,284 / 37,600 / 60,574 |
| hours | 134.4 / 14.8 / 19.1 | 549.7 / 63.7 / 89.8 |
| complete | 49.3% / 50.4% / 49.0% | 50.4% / 49.8% / 52.0% |
| median length | 9.80 / 9.60 / 9.20 s | 3.50 / 3.56 / 3.28 s |
| > 8 s (cropped by the window) | 60.3% / 59.4% / 55.5% | 22.1% / 22.1% / 18.1% |
| < 1 s | 0.5% / 0.5% / 0.6% | 8.3% / 8.6% / 8.4% |

`--datasets smart-turn` or `ttd` uses exactly the corresponding column; `both` uses the sum (first table).

## Sources

| source | train / validation / test clips | complete (all splits) | median length | > 8 s |
|---|---|---|---|---|
| spokenwoz (ttd) | 167,376 / 20,510 / 40,353 | 58.0% | 2.8–2.9 s | 10% |
| seamless (ttd) | 165,908 / 17,090 / 20,221 | 42.1% | 4.8–5.2 s | 34–36% |
| liva_1 | 27,893 / 3,029 / 3,831 | 49.9% | 12.0–12.2 s | 91% |
| chirp3_1 | 11,762 / 1,353 / 1,471 | 49.3% | 7.6–7.7 s | 45–47% |
| midcentury_1 | 7,624 / 865 / 1,044 | 48.3% | 3.7 s | 0–0.2% |
| orpheus_endfiller_1 / grammar_1 / midfiller_1 | 1,399–1,495 / 134–171 / 140–181 each | 48.6% / 51.7% / 49.3% | 4.8–6.9 s | 17–28% |
| chirp3_3_short | 704 / 88 / 104 | 43.2% | 1.0–1.2 s | 0% |
| human_convcollector_1 | 608 / 71 / 90 | 46.9% | 2.9–3.1 s | 0% |
| human_5 (holdout, human) | – / – / 402 (+ 3,527 excluded) | 50.8% | 4.2–4.3 s | 3% |
| rime_2 (holdout, TTS) | – / – / 394 (+ 3,505 excluded) | 45.4% | 5.8–5.9 s | 12–15% |

## Findings

1. **The TTD test split has a different source mix from TTD train.** SpokenWOZ is 50.2% of TTD train, 54.5% of
   validation and 66.6% of test (40,353 of 60,574). SpokenWOZ is 57.5–58.2% complete and Seamless 40.5–42.5%, and
   the mix accounts exactly for the TTD label ratios (train 50.4%, validation 49.8%, test 52.0%).
   *Interpretation:* the earlier runs scored SpokenWOZ much higher than Seamless (TTD-trained test ROC AUC 0.927 vs
   0.742), so a pooled TTD (or pooled test) number is higher than it would be with train's mix. Per-source numbers
   are not affected.

2. **TTD clips are capped at 30 s.** 11,562 TTD clips (2.7%) are exactly 30.000 s and none is longer; 11,426 of
   them are Seamless (5.6% of Seamless). Seamless clips are long: median 4.8–5.2 s, 90th percentile 20.9–25.1 s,
   34–36% longer than 8 s.
   *Interpretation:* the cap looks like a dataset-construction limit. With the last-8 s window the model doesn't
   see the start of the clip for about a third of Seamless clips.

3. **The 8 s window crops most smart-turn clips and pads others.** 91% of `liva_1` clips (median 12.1 s) are longer
   than 8 s, so 60% of smart-turn train is cropped; `midcentury_1` (median 3.7 s) and `chirp3_3_short` (median
   1.0–1.2 s, 33–42% under 1 s) are mostly left padding.

4. **Clip length is a strong label cue in the two holdout sources, but not in the training sources.** Duration AUC:
   `human_5` 0.274 (median complete 3.65 s vs incomplete 4.82 s), `rime_2` 0.371 (5.46 vs 6.20 s). In the sources
   used for training it is 0.496–0.591 (complete clips slightly longer or no difference), except `chirp3_3_short`
   0.375 (896 clips, median 1.0–1.2 s); TTD: Seamless 0.462, SpokenWOZ 0.479. Both holdout sources are mostly under
   8 s (3% and 12–15% cropped), so the model sees their length through the amount of padding.
   *Interpretation:* the holdout clips carry a length cue the training data does not teach, with the opposite sign
   to most training sources; holdout results may partly reflect how the model treats clip length.

5. **The holdout sources' test clips are less often complete than their train clips:** `human_5` 44.3% in test vs
   51.6% in train (excluded), `rime_2` 42.4% vs 45.7%. Both are on the low side of the other smart-turn sources
   (48–52% over all splits; per split `chirp3_3_short` 42–49% and `human_convcollector_1` 42–48%).

6. **Short clips (< 1 s) are 8–9% of TTD and go in opposite directions by source:** in Seamless 10.3% of complete vs
   5.6% of incomplete clips are under 1 s; in SpokenWOZ 7.5% vs 11.2%. In smart-turn only `chirp3_3_short` has many
   (45.5% of complete, 28.9% of incomplete) and `human_convcollector_1` a few (8.3% of complete, 0.7% of
   incomplete).

7. **Within each corpus, validation and test look like train** in length and label ratio: smart-turn medians 9.80 /
   9.60 / 9.20 s and 49–50% complete; TTD differences come from the source mix (finding 1), per source the medians
   agree within 0.42 s (Seamless 4.81 / 5.23 / 5.00 s).

## Not covered here

Acoustic properties (silence before and after speech with a VAD, level, noise, bandwidth), duplicate or prefix audio
across splits, and TTD's metadata (turn structure, transcripts, speakers) were not measured in this pass; a full
audio pass costs about an hour on this instance (61-CPU quota). Already established elsewhere: smart-turn is all mono
16 kHz (`explore_dataset.py`), TTD clips are mono on 600 sampled clips (200 per split) (all 16 kHz, as the old training code
rejected other rates), splits share no ids, TTD conversations or Seamless participants, and smart-turn clips end
~0.7 s after the last speech vs ~0.2 s for TTD (`../ttd_vs_smart_turn/FINDINGS.md`, VAD on samples).
