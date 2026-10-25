# turn-detection

End-of-turn (EoT) detection for voice agents: at a pause, is the user's turn complete? The model sees the **last
`window_s` seconds** (2, 4 or 8; default 8) of 16 kHz mono audio and outputs P(turn complete); positive class =
complete.

Data: [`pipecat-ai/smart-turn-data-v3.2-train`](https://huggingface.co/datasets/pipecat-ai/smart-turn-data-v3.2-train)
and [`-test`](https://huggingface.co/datasets/pipecat-ai/smart-turn-data-v3.2-test) (CC-BY-4.0; English only here),
plus [`mavleo96/turn-detection-data`](https://huggingface.co/datasets/mavleo96/turn-detection-data) (TTD: SpokenWOZ
phone calls + Seamless Interaction).

## Layout

| Path | What it holds |
|---|---|
| `src/turn_detection/data/` | Constants, paths and the preprocessing defaults (`__init__.py`); per-clip audio steps shared with serving: last-`window_s` crop, 8 kHz round trip, loudness normalization, padding (`audio.py`); the splits (`load_split`), `EoTDataset`, `EoTCollator` (`dataset.py`); augmentation with torch-audiomentations (`transforms.py`) |
| `src/turn_detection/model/` | The model only: `EoTConfig` (`config.py`), attention pooling (`modules.py`), `EoTModel`, a transformers `PreTrainedModel` with the encoder setup and layer weights (`model.py`) |
| `src/turn_detection/train/` | HF `Trainer` setup (`trainer.py`), metrics with scikit-learn (`metrics.py`); `python -m turn_detection.train` |
| `scripts/` | One-off runs: `explore_dataset.py`, `samples.py`, `run_eval.py` |
| `tests/` | Tests for all of the above (CPU only, no network: tiny in-memory datasets stand in for the Hub) |
| `results/` | Analyses: `data_exploration/` (`explore_dataset.py` output + `EXPLORE.md`), `ttd_vs_smart_turn/` (`FINDINGS.md`), and test evaluations (`eval/<run>/<checkpoint>/`, `run_eval.py`), tracked; listening samples (`samples/`), gitignored |
| `artifacts/` | Generated, gitignored: training runs |

`pip install -e ".[train]"` installs everything for training, evaluation, scripts and tests; `pip install .` only the
core (model + audio preprocessing) for serving.

## Setup

From the repo root, in the conda env `turn` (`environment.yml`; it includes FFmpeg, which torchcodec needs):

```bash
pip install -r requirements.txt
pip install --no-deps -e .   # the turn_detection package (editable: paths point at this checkout)
wandb login                  # once; training logs to Weights & Biases (or pass --report_to none)
pytest                       # ~25 s
```

Lint and format with [ruff](https://docs.astral.sh/ruff/) (config in `ruff.toml`) through a pre-commit hook:

```bash
pip install -r requirements-dev.txt
pre-commit install            # once per clone
pre-commit run --all-files    # optional: check everything git tracks
```

On each commit the hook runs `ruff check --fix` and `ruff format` on the staged Python files. If either one changes
a file, the commit stops: review the changes, `git add` them, and commit again.

## Steps

```bash
python scripts/explore_dataset.py                    # smart-turn repos + the training splits -> results/data_exploration/
python scripts/samples.py                            # -> results/samples/ (WAVs + CSV), for listening
python -m turn_detection.train                       # smart-turn + TTD, 8 s -> artifacts/runs/<timestamp>/
python -m turn_detection.train --datasets ttd        # one dataset
python -m turn_detection.train --window_s 2          # a 2 s window (one run per window)
python scripts/run_eval.py artifacts/runs/<run>/best     # -> results/eval/<run>/best/
```

| Step | Main flags | Disk | Time |
|---|---|---|---|
| `explore_dataset` | `--num-proc 16`, `--no-audio` | ~94 GB HF cache (46 GB parquet + `datasets`' Arrow copy), < 1 MB results | first run download-bound; from the cache a few minutes |
| `samples` | `--per-bucket 15`, `--seed` | ~50 MB | seconds |
| `python -m turn_detection.train` | `--datasets smart-turn ttd`, `--window_s`, `--max_train_samples`, `--augment false`, `--gain_db`, `--noise_p`, `--freeze_encoder`, plus every `TrainingArguments` flag (`--learning_rate`, `--num_train_epochs`, `--max_steps`, `--report_to`, ...) | ~200 MB per run (2 checkpoints + best) | ~7 min per epoch on the RTX 3060 for both datasets (~35 min for 5), frozen encoder; data loading is the limit |
| `run_eval` | the checkpoint, `--out`, `--workers`, `--cpu` | ~5 MB | ~1 min on the RTX 4090 |

The data lives in the Hugging Face cache (`HF_HOME`, default `/workspace/.hf_home`). `/workspace` is not a
persistent volume on this instance: after a recycle the first step downloads it again.

## How the data flows

- **`data.dataset.load_split(split, datasets)`** builds `train`, `validation` or `test` of `smart-turn` and/or `ttd`
  (default both) from the cache (~20 s the first time, then cached by `datasets`) as one table, `id, audio, dataset,
  corpus, label`, with 16 kHz mono audio (`datasets` decodes with torchcodec) and `label` a `ClassLabel(incomplete,
  complete)`:
  - smart-turn: English only. Its train split without the holdout sources `human_5` and `rime_2` is split 90/10 at
    random (`train_test_split`, seed 42, on the Hub's row order) into train and validation; test is its whole
    English test split, so the holdout sources' test clips (402 + 394) come from sources never trained or tuned on.
  - TTD: its own train / validation / test splits (`data_source` becomes `dataset`).

  - Left out: 4 SpokenWOZ train clips FFmpeg can't decode (`UNDECODABLE` in `turn_detection.data`; soundfile can).

  Real splits (smart-turn + TTD): train 52,893 + 333,280 / validation 5,877 + 37,600 / test 7,820 + 60,574 clips.
- **`data.dataset.EoTDataset(split, augment, ...)`**, per item, decodes the FLAC (~2 ms), keeps the last `window_s`
  seconds, band-limits them to 4 kHz (`band_limit`: a 16 → 8 → 16 kHz round trip, ~1.5 ms), normalizes them to
  −30 dBFS RMS (near-silent clips are left alone), optionally augments, and only then zero-pads on the left to
  `window_s`, so the noise level follows the audio, not the padding. Items are `{"input_values": waveform,
  "labels"}`; no preprocessed copies of the audio.
- **`data.dataset.EoTCollator(feature_extractor)`** turns a batch of items into the model's inputs,
  `{"input_features": (B, 80, 100·window_s), "labels": (B,)}`, with `WhisperFeatureExtractor` (clip by clip: its
  batched path is ~2.4× slower on a CPU worker). It rejects audio of another length than the extractor's window,
  which would otherwise keep only the start of the clip. The `Trainer` uses it as `data_collator`
  (`remove_unused_columns=False`, so the waveforms reach it).
- **`data.transforms`** builds the training augmentation with `torch-audiomentations`:
  `Compose([Gain(±6 dB, p=1), AddColoredNoise(20–40 dB SNR, white to brown, p=0.3)])` (validation and test: none).
  It draws from torch's global generator, so the dataset seeds it per (seed, epoch, item) inside
  `torch.random.fork_rng()`: reproducible regardless of workers, without touching the caller's RNG.
- **Evaluation** scores the test clips with the same preprocessing as training (`window_s`, `narrowband` and
  `loudness_dbfs` come from the model's config, the feature extractor from `best/`), without augmentation.

## Scripts

- **explore_dataset**: every row of both smart-turn repos, all 23 languages: languages, sources, labels, synthetic
  flag, fillers, nulls, id checks, the task spec's rough numbers vs. measured, and audio stats for every clip
  (duration, sample rate, channels, clipping, trailing silence); and the training splits (smart-turn + TTD as
  `load_split` builds them): clips, hours, label ratio and clip length per corpus, source and `--datasets` choice.
  Writes `results/data_exploration/summary.md`, `summary.json`, `tables/*.csv` and plots; the write-up of the split
  findings is `results/data_exploration/EXPLORE.md`.
- **samples**: 15 random English smart-turn clips per (label × synthetic) bucket, as WAVs at their original rate.
- **run_eval**: one of our checkpoints on the whole test split, predicting through `Trainer.predict` with the
  model's window, preprocessing and validation threshold (`config.threshold`). Writes `results/eval/<run>/<checkpoint>/`:
  `predictions.parquet` (per clip: id, source, corpus, label, prob, pred), `metrics.csv` and `metrics.md` (n,
  complete share, accuracy, F1, ROC AUC, PR AUC, FNR, FPR for all clips, per corpus and per source).

## Model

Shapes for the default 8 s window; with `--window_s w` (2, 4 or 8): `16000·w` samples, `100·w` mel frames, `50·w`
encoder positions.

```
waveform (128000,) -> WhisperFeatureExtractor (80, 800) -> [batch] -> Whisper Tiny encoder (frozen in training)
  -> hidden states of its 4 layers (4 x (B, 400, 384)) -> learned softmax-weighted sum (B, 400, 384)
  -> attention pooling over time (B, 384) -> MLP 384-64-1 -> logit -> sigmoid = P(turn complete)
```

- Features come from `WhisperFeatureExtractor` sized for the window (loaded from `--whisper` with
  `chunk_length=window_s`; for Whisper Tiny: 80 mel bins, 25 ms window, 10 ms hop), computed by `EoTCollator` in the
  DataLoader workers (~3.7 ms per clip on one CPU thread). The `Trainer` saves the extractor with every checkpoint
  and with `best/`.
- The pretrained `openai/whisper-tiny` encoder is rebuilt for the window: `50·window_s` positions instead of 1500,
  keeping the first rows of its fixed sinusoidal position table. 7.9M parameters in total. The model itself freezes
  nothing: training freezes the encoder (`--freeze_encoder`, default true), which leaves 25k trainable parameters
  (layer weights, pooling, MLP).
- **Layer mix** (`EoTModel.layer_weights`): one learned weight per encoder layer (the embedding output is excluded),
  softmax-normalized and initialized equal. Each layer's hidden states are layer-normalized (no parameters) before
  the sum: their scales differ, and only the last has the encoder's final LayerNorm. The learned weights show which
  layers carry the turn-end signal.
- **Head**: attention pooling (one learned score per frame, softmax over time) and a 2-layer MLP (384 → 64, GELU,
  dropout 0.1, → 1). The logit goes into BCE-with-logits for training; its sigmoid is P(turn complete).
- `EoTModel` is a transformers `PreTrainedModel`: `save_pretrained` / `from_pretrained` store the encoder config (as
  built, including its positions), `window_s`, the preprocessing settings, head settings and the validation-chosen
  `threshold` in `config.json`, so loading needs no download.

## Training

`python -m turn_detection.train` runs the Hugging Face `Trainer` (`src/turn_detection/train/trainer.py`). All settings
are fields of `EoTTrainingArguments` (a `TrainingArguments` subclass with this project's defaults plus its model,
data and augmentation settings), parsed by `HfArgumentParser`, so every setting is logged to Weights & Biases
(project `turn-detection`) and saved with the run. On top of `Trainer`: a callback that sets the dataset's epoch
(augmentation differs per epoch) and, after training, the threshold chosen on validation, stored in
`best/config.json`.

| Setting | Default | Why |
|---|---|---|
| Data | `--datasets smart-turn ttd` (both), splits from `load_split` | 386k train clips (52.9k smart-turn + 333.3k TTD) → ~6,030 steps/epoch at batch 64 |
| Encoder, optimizer | `freeze_encoder` true; the Trainer's AdamW (only parameters that require grad), LR 1e-3, weight decay 0.01 | Only the layer weights, pooling and MLP train; with `--freeze_encoder false` use a much lower LR |
| Schedule | 5% warmup, then cosine (`lr_scheduler_type cosine`, `warmup_steps 0.05`), 5 epochs, batch 64 | Standard for fine-tuning |
| Order | a new random order every epoch (`train_sampling_strategy random`); validation and test keep their order | `run_eval` matches predictions to ids by position |
| Precision | bf16 autocast on the GPU, grad clipping at 1.0 | No loss scaling needed |
| Data loading | 16 workers (`dataloader_num_workers`) | Decode + augment + features: ~1,000 clips/s with 16 workers on the RTX 3060 machine; with the frozen encoder the GPU could take ~1,350, so data loading is the limit |
| Loss | BCE with logits, no class weights | Labels are ~50/50 |
| Preprocessing | last `--window_s` seconds (2, 4 or 8; default 8), 8 kHz round trip (`--narrowband`, default true), RMS normalized to −30 dBFS (`--loudness_dbfs`), zero-padded on the left; all saved in the model's config (defaults: `WINDOW_S`, `NARROWBAND`, `LOUDNESS_DBFS` in `turn_detection.data`) | SpokenWOZ (8 kHz phone) and Seamless must not be separable by bandwidth or by level (median RMS −22 vs −46 dBFS; smart-turn −19); evaluation and inference preprocess the same way. After normalization, +6 dB gain clips ~2% of clips (before: 59% of SpokenWOZ, up to 100% of smart-turn sources) |
| Augmentation | torch-audiomentations: gain ±6 dB, colored noise (white to brown, `f_decay` 0–2) p=0.3 at 20–40 dB SNR, measured against the unpadded clip | Robustness to level and background noise; the clip end is never cut or moved |
| Model selection | evaluate and save every epoch, keep the best by validation ROC AUC (`load_best_model_at_end`); the validation threshold maximizing F1 goes into `best/config.json` | Test is never used for tuning |

A run directory (default `artifacts/runs/<timestamp>`) holds the Trainer's last checkpoints (`checkpoint-<step>/`,
with `trainer_state.json`), `best/` (`config.json` with the threshold, `model.safetensors`, the feature extractor),
and `final_results.json` (validation metrics at the threshold). With W&B reporting (the default), `best/` is also
logged to the run as a model artifact `<run name>-best` (alias `best`; metadata: window, threshold, validation ROC
AUC). Metrics (scikit-learn; positive class = complete):
accuracy, F1, **FPR** (incomplete turns predicted complete: the agent interrupts), **FNR** (complete turns predicted
incomplete: the agent waits too long), ROC AUC and PR AUC (average precision).

## Assumptions and known limitations

- **English only**: smart-turn's 73,622 English clips (67% human recordings, 33% synthetic TTS; 23 languages in
  all) and TTD (English).
- **Last-`window_s` window**: at the default 8 s, 60% of smart-turn train clips (91% of `liva_1`) and 22% of TTD
  train clips are longer and get cropped; shorter ones are left-padded with zeros so the clip end (the pause) always
  sits at the window end (`results/data_exploration/EXPLORE.md`).
- **16 kHz input, telephone band**: all audio is 16 kHz and goes through an 8 kHz round trip (`narrowband`) in
  training and evaluation; serving must do the same (the `serve` branch).
- **Possible leakage within smart-turn**: `spoken_text` is empty, so complete and truncated versions of the same
  sentence can't be grouped, and train/validation is split per clip (no speaker ids): the same speaker or sentence
  can be in both, so smart-turn validation scores may be optimistic. The holdout sources (`human_5`, `rime_2`) are
  the exception. TTD's splits share no conversation (Seamless: no participant).
- **The test mixes differ from train**: TTD test is 67% SpokenWOZ vs 50% in TTD train, so pooled test numbers lean
  on SpokenWOZ; compare per source (`EXPLORE.md`).
- GPU kernels are not forced to be deterministic, so training runs can differ in the last digits.
