# Data, training and evaluation

End-of-turn (EoT) detector for voice agents: at a pause, is the user's turn complete? Data:
[`pipecat-ai/smart-turn-data-v3.2-train`](https://huggingface.co/datasets/pipecat-ai/smart-turn-data-v3.2-train)
and [`-test`](https://huggingface.co/datasets/pipecat-ai/smart-turn-data-v3.2-test) (CC-BY-4.0), English only.
Model input is the **last 8 s** of 16 kHz mono audio; output is P(turn complete).

## Layout

| Path | What it holds |
|---|---|
| `src/turn_detection/data/` | Everything data: constants and paths (`constants.py`); per-clip audio steps shared with serving: decoding, last-8 s crop, 8 kHz round trip, loudness normalization, padding (`audio.py`); the Hugging Face dataset, `EoTDataset`, `EoTCollator` (`dataset.py`); augmentation with torch-audiomentations (`transforms.py`) |
| `src/turn_detection/model/` | The model only: `EoTConfig` (`config.py`), attention pooling (`modules.py`), `EoTModel`, a transformers `PreTrainedModel` with the encoder setup and layer weights (`model.py`) |
| `src/turn_detection/train/` | HF `Trainer` setup (`trainer.py`), metrics with scikit-learn (`metrics.py`); `python -m turn_detection.train` |
| `scripts/` | One-off runs: explore_dataset, split, samples, run_eval, predict, benchmark |
| `tests/` | Tests for all of the above (CPU only, no network: tiny in-memory datasets stand in for the Hub) |
| `results/` | Whole-dataset exploration (`explore_dataset.py`) and listening samples (`results/samples/`, gitignored) |
| `artifacts/` | Generated, gitignored: manifests, training runs |

## Setup

From the repo root, in the conda env `turn`:

```bash
pip install -r requirements.txt
pip install --no-deps -e .   # the turn_detection package (editable: paths point at this checkout)
wandb login             # once; training logs to Weights & Biases (or pass --report_to none)
pytest                  # ~20 s
```

## Steps

```bash
python scripts/explore_dataset.py   # downloads both splits into the HF cache; stats -> results/
python scripts/split.py             # -> artifacts/manifests/ (smart-turn train/val/test ids)
python scripts/samples.py           # -> results/samples/ (WAVs + CSV), for listening
python -m turn_detection.train                         # TTD -> artifacts/runs/<timestamp>/
python -m turn_detection.train --dataset smart-turn    # English smart-turn (uses the manifests)
python scripts/run_eval.py --model artifacts/runs/<run>/best --benchmark
python scripts/predict.py --model artifacts/runs/<run>/best clip.wav
python scripts/benchmark.py --url http://127.0.0.1:8000 --workers 1,10 --calls 100   # the API (serve/), running
```

| Step | Main flags | Disk | Time |
|---|---|---|---|
| `explore_dataset` | `--num-proc 16`, `--no-audio` | ~94 GB HF cache (46 GB parquet + `datasets`' Arrow copy), 13 MB results | first run download-bound; from the cache ~2 min |
| `split` | `--val-frac` | < 10 MB | ~5 s |
| `samples` | `--per-bucket 15`, `--seed` | ~50 MB | seconds |
| `python -m turn_detection.train` | `--dataset ttd\|smart-turn`, `--max_train_samples`, `--augment false`, `--gain_db`, `--noise_p`, `--freeze_encoder`, plus every `TrainingArguments` flag (`--learning_rate`, `--num_train_epochs`, `--max_steps`, `--report_to`, ...) | ~200 MB per run (2 checkpoints + best) | ~70–80 s per epoch on the RTX 3060, frozen encoder (estimate; data loading is the limit) |
| `run_eval` | `--model`, `--benchmark` | small | ~30 s |

The dataset lives in the Hugging Face cache (`HF_HOME`, default `/workspace/.hf_home`). `/workspace` is
not a persistent volume on this instance: after a recycle the first step downloads it again.

## How the data flows

- **`data.dataset.load_split(split)`** loads a split from the cache in ~2 s (memory-mapped Arrow) with the
  audio left as FLAC bytes (`Audio(decode=False)`; `datasets`' own decoding would need torchcodec).
- **`scripts/split.py`** (smart-turn only; TTD has its own splits) reads the English metadata and writes
  manifests (ids + metadata): `test` = the official test split; `train`/`val` = the official train split
  sorted by id, last 10% to val (ids are random UUIDs, so this is a random split independent of the Hub's
  row order); ids also in test are dropped from train/val.
- **`data.dataset.EoTDataset(source, ids, augment)`** maps ids to rows of the cached split and, per
  item, decodes the FLAC (soundfile, ~3 ms) to 16 kHz mono, keeps the last 8 s, band-limits it to 4 kHz
  (`narrowband`: a 16 → 8 → 16 kHz round trip, ~1.5 ms), normalizes it to −30 dBFS RMS
  (`loudness_dbfs`; near-silent clips are left alone), optionally augments, and only then zero-pads to 8 s
  (on the left by default), so the noise level follows the audio, not the padding. Items are
  `{"input_values": waveform, "labels"}`; no preprocessed copies of the audio.
- **`data.dataset.EoTCollator(feature_extractor)`** turns a batch of items into the model's inputs,
  `{"input_features": (B, 80, 800), "labels": (B,)}`, with `WhisperFeatureExtractor` (clip by clip:
  its batched path is ~2.4× slower on a CPU worker). The `Trainer` uses it as `data_collator`
  (`remove_unused_columns=False`, so the waveforms reach it).
- **`data.transforms`** builds the training augmentation with `torch-audiomentations`:
  `Compose([Gain(±6 dB, p=1), AddColoredNoise(20–40 dB SNR, white to brown, p=0.3)])` (validation and
  test: none).
  It draws from torch's global generator, so the dataset seeds it per (seed, epoch, item) inside
  `torch.random.fork_rng()`: reproducible regardless of workers, without touching the caller's RNG.
- **Evaluation** scores the test clips with the same preprocessing as training (padding side, `narrowband`
  and `loudness_dbfs` come from the model's config), without augmentation.

Real smart-turn splits: train 59,221 / val 6,581 / test 7,820 English clips.

## What the other scripts do

- **explore_dataset**: every row of both splits, all 23 languages: languages, sources, labels, synthetic
  flag, fillers, nulls, id checks, the task spec's rough numbers vs. measured, and audio stats for
  every clip (duration, sample rate, channels, clipping, trailing silence). Writes `results/summary.md`,
  `summary.json`, `tables/*.csv`, `clip_stats.parquet` and plots.
- **samples**: 15 random English clips per (label × synthetic) bucket, as WAVs at their original rate.
- **run_eval**: metrics at 0.5 and at the model's val threshold (`config.threshold`), by source and
  synthetic flag, predicting through `Trainer.predict`; `--benchmark` times batch-1 inference.
- **predict**: `EoTPredictor` (audio of any length and rate → P(turn complete)) and a small CLI.
- **benchmark**: load test for the running inference API (`serve/`, local or its Docker image). Closed loop:
  `--workers 1,10` workers, each making `--calls` back-to-back requests; or open loop: `--rps 5,10,20`
  (Poisson arrivals, latency counted from the scheduled send time). Per step: OK / 503 / errors, req/s,
  p50–p99 and max latency, the server's own time (`Server-Timing`) and mean batch size, and pass/fail
  against `--slo-ms` (default p95 < 100 ms). `--json` saves the results.

Everything is deterministic: rows sorted by id, strata visited in sorted order, fixed seeds (default
42), per-clip generators for augmentation.

## Model

```
waveform (128000,) -> WhisperFeatureExtractor (80, 800) -> [batch] -> Whisper Tiny encoder (frozen in training)
  -> hidden states of its 4 layers (4 x (B, 400, 384)) -> learned softmax-weighted sum (B, 400, 384)
  -> attention pooling over time (B, 384) -> MLP 384-64-1 -> logit -> sigmoid = P(turn complete)
```

- Features come from `WhisperFeatureExtractor` sized for 8 s (loaded from `--whisper` with
  `chunk_length=8`; for Whisper Tiny: 80 mel bins, 25 ms window, 10 ms hop → 80 × 800), computed by `EoTCollator` in the DataLoader workers (~3.7 ms
  per clip on one CPU thread). The `Trainer` saves the extractor with every checkpoint and with `best/`.
- The pretrained `openai/whisper-tiny` encoder is rebuilt for 8 s: 400 positions instead of 1500,
  keeping the first 400 rows of its fixed sinusoidal position table. 7.9M parameters in total. The
  model itself freezes nothing: training freezes the encoder (`--freeze_encoder`, default true), which
  leaves 25k trainable parameters (layer weights, pooling, MLP).
- **Layer mix** (`EoTModel.layer_weights`): one learned weight per encoder layer (the embedding output is
  excluded), softmax-normalized and initialized equal. Each layer's hidden states are layer-normalized
  (no parameters) before the sum: their scales differ, and only the last has the encoder's final
  LayerNorm. The learned weights show which layers carry the turn-end signal.
- **Head**: attention pooling (one learned score per frame, softmax over time) and a 2-layer MLP
  (384 → 64, GELU, dropout 0.1, → 1). The logit goes into BCE-with-logits for training; the sigmoid
  of it is P(turn complete).
- `EoTModel` is a transformers `PreTrainedModel`: `save_pretrained` / `from_pretrained` store the encoder
  config, head settings and the val-chosen `threshold` in `config.json`, so loading needs no download.

## Training

`python -m turn_detection.train` runs the Hugging Face `Trainer` (`src/turn_detection/train/trainer.py`). All settings are fields of
`EoTTrainingArguments` (a `TrainingArguments` subclass with this project's defaults plus its model, data
and augmentation settings), parsed by `HfArgumentParser`, so every setting is logged to Weights & Biases
(project `turn-detection`) and saved with the run. On top of `Trainer`: a callback that sets the
dataset's epoch (augmentation differs per epoch) and, after training, the threshold chosen on val,
stored in `best/config.json`.

| Setting | Default | Why |
|---|---|---|
| Encoder, optimizer | `freeze_encoder` true; the Trainer's AdamW (only parameters that require grad), LR 1e-3, weight decay 0.01 | Only the layer weights, pooling and MLP train; with `--freeze_encoder false` use a much lower LR |
| Schedule | 5% warmup, then cosine (`lr_scheduler_type cosine`, `warmup_steps 0.05`) | Standard for fine-tuning |
| Batch, epochs | 64, 5 | 59k English train clips → ~925 steps/epoch |
| Precision | bf16 autocast, grad clipping at 1.0 | The RTX 3060 supports bf16; no loss scaling needed |
| Data loading | 16 workers (`dataloader_num_workers`) | Decode + augment + collate (features): ~1,000 clips/s with 16 workers. With the frozen encoder the GPU could take ~1,350, so data loading is the limit |
| Loss | BCE with logits, no class weights | Labels are ~50/50 |
| Preprocessing | last 8 s, 8 kHz round trip (`--narrowband`, default true), RMS normalized to −30 dBFS (`--loudness_dbfs`), zero-padded on the left (`--padding`); all saved in the model's config | SpokenWOZ (8 kHz phone) and Seamless must not be separable by bandwidth or by level (median RMS −22 vs −46 dBFS; smart-turn −19); evaluation and inference preprocess the same way. After normalization, +6 dB gain clips ~2% of clips (before: 59% of SpokenWOZ, up to 100% of smart-turn sources) |
| Augmentation | torch-audiomentations: gain ±6 dB, colored noise (white to brown, `f_decay` 0–2) p=0.3 at 20–40 dB SNR, measured against the unpadded clip | Robustness to level and background noise; the clip end is never cut or moved |
| Model selection | evaluate and save every epoch, keep the best by val ROC AUC (`load_best_model_at_end`); the val threshold maximizing F1 goes into `best/config.json` | Test is never used for tuning |

Measured on the RTX 3060 (frozen encoder): a batch-64 bf16 training step takes ~47 ms (~0.5 GB peak).
A 60-step smoke run on 3,840 clips reaches val ROC AUC 0.83 (test 0.83); the earlier fine-tuned encoder
reached 0.95 in the same smoke run, so a full run will show how much freezing costs.

Each run directory holds the Trainer's last checkpoints (`checkpoint-<step>/`, with `trainer_state.json`),
`best/` (`config.json` with the threshold + `model.safetensors`), and `final_results.json` (val metrics
at the threshold). Metrics (scikit-learn): accuracy, precision, recall, F1, ROC AUC and **false endpoint
rate** (incomplete turns predicted complete, i.e. the agent would interrupt).

## Assumptions

- **English only** (`language == "eng"`): 73,622 clips, 67% human recordings, 33% synthetic TTS.
- **Last-8 s window**: 54% of English clips are longer and get cropped (92% of `liva_1`); shorter ones are
  left-padded with zeros so the clip end always sits at the same position.
- **16 kHz input, telephone band**: all audio is 16 kHz and goes through an 8 kHz round trip
  (`narrowband`) in training, evaluation and serving.
- GPU kernels are not forced to be deterministic, so training runs can differ in the last digits.

## Known limitations

- `spoken_text` is empty, so complete and truncated versions of the same sentence can't be grouped and
  may land in different splits (possible leakage; val/test scores may be optimistic).
- smart-turn train/val is split per clip: the same speaker or session can be in both (no speaker ids), so
  smart-turn val scores may be optimistic. TTD's splits share no conversation (Seamless: no person).

## Differences from the original data-task spec

- The dataset is downloaded whole through `datasets` (cached) instead of streamed into English-only
  shards; training decodes clips on the fly instead of reading exported arrays.
- One-off steps are scripts in `scripts/`; `inspect.py` is `samples.py` (so it can't shadow Python's
  `inspect`); the English data report is part of `explore_dataset.py`'s `results/summary.md`.
- Resampling uses `soxr` (librosa's high-quality backend) directly.
- Runs in the existing Python 3.12 env; the code avoids 3.12-only syntax.
