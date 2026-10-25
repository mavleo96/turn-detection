# turn-detection

End-of-turn (EoT) detection for voice agents: when the user pauses, is their turn complete? The model sees the last
**2, 4 or 8 s** (`window_s`, default 8) of 16 kHz mono audio, ending at the pause where a VAD stopped, and outputs
P(turn complete). A Whisper Tiny encoder with a small head, trained on public conversation data plus our own TTD set,
served as an ONNX model behind FastAPI.

## Results

Test ROC AUC with 95% bootstrap intervals (68,394 clips: TTD 60,574, smart-turn English 7,820). Our models: frozen
encoder, 5 epochs, threshold chosen on validation. Full tables: `results/eval/`.

| model (training data, window) | all | TTD | smart-turn |
|---|---|---|---|
| **both, 8 s** | **0.898** [0.896, 0.901] | **0.887** [0.884, 0.889] | 0.965 [0.961, 0.968] |
| both, 4 s | 0.895 [0.892, 0.897] | 0.883 [0.881, 0.886] | 0.958 [0.953, 0.962] |
| both, 2 s | 0.883 [0.881, 0.885] | 0.874 [0.871, 0.877] | 0.940 [0.935, 0.945] |
| TTD only, 8 s | 0.878 [0.876, 0.881] | 0.886 [0.883, 0.888] | 0.815 [0.805, 0.825] |
| smart-turn only, 8 s | 0.811 [0.808, 0.814] | 0.781 [0.777, 0.784] | 0.974 [0.971, 0.977] |
| Smart Turn v3.2 (Pipecat, baseline) | 0.791 [0.788, 0.794] | 0.750 [0.747, 0.754] | **0.986** [0.983, 0.988] |

- **both, 8 s** at its threshold: F1 0.826, FPR 0.296 (incomplete turns called complete: the agent interrupts), FNR
  0.101 (complete turns missed: the agent waits).
- **Against a plain VAD:** a silence timeout has to wait 0.74 s after speech, 0.52 s longer than our model, to
  interrupt as rarely on TTD (`results/eval/vad-timeout/`).
- **Smart Turn v3.2** was trained on all of smart-turn (its smart-turn number is in-domain) and never on TTD, whose
  clips end ~0.5 s sooner after speech than smart-turn's: it misses 42% of completed turns there.
- **Serving** (both, 8 s; ONNX Runtime on 2 CPU cores): one caller p50 46 ms / p95 67 ms, 20.5 req/s; up to
  23.3 req/s under load (`results/benchmark/`).
- Why the datasets behave differently: `results/ttd_vs_smart_turn/FINDINGS.md`; data exploration:
  `results/data_exploration/EXPLORE.md`.

The intervals resample clips independently; TTD clips from one conversation are correlated, so its intervals are
optimistic.

## Data

| | smart-turn ([pipecat-ai v3.2](https://huggingface.co/datasets/pipecat-ai/smart-turn-data-v3.2-train), CC-BY-4.0) | TTD ([mavleo96/turn-detection-data](https://huggingface.co/datasets/mavleo96/turn-detection-data)) |
|---|---|---|
| What | single-speaker utterances, complete or cut off (67% human, 33% TTS); English only here | real two-party conversations: SpokenWOZ phone calls and Seamless Interaction; a clip ends where a VAD fired, 0.224 s after speech; label: does the other speaker talk next |
| Train / validation / test | 52,893 / 5,877 / 7,820 | 333,280 / 37,600 / 60,574 |
| Split | test: the official test split; the rest 90/10 at random (seed 42), without the sources `human_5` and `rime_2`, which appear only in test | the dataset's own splits; no conversation (Seamless: no participant) is in two splits |

`load_split(split, datasets)` builds a split of either or both as one table, `id, audio, dataset, corpus, label`
(4 SpokenWOZ train clips FFmpeg can't decode are left out). Every clip gets the same preprocessing in training,
evaluation and serving (`turn_detection.data.audio`): the last `window_s` seconds → 4 kHz band (16 → 8 → 16 kHz, so
phone and wideband audio look alike) → RMS normalized to −30 dBFS → zero-padded on the left, so the pause is always at
the end of the window. Training adds gain (±6 dB) and colored noise (p 0.3, 20–40 dB SNR) before the padding.

## Model and training

```
log-mel (80, 100·window_s) -> Whisper Tiny encoder (frozen) -> softmax-weighted sum of its 4 layers
  -> attention pooling over time -> MLP 384-64-1 -> P(turn complete)
```

- The encoder is rebuilt for the window (`50·window_s` positions instead of 1500); 7.8M parameters, 25k trainable.
- `EoTModel` is a transformers `PreTrainedModel`; `config.json` stores the encoder, the window, the preprocessing and
  the validation threshold, so a checkpoint loads (and serves) without downloads.
- `python -m turn_detection.train` runs the Hugging Face `Trainer`: AdamW lr 1e-3, 5% warmup + cosine, 5 epochs,
  batch 64, bf16, BCE; best epoch by validation ROC AUC, threshold = best F1 on validation; never tuned on test.
  Every setting is a flag (`EoTTrainingArguments`) and is logged to Weights & Biases; `best/` is uploaded as model
  artifact `<run>-best`, e.g. `wandb artifact get mavleo96-team/turn-detection/both-8s-5ep-20261024-225310-best:best`.

## Running it

Setup, in the conda env `turn` (`environment.yml`, which includes the FFmpeg that torchcodec needs):

```bash
pip install -r requirements.txt
pip install --no-deps -e .   # the turn_detection package
wandb login                  # once (or train with --report_to none)
pytest                       # ~1 min, CPU, no network
```

| Command | Does | Output | Time (RTX 4090, 61-CPU quota) |
|---|---|---|---|
| `python -m turn_detection.train [--datasets ttd] [--window_s 2]` | train (default: both datasets, 8 s; `--dataloader_num_workers 32` for speed) | `artifacts/runs/<timestamp>/` | 3 min (smart-turn) to 15 min (both, 8 s) |
| `python scripts/run_eval.py <run>/best` | one of our checkpoints on the test split | `results/eval/<run>/best/` | ~2 min |
| `python scripts/run_eval_baseline.py` | Pipecat's Smart Turn v3.2 on the same split | `results/eval/smart-turn-v3.2-gpu/` | ~10 min (CPU) |
| `python scripts/run_eval_vad.py` | silence-timeout VAD on TTD: errors vs latency | `results/eval/vad-timeout/` | < 1 min |
| `python scripts/explore_dataset.py` | the smart-turn repos and the splits | `results/data_exploration/` | a few minutes |
| `python scripts/samples.py` | listening samples | `results/samples/` | seconds |
| `python scripts/export_onnx.py <run>/best` | checkpoint → `model.onnx` for `serve/` (checked against torch) | `<run>/best/model.onnx` | seconds |
| `python -m serve --model <dir>` | the inference API ([serve/README.md](serve/README.md); Dockerfile there) | http://127.0.0.1:8000 | |
| `python scripts/benchmark.py --model both-8s --url <api>` | load test of the running API | `results/benchmark/` | 1–2 min |

`run_eval` writes per-clip predictions and metrics (accuracy, F1, ROC AUC with its interval, PR AUC, FNR, FPR) for all
clips, per corpus and per source. The data is in the Hugging Face cache (`HF_HOME=/workspace/.hf_home`: 69 GB
downloaded, 74 GB prepared Arrow, plus `datasets`' caches); `/workspace` is not a persistent volume on this instance.

Code: `src/turn_detection/` (`data/`, `model/`, `train/`), `scripts/`, `serve/`, `tests/`; `results/` holds the
tracked analyses and `artifacts/` the gitignored runs. Lint and format with ruff through pre-commit
(`pip install -r requirements-dev.txt && pre-commit install`): each commit runs `ruff check --fix` and `ruff format`
on the staged files and stops if they change anything.

## Limitations

- **English only**, and TTD's labels come from conversation timing (who speaks next), not human judgement.
- **Possible leakage within smart-turn:** no speaker or sentence ids, so train and validation are split per clip and
  may share speakers or sentences; only the holdout sources are truly unseen.
- **Seamless stays at 0.73–0.74** for every model trained with TTD (0.67–0.69 without): its label depends on the other
  speaker, and neither acoustic nor lexical cues separate it well (`FINDINGS.md`).
- **One run per configuration:** no seed variance; differences of a few thousandths between models are within noise.
- **TTD test is 67% SpokenWOZ** against 50% in train, so pooled TTD numbers lean on SpokenWOZ; compare per source.
