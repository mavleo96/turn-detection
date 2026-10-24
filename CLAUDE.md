# turn-detection: notes for Claude

End-of-turn (EoT) detection for voice agents: at a VAD pause, is the user's turn complete? Input = the last 2, 4 or
8 s (`window_s`, default 8) of 16 kHz mono audio, output = P(turn complete). Positive class = complete.

## Working with this user
- Discuss plans and open questions before writing code; wait for agreement. Exception: "write the code, I'll review".
- Commit only when asked (on `main`); push only when asked. Never start long jobs/servers unasked; stop temporary ones.
- Prefer less code: no thin wrappers, no one-line helpers, no options nobody uses. Prefer libraries over hand-written code.
- Style: one step per line with shape comments (`# (B, T, D)`), blank lines between blocks, no `einsum`, `from_*`
  classmethods last in a class, no dtype casts that bf16 autocast makes no-ops.
- Analyses: check measurement methods (e.g. VAD, not energy gates) and dataset composition before stating facts;
  label interpretation vs measurement; the user asks for peer review and dislikes exaggeration.

## Environment
- Python env: `/venv/turn` (conda env `turn`, torch 2.14.1+cu130). `/venv/main` has no transformers.
- Ruff: `/venv/turn/bin/ruff` (requirements-dev.txt, same version as pre-commit, which runs ruff check --fix +
  format). Line length 120.
- Tests: `/venv/turn/bin/python -m pytest -q` (~25 s, CPU, no network: they pass with HF_HUB_OFFLINE=1 and an empty
  HF_HOME).
- GPU: RTX 4090; 61-CPU cgroup quota (`nproc` says 256), 503 GB RAM; earlier timings were on an RTX 3060.
  `/workspace` is not a volume (lost on recycle).
- HF cache: `HF_HOME=/workspace/.hf_home`, empty on a fresh instance; needs TTD ~26 GB prepared, smart-turn train
  39 GB + test 5 GB. Building the splits takes ~20 s the first time (cached by `datasets` after that).
- W&B logs to a CoreWeave-hosted server (`api.forge.coreweave.com`, team `mavleo96-team`, project `turn-detection`);
  `train()` sets `WANDB_PROJECT` from `--project` (the HF W&B callback ignores `args.project`).
- librosa `pyin` segfaults in `/venv/turn` (numba 0.68); use a numpy pitch tracker. Use spawned (not forked)
  process pools when librosa/numba is involved.
- `environment.yml` = conda export of `turn`; `requirements.txt` = pinned training env; `pyproject.toml` = core deps
  (model + audio) + `[train]` extra.

## Layout
- `src/turn_detection/` (installed with `pip install --no-deps -e .`):
  - `data/__init__.py` constants and paths: repos `SMART_TURN_TRAIN` / `SMART_TURN_TEST` (pipecat-ai v3.2, one
    split "train" each) and `TTD` (`mavleo96/turn-detection-data`), `DATASETS` (`smart-turn`, `ttd`), `HOLDOUT`
    (`human_5`, `rime_2`), `VAL_FRAC`, and the only preprocessing defaults: `WINDOW_S` 8, `NARROWBAND` True,
    `LOUDNESS_DBFS` -30.
  - `data/audio.py` per-clip steps shared by training, eval and serving: `prepare_clip(x, *, window_s, narrowband,
    loudness_dbfs)` (last `window_s` s -> `band_limit` 16k->8k->16k -> `normalize_loudness` to -30 dBFS RMS),
    `pad_window(x, window_s)` (left zero-pad).
  - `data/dataset.py` `load_split(split, datasets)` (train / validation / test of smart-turn and/or TTD as `id,
    audio, dataset, label`; audio `AUDIO` = 16 kHz mono, decoded by `datasets`/torchcodec; label
    `ClassLabel(incomplete, complete)`), `EoTDataset` (augments between prepare_clip and padding, so noise SNR
    follows the speech), `EoTCollator` (WhisperFeatureExtractor sized for the window; rejects other lengths; bool
    labels).
  - `data/transforms.py` `build_augment(gain_db=6, noise_p=0.3)`: gain ±6 dB + colored noise (white..brown) 20-40 dB.
  - `model/` `EoTConfig` (whisper name; `encoder_config` = a `WhisperConfig` sub-config with 50 * window_s
    positions, saved and restored, default = Whisper Tiny's architecture, no download; window_s 2/4/8 -> 100 *
    window_s mel frames; head_hidden 64, dropout, threshold, narrowband, loudness_dbfs), `EoTModel`
    (frozen-by-trainer Whisper Tiny encoder -> softmax layer weights over 4 layers -> attention pooling -> MLP).
  - `train/` HF Trainer setup (`trainer.py`), sklearn metrics (`metrics.py`), CLI `python -m turn_detection.train`.
- `scripts/`: `run_eval.py` (the test split, `--datasets`, per-source breakdown, `--benchmark`) and the exploration
  scripts `explore_dataset.py`, `samples.py` (raw repos via `load_dataset`; tests patch
  `load_dataset` in `dataset`, `explore_dataset` and `samples`).
- The browser mic -> Silero VAD + Nemotron streaming ASR demo was removed from `main`; it is on the `backup` branch
  and in git history (before the removal commit).
- `serve/` (FastAPI + micro-batching, CPU Dockerfile) lives on the `serve` branch only (ignored on `main`).
- Generated, gitignored: `artifacts/` (runs, models/untrained), `results/` (analyses; tracked: `data_exploration/`
  = `explore_dataset.py` output + `EXPLORE.md`, `ttd_vs_smart_turn/` = `FINDINGS.md`).

## Training and evaluation
- `python -m turn_detection.train [--datasets smart-turn ttd] [--window_s 2|4|8]` (defaults: both, 8). One run
  per window; `window_s` is saved in the model config and `best/`'s feature extractor, so run_eval follows it;
  `EoTCollator` rejects audio of another length. 5 epochs, batch 64, AdamW lr 1e-3, wd 0.01, 5% warmup + cosine
  to 0, grad clip 1.0, bf16, frozen encoder (25k trainable params), best epoch by val ROC AUC, threshold = best F1
  on val (saved in `best/config.json`); `best/` is logged to the W&B run as model artifact `<run name>-best`
  (after the threshold is set; `WANDB_LOG_MODEL` would upload it before). The 3 runs of 2026-10-24 were uploaded
  by hand the same way.
- Splits (`load_split`): train = smart-turn English train without `HOLDOUT`, 90% (random, seed 42) + TTD train
  (52,893 + 333,280; 4 TTD clips torchcodec can't decode are left out: `UNDECODABLE`); validation = the other
  10% + TTD validation (5,877 + 37,600); test = all English smart-turn test (holdout sources included: 402 human_5
  + 394 rime_2, never seen in train or val) + TTD test (7,820 + 60,574).
- Metrics everywhere: accuracy, f1, fpr (incomplete predicted complete = interrupts), fnr (complete predicted
  incomplete = waits too long), roc_auc, pr_auc.
- Time on the RTX 3060: TTD ~30 min / 5 epochs (~1,100 clips/s), smart-turn ~7 min, both ~35 min; eval ~2 min.
- `python scripts/run_eval.py --model artifacts/runs/<run>/best --benchmark`.

## Results so far (test ROC AUC; runs `*-5ep-20261024-075039`)
| trained on | TTD (SpokenWOZ / Seamless) | smart-turn |
|---|---|---|
| ttd | 0.885 (0.927 / 0.742) | 0.810 |
| smart-turn | 0.777 (0.813 / 0.682) | 0.977 |
| both | **0.887** (0.928 / 0.743) | 0.968 |
Layer weights (layers 1-4): ttd 0.27/0.20/0.21/0.32, smart-turn 0.15/0.16/0.25/0.43, both 0.21/0.18/0.21/0.40.
Batch-1 latency: ~14 ms GPU, ~120 ms CPU. F1 thresholds lean to interrupting (TTD FPR ~0.28-0.30 vs FNR ~0.12).

## Data facts worth remembering
- TTD clips end where its VAD fires (Silero, 224 ms after speech end, 25 dB gate); the demo VAD (`backup` branch;
  threshold 0.5, min_silence 200 ms → fires after 256 ms) does not match yet. Serving must send the turn so far,
  ending at the stop.
- TTD: SpokenWOZ 58% complete vs Seamless 42% (source-only AUC 0.58, not rebalanced); Seamless is ~24 dB quieter
  than SpokenWOZ before normalization; splits share no conversation (Seamless: no person); Hub rows are ordered by
  source (SpokenWOZ first), so never take "the first N rows".
- smart-turn English: 69% human speech; clips end ~0.7 s after the last speech; train/val split is per clip
  (possible speaker leakage, except for the holdout sources); Hub rows are shuffled; no train/test id overlap.
- Full analysis (peer-reviewed): `results/ttd_vs_smart_turn/FINDINGS.md`.

## Open ideas (not done)
- Threshold for a target FPR instead of best F1.
- Unfreeze the encoder with a separate low encoder LR (the trainer has one LR for all params today).
- Re-cut smart-turn clips to VAD-style endings; match the demo VAD to the dataset's.

## Known issues (deferred)
- The `serve` branch is behind `main`: it imports `turn_detection.data.audio.decode` (removed: decoding is the
  caller's now) and `turn_detection.data.constants` (now `turn_detection.data`), and calls `prepare_clip` /
  `pad_window` the old way (settings are keyword-only now, incl. `window_s` from the model config). Serving should
  decode like training: torchcodec `AudioDecoder(..., sample_rate=16000, num_channels=1)` (stereo -> (L + R) /
  sqrt 2, FFmpeg resampling); if torchcodec/FFmpeg can't go in the serving image, serve defines its own decoder
  (soundfile channel mean differs only for stereo, by 3 dB, which loudness normalization removes). All dataset
  clips are 16 kHz mono (smart-turn: `explore_dataset.py`, all 302,473; TTD: 600 sampled).
