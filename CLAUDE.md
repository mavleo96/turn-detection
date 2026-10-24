# turn-detection: notes for Claude

End-of-turn (EoT) detection for voice agents: at a VAD pause, is the user's turn complete? Input = the last 8 s of
16 kHz mono audio, output = P(turn complete). Positive class = complete.

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
- Ruff: `/venv/turn/bin/ruff` (from requirements-dev.txt, same version as pre-commit, which runs ruff check --fix + format). Line length 120.
- Tests: `/venv/turn/bin/python -m pytest -q` (~30 s, CPU, no network).
- GPU: RTX 4090 (256 CPUs, 503 GB RAM); earlier timings were on an RTX 3060. `/workspace` is not a volume (lost on recycle).
- HF cache: `HF_HOME=/workspace/.hf_home`, empty on a fresh instance; needs TTD ~26 GB prepared, smart-turn train 39 GB +
  test 5 GB. Then run `scripts/split.py` for the smart-turn manifests.
- W&B logs to a CoreWeave-hosted server (`api.forge.coreweave.com`, team `mavleo96-team`, project `turn-detection`);
  `train()` sets `WANDB_PROJECT` from `--project` (the HF W&B callback ignores `args.project`).
- librosa `pyin` segfaults in `/venv/turn` (numba 0.68); use a numpy pitch tracker. Use spawned (not forked)
  process pools when librosa/numba is involved.
- `environment.yml` = conda export of `turn`; `requirements.txt` = pinned training env; `pyproject.toml` = core deps
  (model + audio) + `[train]` extra.

## Layout
- `src/turn_detection/` (installed with `pip install --no-deps -e .`):
  - `data/constants.py` `DATASETS` registry: `smart-turn` (pipecat-ai v3.2, English only, label `endpoint_bool`,
    source `dataset`) and `ttd` (`mavleo96/turn-detection-data`, label `label`, source `data_source`).
  - `data/audio.py` per-clip steps shared by training, eval and serving: `decode`, `prepare_clip` (last 8 s ->
    `narrowband` 16k->8k->16k -> `normalize_loudness` to -30 dBFS RMS), `pad_window` (left zero-pad to 8 s).
  - `data/dataset.py` `load_split(split, dataset=...)`, `EoTDataset` (augment between prepare_clip and padding, so
    noise SNR follows the speech), `EoTConcat` (joins datasets), `EoTCollator` (WhisperFeatureExtractor, 8 s).
  - `data/transforms.py` `build_augment(gain_db=6, noise_p=0.3)`: gain ±6 dB + colored noise (white..brown) 20-40 dB.
  - `model/` `EoTConfig` (whisper name, input_frames 800, head_hidden 64, dropout, threshold, narrowband,
    loudness_dbfs; encoder config derived from the whisper name and saved), `EoTModel` (frozen-by-trainer Whisper
    Tiny encoder -> softmax layer weights over 4 layers -> attention pooling -> 384-64-1 MLP).
  - `train/` HF Trainer setup (`trainer.py`), sklearn metrics (`metrics.py`), CLI `python -m turn_detection.train`.
- `scripts/`: `split.py` (smart-turn train/val = sorted by id, last 10% val; writes `artifacts/manifests/`),
  `run_eval.py` (both test sets, per-source breakdowns, `--benchmark`), `predict.py`, `explore_dataset.py`, `samples.py`.
- `demo/`: separate browser mic -> Silero VAD + Nemotron streaming ASR harness (own env/tests).
- `serve/` (FastAPI + micro-batching, CPU Dockerfile) lives on the `serve` branch only (ignored on `main`).
- Generated, gitignored: `artifacts/` (manifests, runs, models/untrained), `results/` (analyses).

## Training and evaluation
- `python -m turn_detection.train --dataset ttd|smart-turn|ttd,smart-turn` (default `ttd`). 5 epochs, batch 64,
  AdamW lr 1e-3, wd 0.01, 5% warmup + cosine to 0, grad clip 1.0, bf16, frozen encoder (25k trainable params),
  best epoch by val ROC AUC, threshold = best F1 on val (saved in `best/config.json`).
- TTD uses its own train/validation splits; smart-turn uses `artifacts/manifests/` (run `scripts/split.py` first).
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
- TTD clips end where its VAD fires (Silero, 224 ms after speech end, 25 dB gate); the demo VAD (threshold 0.5,
  min_silence 200 ms → fires after 256 ms) does not match yet. Serving must send the turn so far, ending at the stop.
- TTD: SpokenWOZ 58% complete vs Seamless 42% (source-only AUC 0.58, not rebalanced); Seamless is ~24 dB quieter
  than SpokenWOZ before normalization; splits share no conversation (Seamless: no person); Hub rows are ordered by
  source (SpokenWOZ first), so never take "the first N rows".
- smart-turn English: 69% human speech; clips end ~0.7 s after the last speech; train/val split is per clip
  (possible speaker leakage); Hub rows are shuffled.
- Full analysis (peer-reviewed): `results/ttd_vs_smart_turn/FINDINGS.md`.

## Open ideas (not done)
- Make `--dataset ttd,smart-turn` the default; threshold for a target FPR instead of best F1.
- Unfreeze the encoder with a separate low encoder LR (the trainer has one LR for all params today).
- Re-cut smart-turn clips to VAD-style endings; match the demo VAD to the dataset's.
