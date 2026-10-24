# turn-detection

End-of-turn detection for voice-agent pipelines.

| Directory | What it holds |
|---|---|
| `src/turn_detection/` | The installable package: `data/` (constants, audio preprocessing, Hugging Face dataset, `EoTDataset`, augmentation), `model/` (the EoT model), `train/` (HF Trainer setup, metrics; `python -m turn_detection.train`). `pip install -e ".[train]"` for training, `pip install .` (core: model + audio) for serving |
| [`scripts/`](scripts/README.md) | One-off runs: dataset exploration, splits, listening samples, evaluation, prediction, API benchmark |
| `results/` | Dataset exploration results |
| `tests/` | Tests for `src/` and `scripts/` |
| [`serve/`](serve/README.md) | Inference API (FastAPI, dynamic micro-batching) and its CPU Dockerfile |
| [`demo/`](demo/README.md) | Self-contained demo: live browser mic → streaming ASR + VAD with per-stage latency measurement, split into `frontend/` and `backend/` |

## Development

Lint and format with [ruff](https://docs.astral.sh/ruff/) (config in `ruff.toml`) through a pre-commit hook:

```bash
pip install -r requirements-dev.txt   # into the conda env `turn`
pre-commit install                    # once per clone
pre-commit run --all-files            # optional: check everything git tracks
```

On each commit the hook runs `ruff check --fix` and `ruff format` on the staged Python files. If either
one changes a file, the commit stops: review the changes, `git add` them, and commit again.
