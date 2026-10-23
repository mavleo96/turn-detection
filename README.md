# turn-detection

End-of-turn detection for voice-agent pipelines.

| Directory | What it holds |
|---|---|
| `src/` | The turn-detection model (not started yet) |
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
