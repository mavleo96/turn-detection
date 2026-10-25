"""End-of-turn classifier: P(turn complete) from the last `window_s` seconds of audio (2, 4 or 8; default 8).

    log-mel (B, 80, 100 * window_s) -> Whisper encoder -> weighted layer sum -> attention pooling -> MLP -> logit

- config.py: EoTConfig (encoder as a WhisperConfig, head, preprocessing, threshold), saved as config.json by
  save_pretrained, so loading needs no download
- modules.py: AttentionPool
- model.py: EoTModel; the encoder is rebuilt for 50 * window_s positions (400 for 8 s) instead of 1500 (30 s);
  freezing is the trainer's job

    model = EoTModel.from_whisper("openai/whisper-tiny")     # pretrained encoder, new head
    out = model(input_features=feats, labels=labels)         # out.logits (B,), out.loss
    model = EoTModel.from_pretrained("artifacts/runs/<run>/best")
"""

from .config import EoTConfig
from .model import EoTModel

__all__ = ["EoTConfig", "EoTModel"]
