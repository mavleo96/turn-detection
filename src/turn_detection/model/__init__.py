"""End-of-turn classifier: P(turn complete) from the last 8 s of audio.

    log-mel (B, 80, 800) -> Whisper encoder -> softmax-weighted sum of its layers -> attention pooling -> MLP -> logit

- config.py: EoTConfig (encoder, head, preprocessing, threshold), saved as config.json by save_pretrained
- modules.py: AttentionPool
- model.py: EoTModel; the encoder is rebuilt for 400 positions (8 s) instead of 1500 (30 s), freezing is the
  trainer's job

    model = EoTModel.from_whisper("openai/whisper-tiny")     # pretrained encoder, new head
    out = model(input_features=feats, labels=labels)         # out.logits (B,), out.loss
    model = EoTModel.from_pretrained("artifacts/runs/<run>/best")
"""

from .config import EoTConfig
from .model import EoTModel

__all__ = ["EoTConfig", "EoTModel"]
