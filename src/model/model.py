"""EoTModel: predicts whether the speaker's turn is complete from the last 8 s of audio (`input_frames`).

    Whisper log-mel features
    -> Whisper Tiny encoder
    -> learned weighted sum of its layers
    -> attention pooling over time
    -> 2-layer MLP
    -> logit (sigmoid: P(turn complete))

Which parts train (e.g. a frozen encoder) is decided by the training code, not here.

A regular transformers `PreTrainedModel`: `save_pretrained` / `from_pretrained` store EoTConfig (including
the encoder's config and the decision threshold chosen on val) next to the weights. The features come from
`WhisperFeatureExtractor` sized for 8 s (data.dataset.make_feature_extractor), which training saves
into the same directory.

Whisper's encoder is built for 30 s (3000 mel frames, 1500 positions). For an 8 s window (800 frames, 400
positions after the stride-2 conv) it is built with 400 positions, and `from_whisper` copies the pretrained
weights, keeping the first 400 rows of the (fixed, sinusoidal) position table.

    model = EoTModel.from_whisper("openai/whisper-tiny")     # pretrained encoder, new head
    out = model(input_features=feats, labels=labels)         # (B, 80, 800) -> out.logits (B,), out.loss
    model = EoTModel.from_pretrained("artifacts/runs/<run>/best")
"""

import torch
from torch import nn
from transformers import PreTrainedModel, WhisperConfig, WhisperModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.whisper.modeling_whisper import WhisperEncoder

from .config import EoTConfig
from .modules import AttentionPool


class EoTModel(PreTrainedModel):
    config_class = EoTConfig
    main_input_name = "input_features"
    base_model_prefix = "eot"

    def __init__(self, config: EoTConfig):
        super().__init__(config)

        encoder_config = WhisperConfig.from_dict(config.encoder)
        encoder_config.max_source_positions = config.n_positions
        self.encoder = WhisperEncoder(encoder_config)
        dim = encoder_config.d_model

        # One weight per encoder layer (softmax-normalized, initialized equal); the embedding output is excluded.
        self.layer_weights = nn.Parameter(torch.zeros(encoder_config.encoder_layers))

        self.pool = AttentionPool(dim)

        self.head = nn.Sequential(
            nn.Linear(dim, config.head_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.head_hidden, 1),
        )

        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        # The encoder never gets here: WhisperEncoder initializes itself (incl. the sinusoidal positions),
        # or it is loaded.
        if isinstance(module, EoTModel):
            nn.init.zeros_(module.layer_weights)  # equal layer weights after the softmax

        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            nn.init.zeros_(module.bias)

    def forward(self, input_features: torch.Tensor, labels: torch.Tensor | None = None) -> SequenceClassifierOutput:
        """(B, 80, input_frames) log-mel features -> logits (B,) of P(turn complete); BCE loss with labels."""
        encoder_out = self.encoder(input_features=input_features, output_hidden_states=True)
        hidden_states = encoder_out.hidden_states[1:]  # one (B, T, D) per layer; [0] is the embedding output

        # Layer-normalize each layer before mixing (parameter-free): their scales differ, and only the last
        # one has been through the encoder's final LayerNorm.
        layers = torch.stack([nn.functional.layer_norm(h, h.shape[-1:]) for h in hidden_states])  # (L, B, T, D)
        layer_weights = torch.softmax(self.layer_weights, dim=0)  # (L,)
        h = (layer_weights[:, None, None, None] * layers).sum(dim=0)  # (B, T, D)

        pooled = self.pool(h)  # (B, D)
        logits = self.head(pooled).squeeze(-1).float()  # (B,); float32 even under bf16 autocast

        loss = None
        if labels is not None:
            loss = nn.functional.binary_cross_entropy_with_logits(logits, labels.float())

        return SequenceClassifierOutput(loss=loss, logits=logits)

    @classmethod
    def from_whisper(cls, whisper: str = "openai/whisper-tiny", **config_kwargs) -> "EoTModel":
        """New model with the pretrained encoder of `whisper` (a Hub id or a local directory) and a fresh head."""
        pretrained = WhisperModel.from_pretrained(whisper).get_encoder()
        config = EoTConfig(whisper=whisper, **config_kwargs)
        model = cls(config)

        # Keep the first n_positions rows of the 1500-row position table.
        state = pretrained.state_dict()
        positions = state["embed_positions.weight"]
        state["embed_positions.weight"] = positions[: config.n_positions].clone()

        model.encoder.load_state_dict(state)
        return model
