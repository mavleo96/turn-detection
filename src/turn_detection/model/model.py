import torch
from torch import nn
from transformers import PreTrainedModel, WhisperConfig, WhisperModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.whisper.modeling_whisper import WhisperEncoder

from .config import EoTConfig
from .modules import AttentionPool


class EoTModel(PreTrainedModel):
    """Whisper encoder + layer mix + attention pooling + MLP -> logit of P(turn complete)."""

    config_class = EoTConfig
    main_input_name = "input_features"
    base_model_prefix = "eot"

    def __init__(self, config: EoTConfig):
        super().__init__(config)

        encoder_config = WhisperConfig.from_dict(config.encoder_config)
        encoder_config.max_source_positions = config.n_positions
        self.encoder = WhisperEncoder(encoder_config)
        dim = encoder_config.d_model

        # Note: one weight per layer, softmax-normalized; the embedding output is excluded
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
        # Note: never reached for the encoder: WhisperEncoder initializes itself (incl. the sinusoidal positions)
        if isinstance(module, EoTModel):
            nn.init.zeros_(module.layer_weights)

        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            nn.init.zeros_(module.bias)

    def forward(self, input_features: torch.Tensor, labels: torch.Tensor | None = None) -> SequenceClassifierOutput:
        """(B, 80, input_frames) log-mel features -> logits (B,) of P(turn complete); BCE loss with labels."""
        encoder_out = self.encoder(input_features=input_features, output_hidden_states=True)
        hidden_states = encoder_out.hidden_states[1:]  # L x (B, T, D); [0] is the embedding output

        # Note: normalize before mixing: layer scales differ, and only the last has the encoder's final LayerNorm
        layers = torch.stack(hidden_states)  # (L, B, T, D)
        layers = nn.functional.layer_norm(layers, layers.shape[-1:])  # (L, B, T, D)
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
        """Pretrained encoder of `whisper`, new head."""
        pretrained = WhisperModel.from_pretrained(whisper).get_encoder()
        config = EoTConfig(whisper=whisper, **config_kwargs)
        model = cls(config)

        state = pretrained.state_dict()
        state["embed_positions.weight"] = state["embed_positions.weight"][: config.n_positions]  # first 400 of 1500

        model.encoder.load_state_dict(state)
        return model
