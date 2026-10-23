"""EoTConfig: everything needed to rebuild EoTModel, saved as config.json by save_pretrained."""

from transformers import PretrainedConfig, WhisperConfig

from data.constants import SAMPLE_RATE, WINDOW_S

HOP_LENGTH = 160  # Whisper features: 10 ms hop
CONV_STRIDE = 2  # Whisper's second conv halves the frame rate


class EoTConfig(PretrainedConfig):
    model_type = "eot"

    def __init__(
        self,
        encoder: dict | None = None,
        whisper: str = "openai/whisper-tiny",
        window_s: int = WINDOW_S,
        head_hidden: int = 256,
        dropout: float = 0.1,
        threshold: float = 0.5,
        padding: str = "left",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoder = encoder if encoder is not None else WhisperConfig().to_dict()  # Whisper encoder config
        self.whisper = whisper  # where the pretrained encoder came from
        self.window_s = window_s
        self.head_hidden = head_hidden
        self.dropout = dropout
        self.threshold = threshold  # P(complete) >= threshold -> "turn complete"; set from val after training
        self.padding = padding  # side short clips were zero-padded on in training; inference must match

    @property
    def n_positions(self) -> int:
        """Encoder positions for one window: 8 s -> 800 mel frames -> 400 after the stride-2 conv."""
        return self.window_s * SAMPLE_RATE // HOP_LENGTH // CONV_STRIDE
