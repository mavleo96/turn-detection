"""EoTConfig: everything needed to rebuild EoTModel, saved as config.json by save_pretrained."""

from transformers import PretrainedConfig, WhisperConfig

CONV_STRIDE = 2  # WhisperEncoder's conv2 halves the frame rate (hard-coded there, not in WhisperConfig)


class EoTConfig(PretrainedConfig):
    model_type = "eot"

    def __init__(
        self,
        whisper: str = "openai/whisper-tiny",
        input_frames: int = 800,
        head_hidden: int = 64,
        dropout: float = 0.1,
        threshold: float = 0.5,
        padding: str = "left",
        narrowband: bool = True,
        encoder: dict | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.whisper = whisper  # pretrained Whisper the encoder comes from (a Hub id or a local directory)
        self.input_frames = input_frames  # log-mel frames per input, 10 ms each: 800 = the last 8 s
        self.head_hidden = head_hidden
        self.dropout = dropout
        self.threshold = threshold  # P(complete) >= threshold -> "turn complete"; set from val after training
        self.padding = padding  # side short clips were zero-padded on in training; inference must match
        self.narrowband = narrowband  # audio band-limited to 4 kHz (8 kHz round trip) in training; inference must match

        # The Whisper encoder's config: derived from `whisper` for a new config, the saved one in from_pretrained.
        if encoder is None:
            encoder = WhisperConfig.from_pretrained(whisper).to_dict()
        self.encoder = encoder

    @property
    def n_positions(self) -> int:
        """Encoder positions for one input: 800 mel frames -> 400 after the stride-2 conv."""
        return self.input_frames // CONV_STRIDE
