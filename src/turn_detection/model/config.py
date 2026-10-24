from transformers import PretrainedConfig, WhisperConfig

from turn_detection.data import LOUDNESS_DBFS, NARROWBAND, WINDOW_S


class EoTConfig(PretrainedConfig):
    """Encoder, head and preprocessing settings of EoTModel, plus the decision threshold."""

    model_type = "eot"
    sub_configs = {"encoder_config": WhisperConfig}

    def __init__(
        self,
        whisper: str = "openai/whisper-tiny",
        encoder_config: dict | None = None,
        window_s: int = WINDOW_S,
        head_hidden: int = 64,
        dropout: float = 0.1,
        threshold: float = 0.5,
        narrowband: bool = NARROWBAND,
        loudness_dbfs: float | None = LOUDNESS_DBFS,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.whisper = whisper  # the pretrained encoder's Hub id or local directory (from_whisper)
        self.window_s = window_s  # seconds of audio the model sees: 2, 4 or 8
        self.head_hidden = head_hidden
        self.dropout = dropout
        self.threshold = threshold  # set from val after training
        # Note: preprocessing used in training; evaluation and serving must match
        self.narrowband = narrowband
        self.loudness_dbfs = loudness_dbfs

        # Note: the encoder as built, saved in config.json, so loading needs no download; the default
        # WhisperConfig() is Whisper Tiny's architecture, and from_whisper passes the pretrained one
        self.encoder_config = WhisperConfig(**(encoder_config or {}))
        self.encoder_config.max_source_positions = self.n_positions

    @property
    def n_positions(self) -> int:
        """Encoder positions per input: 8 s -> 800 mel frames -> 400."""
        # Note: Whisper's log-mel has 100 frames per second, and WhisperEncoder's conv2 halves that
        return self.window_s * 100 // 2
