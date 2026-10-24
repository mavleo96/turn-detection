from transformers import PretrainedConfig, WhisperConfig


class EoTConfig(PretrainedConfig):
    """Encoder, head and preprocessing settings of EoTModel, plus the decision threshold."""

    model_type = "eot"

    def __init__(
        self,
        whisper: str = "openai/whisper-tiny",
        input_frames: int = 800,
        head_hidden: int = 64,
        dropout: float = 0.1,
        threshold: float = 0.5,
        narrowband: bool = True,
        loudness_dbfs: float | None = -30.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.whisper = whisper  # Hub id or local directory
        self.input_frames = input_frames  # 10 ms mel frames: 800 = 8 s
        self.head_hidden = head_hidden
        self.dropout = dropout
        self.threshold = threshold  # set from val after training
        # Note: preprocessing used in training; evaluation and serving must match
        self.narrowband = narrowband
        self.loudness_dbfs = loudness_dbfs

        # TODO: take the saved/passed encoder config instead of re-deriving it: loading fails offline or when a local
        # `whisper` dir is gone, and tests/helpers.py's tiny encoder is ignored (see CLAUDE.md, Known issues)
        self.encoder_config = WhisperConfig.from_pretrained(whisper).to_dict()

    @property
    def n_positions(self) -> int:
        """Encoder positions per input: 800 mel frames -> 400."""
        # Note: WhisperEncoder's conv2 halves the frame rate
        conv_stride = 2
        return self.input_frames // conv_stride
