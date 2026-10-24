import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from transformers import Trainer, TrainerCallback, TrainingArguments, WhisperFeatureExtractor
from transformers.trainer_utils import IntervalStrategy, SaveStrategy, SchedulerType

from turn_detection.data import DATASETS, LOUDNESS_DBFS, NARROWBAND, RUNS_DIR, WINDOW_S
from turn_detection.data.dataset import EoTCollator, EoTDataset, load_split
from turn_detection.data.transforms import GAIN_DB, NOISE_P, build_augment
from turn_detection.model import EoTModel

from .metrics import best_f1_threshold, eot_metrics


@dataclass
class EoTTrainingArguments(TrainingArguments):
    """TrainingArguments with this project's defaults and its model / data / augmentation settings, all logged to W&B
    and saved with the run (training_args.bin)."""

    # model (also stored in the saved model's config)
    whisper: str = field(default="openai/whisper-tiny", metadata={"help": "pretrained Whisper for the encoder"})
    freeze_encoder: bool = field(default=True, metadata={"help": "train only the layer weights, pooling and MLP"})
    dropout: float = 0.1

    # data
    datasets: list[str] = field(
        default_factory=lambda: list(DATASETS), metadata={"help": "training and validation data: smart-turn, ttd"}
    )
    max_train_samples: int | None = field(default=None, metadata={"help": "a random N of the training clips"})

    # preprocessing (also stored in the saved model's config)
    narrowband: bool = field(default=NARROWBAND, metadata={"help": "8 kHz round trip: band-limit all audio to 4 kHz"})
    loudness_dbfs: float | None = field(default=LOUDNESS_DBFS, metadata={"help": "normalize clips to this RMS level"})
    window_s: int = field(default=WINDOW_S, metadata={"help": "seconds of audio the model sees: 2, 4 or 8"})

    # augmentation
    augment: bool = True
    gain_db: float = GAIN_DB
    noise_p: float = NOISE_P

    # Trainer defaults for this project
    # Note: overridden fields keep the base class's annotations, which HfArgumentParser relies on
    # Note: a default_factory, not None: TrainingArguments.__post_init__ turns None into "trainer_output"
    output_dir: str | None = field(
        default_factory=lambda: str(RUNS_DIR / datetime.now().strftime("%Y%m%d-%H%M%S")),
        metadata={"help": "default: artifacts/runs/<timestamp>"},
    )
    num_train_epochs: float = 5
    per_device_train_batch_size: int = 64
    per_device_eval_batch_size: int = 128
    # Note: the shuffle. EoTDataset keeps the split's order; training draws a new random order every epoch
    # (RandomSampler), evaluation and predict keep the order (run_eval matches predictions to ids by position), and
    # augmentation is seeded by the item index, not its position in the batch
    train_sampling_strategy: str = "random"
    # Note: for the head; use a much lower LR with --freeze_encoder false
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    lr_scheduler_type: SchedulerType | str = "cosine"
    warmup_steps: float = 0.05
    bf16: bool = torch.cuda.is_available()
    eval_strategy: IntervalStrategy | str = "epoch"
    save_strategy: SaveStrategy | str = "epoch"
    save_total_limit: int | None = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str | None = "roc_auc"
    logging_steps: float = 50
    # Note: ~1,200 clips/s of decode + augment + features with 16 (the GPU needs ~520; RTX 3060)
    dataloader_num_workers: int = 16
    report_to: None | str | list[str] = "wandb"
    # Note: items carry "input_values" (waveforms) for the collator, not forward()
    remove_unused_columns: bool = False
    project: str = "turn-detection"  # W&B project


class SetEpochCallback(TrainerCallback):
    """Seeds the dataset's augmentation with the epoch (DataLoader workers are recreated each epoch)."""

    def __init__(self, dataset: EoTDataset):
        self.dataset = dataset

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.dataset.set_epoch(int(state.epoch or 0))


def train(args: EoTTrainingArguments, model: EoTModel | None = None) -> Path:
    """Train, pick the validation threshold, save `best/`; returns the run directory. `model`: injected by tests,
    by default a new model on the pretrained `whisper` encoder."""
    args.run_name = args.run_name or Path(args.output_dir).name
    # Note: the W&B callback ignores args.project and reads WANDB_PROJECT; the flag wins over the environment
    os.environ["WANDB_PROJECT"] = args.project
    out = Path(args.output_dir)

    # Data
    train_split = load_split("train", args.datasets)
    if args.max_train_samples is not None:
        # Note: a random subset: the split is ordered by dataset and source
        n = min(args.max_train_samples, len(train_split))
        train_split = train_split.shuffle(seed=args.seed).select(range(n))

    augment = None
    if args.augment:
        augment = build_augment(args.gain_db, args.noise_p)
    if augment is not None and args.dataloader_persistent_workers:
        # Note: SetEpochCallback only reaches the workers because they are recreated every epoch
        raise ValueError("dataloader_persistent_workers would repeat the same augmentation every epoch")

    preprocessing = {"narrowband": args.narrowband, "loudness_dbfs": args.loudness_dbfs, "window_s": args.window_s}
    train_ds = EoTDataset(train_split, augment, args.seed, **preprocessing)
    val_ds = EoTDataset(load_split("validation", args.datasets), **preprocessing)

    # Note: the front end of the same Whisper as the encoder, sized for the model's window instead of 30 s
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.whisper, chunk_length=args.window_s)

    # Model
    if model is None:
        model = EoTModel.from_whisper(args.whisper, dropout=args.dropout, window_s=args.window_s)
    if model.config.window_s != args.window_s:
        raise ValueError(f"the model is built for {model.config.window_s} s windows, --window_s is {args.window_s}")
    # Note: saved with the model, so evaluation and inference preprocess the same way
    model.config.narrowband = args.narrowband
    model.config.loudness_dbfs = args.loudness_dbfs
    if args.freeze_encoder:
        model.encoder.requires_grad_(False)

    # Training
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=EoTCollator(feature_extractor),  # waveforms -> log-mel features, per batch
        processing_class=feature_extractor,  # saved with every checkpoint and with best/
        # Note: logits -> P(complete), clipped so exp can't overflow in float32; metrics at threshold 0.5
        compute_metrics=lambda p: eot_metrics(p.label_ids, 1 / (1 + np.exp(-np.clip(p.predictions, -80, 80)))),
        callbacks=[SetEpochCallback(train_ds)],
    )
    trainer.train()  # ends with the best epoch loaded (load_best_model_at_end)

    # Decision threshold: chosen on val, saved in the model's config
    val_logits = trainer.predict(val_ds).predictions  # (N,)
    val_probs = 1 / (1 + np.exp(-np.clip(val_logits, -80, 80)))
    threshold = best_f1_threshold(val_ds.labels, val_probs)
    trainer.model.config.threshold = threshold
    trainer.save_model(str(out / "best"))

    # Final val metrics at that threshold
    val_metrics = eot_metrics(val_ds.labels, val_probs, threshold)
    results = {f"val_at_threshold_{k}": v for k, v in val_metrics.items()}

    trainer.log(results)  # also to wandb
    trainer.save_metrics("final", results)

    return out
