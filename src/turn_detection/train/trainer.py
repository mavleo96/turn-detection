"""Training with the Hugging Face `Trainer`: argument dataclasses, dataset setup and the run itself.

Inputs (`dataset`, data.constants.DATASETS), from the Hugging Face cache (data.dataset.load_split):
  - ttd (default): every clip of its train split; its validation split for model selection and the threshold.
  - smart-turn: its train split, with the id lists written by scripts/split.py (manifests/{train,val}.parquet).

What is ours on top of `Trainer`: freezing the Whisper encoder (`freeze_encoder`, default on: the
Trainer's default AdamW then only gets the layer weights, pooling and MLP), a callback that tells the
dataset the epoch (so augmentation differs per epoch), and, after training, the decision threshold
chosen on val, stored in the saved model's config.

A run directory holds the Trainer's checkpoints, `best/` (the best epoch by val ROC AUC: model with
the threshold in its config, plus the feature extractor, both via save_pretrained), and
`final_results.json`; metrics also go to Weights & Biases.
"""

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset as HFDataset
from transformers import Trainer, TrainerCallback, TrainingArguments, WhisperFeatureExtractor
from transformers.trainer_utils import IntervalStrategy, SaveStrategy, SchedulerType

from turn_detection.data.constants import DATASETS, MANIFESTS_DIR, RUNS_DIR, WINDOW_S
from turn_detection.data.dataset import EoTCollator, EoTConcat, EoTDataset, load_split
from turn_detection.data.transforms import GAIN_DB, NOISE_P, build_augment
from turn_detection.model import EoTModel

from .metrics import best_f1_threshold, compute_metrics, eot_metrics, sigmoid


@dataclass
class EoTTrainingArguments(TrainingArguments):
    """TrainingArguments with this project's defaults plus its model / data / augmentation settings, so
    every setting is logged to wandb and saved with the run (training_args.bin). Overridden fields keep
    the base class's annotations, which HfArgumentParser relies on; every Trainer flag still works."""

    # model (also stored in the saved model's config)
    whisper: str = field(default="openai/whisper-tiny", metadata={"help": "pretrained Whisper for the encoder"})
    freeze_encoder: bool = field(default=True, metadata={"help": "train only the layer weights, pooling and MLP"})
    dropout: float = 0.1

    # data (str paths: TrainingArguments are serialized to JSON for wandb)
    dataset: str = field(
        default="ttd", metadata={"help": "training data: ttd, smart-turn, or both as ttd,smart-turn (data.constants)"}
    )
    manifests_dir: str = str(MANIFESTS_DIR)  # smart-turn
    max_train_samples: int | None = field(
        default=None, metadata={"help": "a random N of the training clips (per dataset)"}
    )

    # preprocessing (also stored in the saved model's config)
    narrowband: bool = field(default=True, metadata={"help": "8 kHz round trip: band-limit all audio to 4 kHz"})
    loudness_dbfs: float | None = field(default=-30.0, metadata={"help": "normalize clips to this RMS level"})

    # augmentation
    augment: bool = True
    gain_db: float = GAIN_DB
    noise_p: float = NOISE_P

    # Trainer defaults for this project
    output_dir: str | None = field(default=None, metadata={"help": "default: artifacts/runs/<timestamp>"})
    num_train_epochs: float = 5
    per_device_train_batch_size: int = 64
    per_device_eval_batch_size: int = 128
    learning_rate: float = 1e-3  # for the head; use a much lower LR with --freeze_encoder false
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
    dataloader_num_workers: int = 16  # ~1,200 clips/s of decode + augment + features (GPU needs ~520)
    report_to: None | str | list[str] = "wandb"
    remove_unused_columns: bool = False  # items carry "input_values" (waveforms) for the collator, not forward()
    project: str = "turn-detection"  # W&B project (set as WANDB_PROJECT in train(); Trackio reads it directly)


class SetEpochCallback(TrainerCallback):
    """Seeds the dataset's augmentation with the epoch (DataLoader workers are recreated each epoch)."""

    def __init__(self, dataset: EoTConcat):
        self.dataset = dataset

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.dataset.set_epoch(int(state.epoch or 0))


def load_sets(name: str, args: EoTTrainingArguments) -> dict[str, tuple[HFDataset, pd.Series]]:
    """Dataset `name`: {train, val} -> (the HF split holding the set's clips, their ids)."""
    if name not in DATASETS:
        raise ValueError(f"dataset must be one of {sorted(DATASETS)}, got {name!r}")

    if name == "ttd":
        train = load_split("train", dataset="ttd")
        val = load_split("validation", dataset="ttd")
        sets = {"train": (train, pd.Series(train["id"])), "val": (val, pd.Series(val["id"]))}
    else:  # smart-turn: train and val both come from its train split (scripts/split.py)
        source = load_split("train", dataset="smart-turn")
        sets = {}
        for name in ("train", "val"):
            ids = pd.read_parquet(Path(args.manifests_dir) / f"{name}.parquet", columns=["id"])["id"]
            sets[name] = (source, ids)

    # A random subset: the ttd train split is ordered by source (all SpokenWOZ rows first).
    if args.max_train_samples is not None:
        ds, ids = sets["train"]
        n = min(args.max_train_samples, len(ids))
        sets["train"] = (ds, ids.sample(n=n, random_state=args.seed))

    return sets


def train(args: EoTTrainingArguments, model: EoTModel | None = None) -> Path:
    """Train, pick the val threshold, save `best/`; returns the run directory. `model` can be injected (tests);
    by default a new model on the pretrained `whisper` encoder."""
    if args.output_dir is None:
        args.output_dir = str(RUNS_DIR / datetime.now().strftime("%Y%m%d-%H%M%S"))
    args.run_name = args.run_name or Path(args.output_dir).name
    os.environ.setdefault("WANDB_PROJECT", args.project)  # the W&B callback ignores args.project
    out = Path(args.output_dir)

    # Data: one EoTDataset per dataset in --dataset, joined
    names = args.dataset.split(",")
    sets = {name: load_sets(name, args) for name in names}

    augment = None
    if args.augment:
        augment = build_augment(args.gain_db, args.noise_p)
    if augment is not None and args.dataloader_persistent_workers:
        # SetEpochCallback only reaches the workers because they are recreated every epoch.
        raise ValueError("dataloader_persistent_workers would repeat the same augmentation every epoch")

    train_parts, val_parts = [], []
    for k, name in enumerate(names):
        kwargs = {
            "seed": args.seed + k,  # different augmentation for item i of each dataset
            "narrowband": args.narrowband,
            "loudness_dbfs": args.loudness_dbfs,
            "label_column": DATASETS[name]["label"],
        }
        train_parts.append(EoTDataset(*sets[name]["train"], augment, **kwargs))
        val_parts.append(EoTDataset(*sets[name]["val"], **kwargs))
    train_ds = EoTConcat(train_parts)
    val_ds = EoTConcat(val_parts)

    # The front end of the same Whisper as the encoder, sized for the model's window instead of 30 s.
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.whisper, chunk_length=WINDOW_S)

    # Model
    if model is None:
        model = EoTModel.from_whisper(args.whisper, dropout=args.dropout)
    # Saved with the model, so evaluation and inference preprocess the same way.
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
        compute_metrics=compute_metrics,
        callbacks=[SetEpochCallback(train_ds)],
    )
    trainer.train()  # ends with the best epoch loaded (load_best_model_at_end)

    # Decision threshold: chosen on val, saved in the model's config
    val_probs = sigmoid(trainer.predict(val_ds).predictions)
    threshold = best_f1_threshold(val_ds.labels, val_probs)
    trainer.model.config.threshold = threshold
    trainer.save_model(str(out / "best"))

    # Final val metrics at that threshold
    val_metrics = eot_metrics(val_ds.labels, val_probs, threshold)
    results = {f"val_at_threshold_{k}": v for k, v in val_metrics.items()}

    trainer.log(results)  # also to wandb
    trainer.save_metrics("final", results)

    return out
