"""Training with the Hugging Face `Trainer`: argument dataclasses, dataset setup and the run itself.

Inputs: the train split from the Hugging Face cache (data.dataset.load_split) and the id lists written by
scripts/split.py: manifests/{train,val}.parquet, or folds/<source>/{train,val,heldout}.parquet.

What is ours on top of `Trainer`: freezing the Whisper encoder (`freeze_encoder`, default on: the
Trainer's default AdamW then only gets the layer weights, pooling and MLP), a callback that tells the
dataset the epoch (so augmentation differs per epoch), and, after training, the decision threshold
chosen on val, stored in the saved model's config.

A run directory holds the Trainer's checkpoints, `best/` (the best epoch by val ROC AUC: model with
the threshold in its config, plus the feature extractor, both via save_pretrained), and
`final_results.json`; metrics also go to Weights & Biases.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset as HFDataset
from transformers import Trainer, TrainerCallback, TrainingArguments
from transformers.trainer_utils import IntervalStrategy, SaveStrategy, SchedulerType

from data.constants import FOLDS_DIR, MANIFESTS_DIR, RUNS_DIR
from data.dataset import EoTCollator, EoTDataset, load_split, make_feature_extractor
from data.transforms import AugmentConfig, build_augment
from model import EoTModel

from .metrics import best_f1_threshold, compute_metrics, eot_metrics, sigmoid


@dataclass
class EoTTrainingArguments(TrainingArguments):
    """TrainingArguments with this project's defaults plus its model / data / augmentation settings, so
    every setting is logged to wandb and saved with the run (training_args.bin). Overridden fields keep
    the base class's annotations, which HfArgumentParser relies on; every Trainer flag still works."""

    # model (also stored in the saved model's config)
    whisper: str = field(default="openai/whisper-tiny", metadata={"help": "pretrained Whisper for the encoder"})
    freeze_encoder: bool = field(default=True, metadata={"help": "train only the layer weights, pooling and MLP"})
    head_hidden: int = 256
    dropout: float = 0.1
    # data (str paths: TrainingArguments are serialized to JSON for wandb)
    manifests_dir: str = str(MANIFESTS_DIR)
    folds_dir: str = str(FOLDS_DIR)
    fold: str | None = field(default=None, metadata={"help": "hold out this human source (folds/<fold>/)"})
    max_train_samples: int | None = field(default=None, metadata={"help": "only the first N training clips"})
    # augmentation
    padding: str = field(default="left", metadata={"help": "zero-pad short clips on the left or the right"})
    augment: bool = True
    gain_db: float = AugmentConfig.gain_db
    noise_p: float = AugmentConfig.noise_p

    # Trainer defaults for this project
    output_dir: str | None = field(default=None, metadata={"help": "default: artifacts/runs/<timestamp>[-fold-<f>]"})
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
    project: str = "turn-detection"


class SetEpochCallback(TrainerCallback):
    """Seeds the dataset's augmentation with the epoch (DataLoader workers are recreated each epoch)."""

    def __init__(self, dataset: EoTDataset):
        self.dataset = dataset

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.dataset.set_epoch(int(state.epoch or 0))


def load_ids(args: EoTTrainingArguments) -> dict[str, pd.Series]:
    """Clip ids of {train, val[, heldout]} from the manifests, or from the fold's files."""
    src = Path(args.folds_dir) / args.fold if args.fold else Path(args.manifests_dir)
    names = ["train", "val"] + (["heldout"] if args.fold else [])
    ids = {name: pd.read_parquet(src / f"{name}.parquet", columns=["id"])["id"] for name in names}
    if args.max_train_samples is not None:
        ids["train"] = ids["train"].iloc[: args.max_train_samples]
    return ids


def default_output_dir(fold: str | None) -> str:
    return str(RUNS_DIR / (datetime.now().strftime("%Y%m%d-%H%M%S") + (f"-fold-{fold}" if fold else "")))


def predict_probs(trainer: Trainer, dataset: EoTDataset) -> np.ndarray:
    return sigmoid(trainer.predict(dataset).predictions)


def train(args: EoTTrainingArguments, model: EoTModel | None = None, source: HFDataset | None = None) -> Path:
    """Train, pick the val threshold, save `best/`; returns the run directory. `model` and `source`
    (the train split) can be injected (tests); by default Whisper Tiny and the cached split are loaded."""
    if args.output_dir is None:
        args.output_dir = default_output_dir(args.fold)
    args.run_name = args.run_name or Path(args.output_dir).name
    source = load_split("train") if source is None else source
    ids = load_ids(args)
    augment = AugmentConfig(gain_db=args.gain_db, noise_p=args.noise_p) if args.augment else None
    feature_extractor = make_feature_extractor()
    train_ds = EoTDataset(source, ids["train"], build_augment(augment), args.seed, args.padding)
    val_ds = EoTDataset(source, ids["val"], seed=args.seed, pad=args.padding)

    if model is None:
        model = EoTModel.from_whisper(args.whisper, head_hidden=args.head_hidden, dropout=args.dropout)
    model.config.padding = args.padding  # saved with the model, so evaluation and inference pad the same way
    if args.freeze_encoder:
        model.encoder.requires_grad_(False)
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

    val_probs = predict_probs(trainer, val_ds)
    threshold = best_f1_threshold(val_ds.labels, val_probs)
    trainer.model.config.threshold = threshold
    out = Path(args.output_dir)
    trainer.save_model(str(out / "best"))
    results = {f"val_at_threshold_{k}": v for k, v in eot_metrics(val_ds.labels, val_probs, threshold).items()}
    if args.fold:
        heldout_ds = EoTDataset(source, ids["heldout"], seed=args.seed, pad=args.padding)
        heldout = eot_metrics(heldout_ds.labels, predict_probs(trainer, heldout_ds), threshold)
        results |= {f"heldout_{k}": v for k, v in heldout.items()}
    trainer.log(results)  # also to wandb
    trainer.save_metrics("final", results)
    return out
