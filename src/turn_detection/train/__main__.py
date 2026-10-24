"""Train the EoT classifier with the Hugging Face Trainer; logs to Weights & Biases (`wandb login` once).

Run (from the repo root):
  python -m turn_detection.train                        # TTD -> artifacts/runs/<timestamp>/
  python -m turn_detection.train --dataset smart-turn   # needs scripts/split.py's manifests first
  python -m turn_detection.train --max_steps 20 --max_train_samples 512 --report_to none   # quick smoke run

Flags: the fields of EoTTrainingArguments in train/trainer.py: this project's model / data /
augmentation settings plus every transformers TrainingArguments flag (--learning_rate, --report_to, ...).
"""

from transformers import HfArgumentParser

from .trainer import EoTTrainingArguments, train


def parse_args() -> EoTTrainingArguments:
    (args,) = HfArgumentParser(EoTTrainingArguments).parse_args_into_dataclasses()
    return args


if __name__ == "__main__":
    train(parse_args())
