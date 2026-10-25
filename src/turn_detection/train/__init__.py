"""Training with the Hugging Face Trainer, and the metrics. Logs to Weights & Biases (`wandb login` once).

    python -m turn_detection.train                        # smart-turn + TTD -> artifacts/runs/<timestamp>/
    python -m turn_detection.train --datasets ttd         # TTD only
    python -m turn_detection.train --window_s 2           # the last 2 s instead of 8 (2, 4 or 8)
    python -m turn_detection.train --max_steps 20 --max_train_samples 512 --report_to none   # quick smoke run

Flags: the fields of EoTTrainingArguments (this project's model / data / augmentation settings) plus every
transformers TrainingArguments flag (--learning_rate, --report_to, ...).

- trainer.py: the train and validation splits of `--datasets` (data.dataset.load_split). On top of `Trainer`: the
  frozen Whisper encoder (`freeze_encoder`; AdamW then only gets the layer weights, pooling and MLP), a callback
  that tells the dataset the epoch (augmentation differs per epoch), and the decision threshold chosen on validation
  (best F1), saved in the model's config. A run directory holds the checkpoints, `best/` (best epoch by validation
  ROC AUC, with the threshold and the feature extractor) and `final_results.json`.
- metrics.py: with scikit-learn, positive class = turn complete. `fpr`: incomplete turns predicted complete (the
  agent interrupts); `fnr`: complete turns predicted incomplete (the agent waits too long); `pr_auc`: average
  precision, next to `roc_auc`.
"""
