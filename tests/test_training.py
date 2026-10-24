import json
import os
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import WhisperFeatureExtractor

from run_eval import evaluate_test
from turn_detection.data import RUNS_DIR, SAMPLE_RATE, WINDOW_S
from turn_detection.data.dataset import load_split
from turn_detection.model import EoTModel
from turn_detection.train import trainer as trainer_module
from turn_detection.train.metrics import best_f1_threshold, eot_metrics
from turn_detection.train.trainer import EoTTrainingArguments, train

from .helpers import tiny_model


def test_metrics():
    labels = np.array([1, 1, 0, 0])
    probs = np.array([0.9, 0.6, 0.4, 0.7])
    m = eot_metrics(labels, probs, 0.5)
    assert m["accuracy"] == 0.75 and m["fpr"] == 0.5 and m["fnr"] == 0.0 and m["roc_auc"] == 0.75
    assert m["f1"] == pytest.approx(0.8) and m["pr_auc"] == pytest.approx(5 / 6) and m["threshold"] == 0.5
    assert set(m) == {"threshold", "accuracy", "f1", "fpr", "fnr", "roc_auc", "pr_auc"}
    assert "roc_auc" not in eot_metrics(np.array([1, 1]), np.array([0.9, 0.2]))  # undefined with one class
    assert best_f1_threshold(labels, np.array([0.9, 0.8, 0.2, 0.1])) == 0.8


def training_args(root, **kw) -> EoTTrainingArguments:
    # Note: a local Whisper dir with the default feature extractor, so train() needs no download
    WhisperFeatureExtractor().save_pretrained(root / "whisper")
    defaults = {
        "output_dir": str(root / "runs" / "smoke"),
        "whisper": str(root / "whisper"),
        "num_train_epochs": 2,
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 8,
        "dataloader_num_workers": 0,
        "bf16": False,
        "use_cpu": True,
        "logging_steps": 1,
        "report_to": "none",
    }
    return EoTTrainingArguments(**(defaults | kw))


def last_state(run):
    """trainer_state.json of the last checkpoint (by step: checkpoint-10 comes after checkpoint-5)."""
    return max(run.glob("checkpoint-*/trainer_state.json"), key=lambda p: int(p.parent.name.split("-")[1]))


def test_train_end_to_end(tmp_path, hub):
    model = tiny_model()
    encoder_before = {k: v.clone() for k, v in model.encoder.state_dict().items()}
    run = train(training_args(tmp_path), model=model)
    results = json.loads((run / "final_results.json").read_text())
    assert {"val_at_threshold_f1", "val_at_threshold_roc_auc", "val_at_threshold_fnr"} <= set(results)
    state = json.loads(last_state(run).read_text())
    evals = [r for r in state["log_history"] if "eval_roc_auc" in r]
    assert len(evals) == 2 and state["best_metric"] == max(r["eval_roc_auc"] for r in evals)

    # best/ is self-contained (config with threshold + weights): evaluation and prediction need no download
    model = EoTModel.from_pretrained(run / "best")
    for k, v in model.encoder.state_dict().items():  # freeze_encoder (default): the encoder did not change
        assert torch.equal(v, encoder_before[k]), k
    assert not torch.equal(model.layer_weights, torch.zeros_like(model.layer_weights))  # the head did
    feature_extractor = WhisperFeatureExtractor.from_pretrained(run / "best")  # saved by the Trainer
    assert model.config.threshold == results["val_at_threshold_threshold"]
    assert feature_extractor.n_samples == WINDOW_S * SAMPLE_RATE and feature_extractor.nb_max_frames == 800
    args = Namespace(batch_size=4, device="cpu", workers=0, bf16=False)
    test = load_split("test")
    result, preds = evaluate_test(model, feature_extractor, test, args)
    assert result["n"] == len(test) == len(preds) and preds["prob"].between(0, 1).all()
    assert preds["id"].tolist() == test["id"] and preds["label"].tolist() == test["label"]
    sources = {"human", "tts", "rime_2", "spokenwoz", "seamless"}
    assert {r["dataset"] for r in result["by_dataset"]} == sources


def test_max_steps_and_limit(tmp_path, hub, monkeypatch):
    monkeypatch.setenv("WANDB_PROJECT", "from-the-environment")  # --project wins
    args = training_args(tmp_path, max_steps=1, max_train_samples=8, augment=False)
    run = train(args, model=tiny_model())
    assert EoTModel.from_pretrained(run / "best").config.loudness_dbfs == -30.0
    assert os.environ["WANDB_PROJECT"] == "turn-detection"
    state = json.loads(last_state(run).read_text())
    assert state["global_step"] == 1 and (run / "best" / "config.json").exists()


def test_cli_defaults_and_flags(monkeypatch):
    from turn_detection.train.__main__ import parse_args

    monkeypatch.setattr("sys.argv", ["train", "--report_to", "none"])
    args = parse_args()
    assert (args.per_device_train_batch_size, args.num_train_epochs, args.lr_scheduler_type) == (64, 5, "cosine")
    assert args.metric_for_best_model == "roc_auc" and args.load_best_model_at_end and args.project == "turn-detection"
    assert args.learning_rate == 1e-3 and args.augment and args.freeze_encoder and args.max_train_samples is None
    assert args.datasets == ["smart-turn", "ttd"] and args.window_s == 8
    assert Path(args.output_dir).parent == RUNS_DIR  # artifacts/runs/<timestamp>, not TrainingArguments' default
    argv = [
        "train",
        "--report_to",
        "none",
        "--max_train_samples",
        "100",
        "--datasets",
        "ttd",
        "--window_s",
        "2",
        "--gain_db",
        "3",
        "--learning_rate",
        "1e-4",
        "--augment",
        "false",
    ]
    monkeypatch.setattr("sys.argv", argv)
    args = parse_args()
    assert (args.max_train_samples, args.gain_db, args.augment, args.learning_rate) == (100, 3.0, False, 1e-4)
    assert args.window_s == 2
    assert (
        args.datasets == ["ttd"] and '"max_train_samples": 100' in args.to_json_string()
    )  # what wandb receives as the run config


def test_persistent_workers_with_augmentation_is_rejected(tmp_path, hub):
    args = training_args(tmp_path, dataloader_persistent_workers=True)
    with pytest.raises(ValueError, match="same augmentation"):
        train(args, model=tiny_model())


def test_train_and_evaluate_a_2s_window(tmp_path, hub):
    args = training_args(tmp_path, window_s=2, max_steps=2, augment=False)
    run = train(args, model=tiny_model(window_s=2))
    model = EoTModel.from_pretrained(run / "best")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(run / "best")
    assert model.config.window_s == 2 and model.encoder.config.max_source_positions == 100
    assert feature_extractor.n_samples == 2 * SAMPLE_RATE  # saved with the model, used by run_eval
    result, _ = evaluate_test(
        model, feature_extractor, load_split("test"), Namespace(batch_size=4, device="cpu", workers=0, bf16=False)
    )
    assert result["n"] == len(load_split("test"))

    with pytest.raises(ValueError, match="built for 8 s"):
        train(training_args(tmp_path, window_s=2), model=tiny_model())


def test_best_is_logged_to_wandb(tmp_path, hub, monkeypatch):
    # the HF W&B callback runs with W&B disabled; our upload goes to a fake to check what would be sent
    monkeypatch.setenv("WANDB_MODE", "disabled")
    logged = []

    class FakeArtifact:
        def __init__(self, name, type, metadata):
            self.name, self.type, self.metadata, self.dirs = name, type, metadata, []

        def add_dir(self, path):
            self.dirs.append(Path(path))

    fake = SimpleNamespace(Artifact=FakeArtifact, run=SimpleNamespace(log_artifact=lambda a, aliases: logged.append(a)))
    monkeypatch.setattr(trainer_module, "wandb", fake)
    args = training_args(tmp_path, max_steps=2, augment=False, report_to="wandb", run_name="smoke")
    run = train(args, model=tiny_model())

    (artifact,) = logged
    config = json.loads((run / "best" / "config.json").read_text())
    assert artifact.name == "smoke-best" and artifact.type == "model" and artifact.dirs == [run / "best"]
    assert artifact.metadata["threshold"] == config["threshold"] and artifact.metadata["window_s"] == 8


def test_train_on_one_dataset(tmp_path, hub):
    run = train(training_args(tmp_path, datasets=["ttd"], max_steps=2, augment=False), model=tiny_model())
    assert "val_at_threshold_roc_auc" in json.loads((run / "final_results.json").read_text())
