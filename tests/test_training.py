import json
import os
from argparse import Namespace

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch
from transformers import EvalPrediction, WhisperFeatureExtractor

from predict import EoTPredictor
from run_eval import evaluate_test, load_test_clips
from turn_detection.data.constants import SAMPLE_RATE, WINDOW_S
from turn_detection.model import EoTModel
from turn_detection.train.metrics import best_f1_threshold, compute_metrics, eot_metrics
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
    logits = np.log(probs / (1 - probs))
    assert compute_metrics(EvalPrediction(predictions=logits, label_ids=labels)) == pytest.approx(m)
    assert best_f1_threshold(labels, np.array([0.9, 0.8, 0.2, 0.1])) == 0.8


def training_args(root, **kw) -> EoTTrainingArguments:
    return EoTTrainingArguments(
        output_dir=str(root / "runs" / "smoke"),
        manifests_dir=str(root / "manifests"),
        num_train_epochs=2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        dataloader_num_workers=0,
        bf16=False,
        use_cpu=True,
        logging_steps=1,
        report_to="none",
        **kw,
    )


def last_state(run):
    """trainer_state.json of the last checkpoint (by step: checkpoint-10 comes after checkpoint-5)."""
    return max(run.glob("checkpoint-*/trainer_state.json"), key=lambda p: int(p.parent.name.split("-")[1]))


def test_train_smart_turn_end_to_end(artifacts, hf_splits):
    model = tiny_model()
    encoder_before = {k: v.clone() for k, v in model.encoder.state_dict().items()}
    run = train(training_args(artifacts, dataset="smart-turn"), model=model)
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
    manifest_path = artifacts / "manifests" / "test.parquet"
    manifest = pd.read_parquet(manifest_path)
    clips = load_test_clips("smart-turn", hf_splits["test"], manifest_path)
    result, preds = evaluate_test(model, feature_extractor, hf_splits["test"], clips, args, name="smart-turn")
    assert result["n"] == len(manifest) == len(preds) and preds["prob"].between(0, 1).all()
    assert preds["label"].tolist() == manifest["endpoint_bool"].astype(int).tolist()
    assert {r["source"] for r in result["by_source"]} == {"human", "tts"} and "by_synthetic" in result

    wav = artifacts / "clip.wav"
    sf.write(wav, 0.1 * np.random.default_rng(0).standard_normal(22050 * 3), 22050)
    audio, sr = sf.read(wav, dtype="float32")
    assert 0 <= EoTPredictor(run / "best").predict(audio, sr) <= 1


def test_max_steps_and_limit(artifacts, hf_splits, monkeypatch):
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    args = training_args(artifacts, dataset="smart-turn", max_steps=1, max_train_samples=8, augment=False)
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
    assert args.learning_rate == 1e-3 and args.augment and args.freeze_encoder
    assert args.dataset == "ttd"
    argv = [
        "train",
        "--report_to",
        "none",
        "--dataset",
        "smart-turn",
        "--gain_db",
        "3",
        "--learning_rate",
        "1e-4",
        "--augment",
        "false",
    ]
    monkeypatch.setattr("sys.argv", argv)
    args = parse_args()
    assert (args.dataset, args.gain_db, args.augment, args.learning_rate) == ("smart-turn", 3.0, False, 1e-4)
    assert '"dataset": "smart-turn"' in args.to_json_string()  # what wandb receives as the run config


def test_persistent_workers_with_augmentation_is_rejected(artifacts, hf_splits):
    args = training_args(artifacts, dataloader_persistent_workers=True)
    with pytest.raises(ValueError, match="same augmentation"):
        train(args, model=tiny_model())


def test_train_and_evaluate_on_ttd(tmp_path, ttd_splits):
    run = train(training_args(tmp_path, dataset="ttd"), model=tiny_model())  # its own validation split
    results = json.loads((run / "final_results.json").read_text())
    assert "val_at_threshold_roc_auc" in results and not any(k.startswith("heldout") for k in results)

    model = EoTModel.from_pretrained(run / "best")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(run / "best")
    args = Namespace(batch_size=4, device="cpu", workers=0, bf16=False)
    clips = load_test_clips("ttd", ttd_splits["test"], tmp_path / "unused.parquet")
    result, preds = evaluate_test(model, feature_extractor, ttd_splits["test"], clips, args, name="ttd")
    assert result["n"] == 20 and preds["label"].tolist() == ttd_splits["test"]["label"]
    assert {r["source"] for r in result["by_source"]} == {"spokenwoz", "seamless"} and "by_synthetic" not in result


def test_ttd_sets(tmp_path, ttd_splits):
    from turn_detection.train.trainer import load_sets

    sets = load_sets("ttd", training_args(tmp_path, max_train_samples=10))
    train_ids, val_ids = sets["train"][1], sets["val"][1]
    assert len(train_ids) == 10 and len(val_ids) == 16 and set(sets) == {"train", "val"}
    sources = {i.split("_")[0] for i in train_ids}
    assert sources == {"spokenwoz", "seamless"}  # a random subset, not the first rows (all spokenwoz)
    with pytest.raises(ValueError, match="dataset must be"):
        load_sets("nope", training_args(tmp_path))


def test_train_on_both_datasets(artifacts, hf_splits, ttd_splits):
    args = training_args(artifacts, dataset="ttd,smart-turn", max_steps=2, augment=False)
    run = train(args, model=tiny_model())
    results = json.loads((run / "final_results.json").read_text())
    assert "val_at_threshold_roc_auc" in results


def test_concat_passes_labels_and_epoch_through(hf_splits, ttd_splits):
    from turn_detection.data.dataset import EoTConcat, EoTDataset

    st = EoTDataset(hf_splits["train"], ["train-000", "train-001"], label_column="endpoint_bool")
    ttd = EoTDataset(ttd_splits["train"], ["spokenwoz_train_001"])
    both = EoTConcat([st, ttd])
    assert len(both) == 3 and both.labels.tolist() == [1.0, 0.0, 1.0]
    both.set_epoch(4)
    assert st.epoch == ttd.epoch == 4
