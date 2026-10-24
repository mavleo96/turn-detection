import torch
from transformers import WhisperModel
from transformers.models.whisper.modeling_whisper import sinusoids

from turn_detection.model import EoTConfig, EoTModel

from .helpers import tiny_model, tiny_whisper_config


def test_from_whisper_keeps_weights_and_first_positions(tmp_path):
    pretrained = WhisperModel(tiny_whisper_config())  # 1500 positions, like the real checkpoints
    pretrained.save_pretrained(tmp_path)
    model = EoTModel.from_whisper(str(tmp_path))
    full = pretrained.get_encoder()
    assert model.encoder.config.max_source_positions == 400 and model.config.whisper == str(tmp_path)
    torch.testing.assert_close(model.encoder.embed_positions.weight, full.embed_positions.weight[:400])
    torch.testing.assert_close(model.encoder.conv1.weight, full.conv1.weight)


def test_fresh_encoder_positions_are_sinusoids():
    torch.testing.assert_close(tiny_model().encoder.embed_positions.weight, sinusoids(400, 32))


def test_forward_loss_and_gradients():
    model = tiny_model()
    x = torch.randn(3, 80, 800)
    out = model(input_features=x, labels=torch.tensor([1.0, 0.0, 1.0]))
    assert out.logits.shape == (3,) and out.logits.dtype == torch.float32 and out.loss.ndim == 0
    out.loss.backward()
    assert model.head[-1].weight.grad is not None and model.pool.score.weight.grad is not None
    assert model.layer_weights.grad is not None
    assert model.encoder.conv1.weight.grad is not None  # the model itself freezes nothing
    assert model(input_features=x).loss is None


def test_layer_weights_cover_the_encoder_layers():
    model = tiny_model()
    assert model.layer_weights.shape == (2,)  # tiny encoder: 2 layers (the embedding output is excluded)
    torch.testing.assert_close(torch.softmax(model.layer_weights, 0), torch.tensor([0.5, 0.5]))


def test_save_pretrained_roundtrip(tmp_path):
    model = tiny_model().eval()
    model.config.threshold = 0.42
    model.save_pretrained(tmp_path)
    loaded = EoTModel.from_pretrained(tmp_path).eval()
    x = torch.randn(2, 80, 800)
    with torch.no_grad():
        torch.testing.assert_close(loaded(input_features=x).logits, model(input_features=x).logits)
    assert loaded.config.threshold == 0.42 and loaded.config.n_positions == 400
    assert loaded.config.encoder_config.encoder_layers == 2  # the saved encoder, not one fetched for `whisper`


def test_config_positions():
    assert [EoTConfig(window_s=w).n_positions for w in (2, 4, 8)] == [100, 200, 400]
    assert EoTConfig().window_s == 8
