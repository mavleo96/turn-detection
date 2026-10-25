"""Export a checkpoint for serve/: <checkpoint>/model.onnx, fp32, checked against torch.

The graph is the whole model plus the sigmoid: input_features (B, 80, 100 * window_s) -> probability (B,), with
a dynamic batch dimension. serve/ runs it with ONNX Runtime and reads everything else (threshold, window,
preprocessing, feature extractor) from the same directory, so the image needs no torch.

Run (from the repo root, in the training env):  python scripts/export_onnx.py artifacts/models/both-8s
"""

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

from turn_detection.model import EoTModel

TOLERANCE = 1e-4  # max |P(ONNX Runtime) - P(torch)|; both-8s: ~4e-6


class Probability(nn.Module):
    def __init__(self, model: EoTModel):
        super().__init__()
        self.model = model

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(input_features=input_features).logits)  # (B,)


def export(model_dir: Path) -> Path:
    """Write model_dir/model.onnx and check it against torch on random features; returns its path."""
    model = EoTModel.from_pretrained(model_dir).eval()
    frames = model.config.window_s * 100
    probability = Probability(model).eval()

    path = model_dir / "model.onnx"
    example = torch.randn(2, 80, frames)  # (B, 80, frames)
    torch.onnx.export(
        probability,
        (example,),
        path,
        input_names=["input_features"],
        output_names=["probability"],
        dynamic_shapes=({0: torch.export.Dim("batch")},),
        external_data=False,  # one file: the weights (~31 MB for Whisper Tiny) inside model.onnx
    )

    # Check batch sizes other than the example's: the batch dimension must stay dynamic
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    for batch in (1, 3):
        x = torch.randn(batch, 80, frames)  # (B, 80, frames)
        with torch.inference_mode():
            expected = probability(x).numpy()  # (B,)
        actual = session.run(None, {"input_features": x.numpy()})[0]  # (B,)
        diff = np.abs(actual - expected).max()
        if diff > TOLERANCE:
            raise RuntimeError(f"ONNX Runtime and torch differ by {diff:.2e} at batch {batch} (> {TOLERANCE})")

    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="Export a checkpoint to ONNX for serve/")
    ap.add_argument("model_dir", type=Path, help="a save_pretrained directory, e.g. artifacts/runs/<run>/best")
    args = ap.parse_args()
    print(f"wrote {export(args.model_dir)}")


if __name__ == "__main__":
    main()
