import torch
from torch import nn


class AttentionPool(nn.Module):
    """Softmax-weighted average over time with one learned score per frame."""

    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        scores = self.score(h).squeeze(-1)  # (B, T)
        weights = torch.softmax(scores, dim=-1)  # (B, T)
        return (weights.unsqueeze(-1) * h).sum(dim=1)  # (B, D)
