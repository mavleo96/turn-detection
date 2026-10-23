"""Building blocks of the EoT head."""

import torch
from torch import nn


class AttentionPool(nn.Module):
    """Softmax-weighted average over time with one learned score per frame."""

    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(h).squeeze(-1).float(), dim=-1).to(h.dtype)  # (B, T)
        return torch.einsum("bt,btd->bd", weights, h)
