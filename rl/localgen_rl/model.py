from __future__ import annotations

import torch
from torch import nn

from .constants import INPUT_FEATURE_COUNT


class ActionValueNet(nn.Module):
    """Shared MLP that scores each candidate move independently."""

    def __init__(
        self,
        input_dim: int = INPUT_FEATURE_COUNT,
        hidden1_size: int = 64,
        hidden2_size: int = 32,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden1_size),
            nn.ReLU(),
            nn.Linear(hidden1_size, hidden2_size),
            nn.ReLU(),
        )
        self.q_head = nn.Linear(hidden2_size, 1)

    def forward(self, action_features: torch.Tensor) -> torch.Tensor:
        if action_features.ndim != 3:
            raise ValueError(
                f"expected [batch, actions, features], got {action_features.shape}"
            )
        batch_size, action_count, feature_count = action_features.shape
        if feature_count != self.input_dim:
            raise ValueError(
                f"expected {self.input_dim} features, got {feature_count}"
            )
        hidden = self.backbone(action_features.reshape(batch_size * action_count, feature_count))
        q_values = self.q_head(hidden).reshape(batch_size, action_count)
        return q_values

    def architecture(self) -> tuple[int, int, int]:
        return self.input_dim, self.hidden1_size, self.hidden2_size


def mask_illegal_q_values(
    q_values: torch.Tensor, legal_mask: torch.Tensor, fill_value: float = -1e9
) -> torch.Tensor:
    if q_values.shape != legal_mask.shape:
        raise ValueError(
            f"q_values shape {q_values.shape} != legal_mask shape {legal_mask.shape}"
        )
    fill = torch.full_like(q_values, fill_value)
    return torch.where(legal_mask, q_values, fill)
