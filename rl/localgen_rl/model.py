from __future__ import annotations

import torch
from torch import nn

from .constants import INPUT_FEATURE_COUNT


class ActionValueNet(nn.Module):
    """Shared MLP that scores each candidate move independently."""

    def __init__(
        self,
        input_dim: int = INPUT_FEATURE_COUNT,
        hidden1_size: int = 96,
        hidden2_size: int = 48,
        hidden3_size: int = 0,
    ) -> None:
        super().__init__()
        if hidden1_size <= 0 or hidden2_size <= 0:
            raise ValueError("hidden1_size and hidden2_size must be positive")
        if hidden3_size < 0:
            raise ValueError("hidden3_size must be non-negative")

        self.input_dim = input_dim
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.hidden3_size = hidden3_size

        layers: list[nn.Module] = [
            nn.Linear(input_dim, hidden1_size),
            nn.ReLU(),
            nn.Linear(hidden1_size, hidden2_size),
            nn.ReLU(),
        ]
        q_head_input_dim = hidden2_size
        if hidden3_size > 0:
            layers.extend(
                [
                    nn.Linear(hidden2_size, hidden3_size),
                    nn.ReLU(),
                ]
            )
            q_head_input_dim = hidden3_size

        self.backbone = nn.Sequential(*layers)
        self.q_head = nn.Linear(q_head_input_dim, 1)

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

    def architecture(self) -> tuple[int, int, int, int]:
        return self.input_dim, self.hidden1_size, self.hidden2_size, self.hidden3_size


def mask_illegal_q_values(
    q_values: torch.Tensor, legal_mask: torch.Tensor, fill_value: float = -1e9
) -> torch.Tensor:
    if q_values.shape != legal_mask.shape:
        raise ValueError(
            f"q_values shape {q_values.shape} != legal_mask shape {legal_mask.shape}"
        )
    fill = torch.full_like(q_values, fill_value)
    return torch.where(legal_mask, q_values, fill)
