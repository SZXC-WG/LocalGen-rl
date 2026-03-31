from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Sequence


@dataclass(slots=True)
class Transition:
    action_features: list[list[float]]
    legal_mask: list[bool]
    action: int
    reward: float
    next_action_features: list[list[float]]
    next_legal_mask: list[bool]
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self._storage: deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._storage)

    def push(self, transition: Transition) -> None:
        self._storage.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        if batch_size > len(self._storage):
            raise ValueError(
                f"requested batch size {batch_size}, but buffer only has {len(self._storage)} items"
            )
        return random.sample(self._storage, batch_size)

    def extend(self, transitions: Sequence[Transition]) -> None:
        self._storage.extend(transitions)
