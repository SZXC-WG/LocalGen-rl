from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class TrainingConfig:
    seed: int = 1337
    episodes: int = 320
    max_half_turns: int = 240
    board_min_size: int = 12
    board_max_size: int = 18
    bc_epochs: int = 10
    bc_batch_size: int = 192
    bc_learning_rate: float = 3e-4
    bc_validation_fraction: float = 0.1
    replay_capacity: int = 50_000
    warmup_steps: int = 512
    batch_size: int = 128
    gamma: float = 0.985
    learning_rate: float = 2.5e-4
    weight_decay: float = 1e-6
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 12_000
    train_interval: int = 4
    target_sync_interval: int = 250
    eval_interval: int = 50
    eval_episodes: int = 12
    log_interval: int = 10
    hidden1_size: int = 64
    hidden2_size: int = 32
    device: str = "auto"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def resolve(self, repo_root: Path) -> dict[str, Any]:
        rl_root = repo_root / "rl"
        dataset_path = rl_root / "datasets" / "xrz_imitation.jsonl"
        return {
            "repo_root": repo_root,
            "rl_root": rl_root,
            "dataset_path": dataset_path,
            "dataset_paths": [dataset_path],
            "runs_dir": rl_root / "runs",
            "checkpoints_dir": rl_root / "checkpoints",
            "checkpoint_path": rl_root / "checkpoints" / "xrz_dqn.pt",
            "best_checkpoint_path": rl_root / "checkpoints" / "xrz_dqn_best.pt",
            "export_header_path": repo_root / "src" / "bots" / "generated" / "xrzRlWeights.h",
        }
