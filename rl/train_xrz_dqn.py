#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from localgen_rl import (
    FEATURE_NAMES,
    ActionValueNet,
    LocalGenMiniEnv,
    ReplayBuffer,
    TrainingConfig,
    Transition,
    export_cpp_header,
    warm_start_model,
)
from localgen_rl.model import mask_illegal_q_values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the XrzBot policy with behavior cloning and RL fine-tuning."
    )
    parser.add_argument("--episodes", type=int, default=None, help="Override RL fine-tuning episode count")
    parser.add_argument("--bc-epochs", type=int, default=None, help="Override behavior-cloning epoch count")
    parser.add_argument("--bc-batch-size", type=int, default=None, help="Override behavior-cloning batch size")
    parser.add_argument(
        "--bc-learning-rate",
        type=float,
        default=None,
        help="Override behavior-cloning learning rate",
    )
    parser.add_argument("--hidden1-size", type=int, default=None, help="Override the first hidden layer width")
    parser.add_argument("--hidden2-size", type=int, default=None, help="Override the second hidden layer width")
    parser.add_argument("--hidden3-size", type=int, default=None, help="Override the optional third hidden layer width")
    parser.add_argument(
        "--dataset",
        type=Path,
        action="append",
        default=None,
        help=(
            "Imitation dataset JSONL path. Repeat the flag to mix multiple files "
            "or intentionally up-weight a corpus by passing the same path again."
        ),
    )
    parser.add_argument("--skip-bc", action="store_true", help="Skip behavior cloning even if a dataset exists")
    parser.add_argument("--skip-rl", action="store_true", help="Skip RL fine-tuning after behavior cloning")
    parser.add_argument("--device", type=str, default="auto", help="auto, mps, cuda, or cpu")
    parser.add_argument("--seed", type=int, default=None, help="Override the random seed")
    parser.add_argument(
        "--export-header",
        type=Path,
        default=None,
        help="Override the exported C++ header path",
    )
    parser.add_argument(
        "--export-namespace",
        type=str,
        default="xrz_rl_model",
        help="Override the C++ namespace used in the exported header",
    )
    parser.add_argument("--eval-only", action="store_true", help="Load a checkpoint and only run evaluation")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint to resume from or evaluate",
    )
    parser.add_argument(
        "--bc-init",
        type=Path,
        default=None,
        help="Warm-start behavior cloning from a checkpoint or exported header",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested, but torch.backends.mps.is_available() is false")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    return torch.device(requested)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def linear_epsilon(config: TrainingConfig, step: int) -> float:
    if step >= config.epsilon_decay_steps:
        return config.epsilon_end
    progress = step / max(1, config.epsilon_decay_steps)
    return config.epsilon_start + (config.epsilon_end - config.epsilon_start) * progress


def observation_to_tensors(
    observation, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    action_features = torch.tensor(
        observation.action_features, dtype=torch.float32, device=device
    ).unsqueeze(0)
    legal_mask = torch.tensor(observation.legal_mask, dtype=torch.bool, device=device).unsqueeze(0)
    return action_features, legal_mask


def choose_action(
    model: ActionValueNet,
    observation,
    epsilon: float,
    *,
    device: torch.device,
    rng: random.Random,
) -> int:
    legal_actions = [index for index, is_legal in enumerate(observation.legal_mask) if is_legal]
    if not legal_actions:
        return 0
    if rng.random() < epsilon:
        return rng.choice(legal_actions)

    with torch.no_grad():
        action_features, legal_mask = observation_to_tensors(observation, device)
        q_values = mask_illegal_q_values(model(action_features), legal_mask)
        return int(q_values.argmax(dim=1).item())


def optimize_model(
    online_net: ActionValueNet,
    target_net: ActionValueNet,
    optimizer: optim.Optimizer,
    replay_buffer: ReplayBuffer,
    config: TrainingConfig,
    *,
    device: torch.device,
) -> Optional[dict[str, float]]:
    if len(replay_buffer) < max(config.batch_size, config.warmup_steps):
        return None

    batch = replay_buffer.sample(config.batch_size)
    action_features = torch.tensor(
        [transition.action_features for transition in batch],
        dtype=torch.float32,
        device=device,
    )
    legal_mask = torch.tensor(
        [transition.legal_mask for transition in batch],
        dtype=torch.bool,
        device=device,
    )
    actions = torch.tensor(
        [transition.action for transition in batch], dtype=torch.long, device=device
    ).unsqueeze(1)
    rewards = torch.tensor(
        [transition.reward for transition in batch], dtype=torch.float32, device=device
    )
    next_action_features = torch.tensor(
        [transition.next_action_features for transition in batch],
        dtype=torch.float32,
        device=device,
    )
    next_legal_mask = torch.tensor(
        [transition.next_legal_mask for transition in batch],
        dtype=torch.bool,
        device=device,
    )
    dones = torch.tensor(
        [transition.done for transition in batch], dtype=torch.bool, device=device
    )

    q_values = online_net(action_features)
    current_q = q_values.gather(1, actions).squeeze(1)

    with torch.no_grad():
        next_online_q = mask_illegal_q_values(online_net(next_action_features), next_legal_mask)
        next_actions = next_online_q.argmax(dim=1, keepdim=True)
        next_target_q_all = target_net(next_action_features)
        next_target_q = next_target_q_all.gather(1, next_actions).squeeze(1)
        no_legal_actions = ~next_legal_mask.any(dim=1)
        next_target_q = torch.where(
            no_legal_actions,
            torch.zeros_like(next_target_q),
            next_target_q,
        )
        targets = rewards + config.gamma * next_target_q * (~dones).float()

    loss = F.smooth_l1_loss(current_q, targets)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(online_net.parameters(), max_norm=5.0)
    optimizer.step()

    return {
        "loss": float(loss.item()),
        "q_mean": float(current_q.mean().item()),
        "target_mean": float(targets.mean().item()),
    }


def evaluate_policy(
    model: ActionValueNet,
    config: TrainingConfig,
    *,
    device: torch.device,
    episodes: int,
    seed_offset: int = 100_000,
) -> dict[str, float]:
    model.eval()
    total_return = 0.0
    wins = 0
    losses = 0
    draws = 0
    total_steps = 0
    eval_rng = random.Random(config.seed + seed_offset)

    for episode in range(episodes):
        env = LocalGenMiniEnv(
            seed=config.seed + seed_offset + episode,
            board_min_size=config.board_min_size,
            board_max_size=config.board_max_size,
            max_half_turns=config.max_half_turns,
        )
        observation = env.reset()
        done = False
        episode_return = 0.0
        info: dict[str, object] = {}
        while not done:
            action = choose_action(model, observation, 0.0, device=device, rng=eval_rng)
            observation, reward, done, info = env.step(action)
            episode_return += reward
            total_steps += 1
        total_return += episode_return
        winner = info.get("winner")
        if winner == 0:
            wins += 1
        elif winner == 1:
            losses += 1
        else:
            draws += 1

    model.train()
    return {
        "average_return": total_return / max(1, episodes),
        "win_rate": wins / max(1, episodes),
        "loss_rate": losses / max(1, episodes),
        "draw_rate": draws / max(1, episodes),
        "average_steps": total_steps / max(1, episodes),
    }


def save_checkpoint(
    checkpoint_path: Path,
    model: ActionValueNet,
    optimizer: optim.Optimizer,
    *,
    config: TrainingConfig,
    episode: int,
    total_steps: int,
    best_win_rate: float,
    device_name: str,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config.as_dict(),
            "episode": episode,
            "total_steps": total_steps,
            "best_win_rate": best_win_rate,
            "device": device_name,
            "feature_names": FEATURE_NAMES,
        },
        checkpoint_path,
    )


def load_checkpoint(
    checkpoint_path: Path,
    model: ActionValueNet,
    optimizer: Optional[optim.Optimizer] = None,
) -> dict[str, object]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return payload


def maybe_load_checkpoint(
    checkpoint_path: Path,
    model: ActionValueNet,
    optimizer: Optional[optim.Optimizer],
    *,
    strict: bool,
) -> Optional[dict[str, object]]:
    try:
        return load_checkpoint(checkpoint_path, model, optimizer)
    except Exception as exc:  # pragma: no cover - defensive compatibility path
        if strict:
            raise
        print(f"Skipping checkpoint {checkpoint_path}: {exc}")
        return None


def read_int(payload: dict[str, object], key: str, default: int) -> int:
    value = payload.get(key, default)
    return int(value) if isinstance(value, (int, float, str)) else default


def read_float(payload: dict[str, object], key: str, default: float) -> float:
    value = payload.get(key, default)
    return float(value) if isinstance(value, (int, float, str)) else default


def iter_imitation_samples(
    dataset_path: Path,
    *,
    expected_feature_count: int,
):
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            sample_action = int(payload["action"])
            sample_legal_mask = [bool(value) for value in payload["legal_mask"]]
            sample_features = [
                [float(value) for value in feature_row]
                for feature_row in payload["action_features"]
            ]

            if len(sample_features) != len(sample_legal_mask):
                raise ValueError(
                    f"dataset {dataset_path} line {line_number}: feature rows != legal mask length"
                )
            if not sample_features:
                raise ValueError(f"dataset {dataset_path} line {line_number}: empty action set")
            if sample_action < 0 or sample_action >= len(sample_features):
                raise ValueError(
                    f"dataset {dataset_path} line {line_number}: action index out of range"
                )
            if not sample_legal_mask[sample_action]:
                raise ValueError(
                    f"dataset {dataset_path} line {line_number}: label points to an illegal action"
                )
            sample_feature_count = len(sample_features[0])
            if any(len(feature_row) != sample_feature_count for feature_row in sample_features):
                raise ValueError(
                    f"dataset {dataset_path} line {line_number}: inconsistent feature row widths"
                )
            if sample_feature_count > expected_feature_count:
                raise ValueError(
                    f"dataset {dataset_path} line {line_number}: expected at most {expected_feature_count} features"
                )

            yield {
                "action": sample_action,
                "action_count": len(sample_features),
                "feature_count": sample_feature_count,
                "features": sample_features,
                "legal_mask": sample_legal_mask,
                "legal_action_count": sum(sample_legal_mask),
            }


def load_imitation_dataset(
    dataset_paths: Sequence[Path],
    *,
    expected_feature_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    if not dataset_paths:
        raise FileNotFoundError("no imitation datasets were provided")

    unique_dataset_paths: list[Path] = []
    dataset_repeat_counts: dict[Path, int] = {}
    for dataset_path in dataset_paths:
        if dataset_path not in dataset_repeat_counts:
            unique_dataset_paths.append(dataset_path)
            dataset_repeat_counts[dataset_path] = 0
        dataset_repeat_counts[dataset_path] += 1

    total_samples = 0
    max_action_count = 0
    min_action_count: Optional[int] = None
    total_action_count = 0
    min_feature_count: Optional[int] = None
    max_feature_count = 0
    total_feature_count = 0
    total_legal_action_count = 0
    dataset_breakdown: list[dict[str, Any]] = []

    for dataset_path in unique_dataset_paths:
        if not dataset_path.exists():
            raise FileNotFoundError(f"imitation dataset not found: {dataset_path}")

        repeat_count = dataset_repeat_counts[dataset_path]

        file_samples = 0
        file_action_count_total = 0
        file_legal_action_count_total = 0
        file_feature_count_total = 0
        file_min_feature_count: Optional[int] = None
        file_max_feature_count = 0

        for sample in iter_imitation_samples(
            dataset_path,
            expected_feature_count=expected_feature_count,
        ):
            sample_action_count = int(sample["action_count"])
            sample_feature_count = int(sample["feature_count"])
            sample_legal_action_count = int(sample["legal_action_count"])

            max_action_count = max(max_action_count, sample_action_count)
            min_action_count = (
                sample_action_count
                if min_action_count is None
                else min(min_action_count, sample_action_count)
            )
            total_action_count += sample_action_count
            min_feature_count = (
                sample_feature_count
                if min_feature_count is None
                else min(min_feature_count, sample_feature_count)
            )
            max_feature_count = max(max_feature_count, sample_feature_count)
            total_feature_count += sample_feature_count
            total_legal_action_count += sample_legal_action_count

            file_action_count_total += sample_action_count
            file_legal_action_count_total += sample_legal_action_count
            file_feature_count_total += sample_feature_count
            file_min_feature_count = (
                sample_feature_count
                if file_min_feature_count is None
                else min(file_min_feature_count, sample_feature_count)
            )
            file_max_feature_count = max(file_max_feature_count, sample_feature_count)

            total_samples += 1
            file_samples += 1

        if file_samples == 0:
            raise RuntimeError(f"imitation dataset is empty: {dataset_path}")

        dataset_breakdown.append(
            {
                "path": str(dataset_path),
                "repeat_count": repeat_count,
                "sample_count": file_samples,
                "effective_sample_count": file_samples * repeat_count,
                "average_action_count": file_action_count_total / max(1, file_samples),
                "average_legal_actions": file_legal_action_count_total / max(1, file_samples),
                "average_feature_count": file_feature_count_total / max(1, file_samples),
                "min_feature_count": file_min_feature_count,
                "max_feature_count": file_max_feature_count,
            }
        )

    if total_samples == 0:
        raise RuntimeError("all provided imitation datasets were empty")

    action_features_tensor = torch.zeros(
        (total_samples, max_action_count, expected_feature_count),
        dtype=torch.float32,
    )
    legal_masks_tensor = torch.zeros((total_samples, max_action_count), dtype=torch.bool)
    actions_tensor = torch.empty(total_samples, dtype=torch.long)
    sample_weights_tensor = torch.empty(total_samples, dtype=torch.int32)

    sample_index = 0
    for dataset_path in unique_dataset_paths:
        repeat_count = dataset_repeat_counts[dataset_path]
        for sample in iter_imitation_samples(
            dataset_path,
            expected_feature_count=expected_feature_count,
        ):
            sample_action_count = int(sample["action_count"])
            sample_feature_count = int(sample["feature_count"])
            action_features_tensor[
                sample_index,
                :sample_action_count,
                :sample_feature_count,
            ] = torch.tensor(sample["features"], dtype=torch.float32)
            legal_masks_tensor[sample_index, :sample_action_count] = torch.tensor(
                sample["legal_mask"],
                dtype=torch.bool,
            )
            actions_tensor[sample_index] = int(sample["action"])
            sample_weights_tensor[sample_index] = repeat_count
            sample_index += 1

    if sample_index != total_samples:
        raise RuntimeError(
            f"dataset reload count mismatch: loaded {sample_index} samples, expected {total_samples}"
        )

    legal_actions_per_state = legal_masks_tensor.sum(dim=1).float()
    stats = {
        "dataset_file_count": len(dataset_breakdown),
        "dataset_breakdown": dataset_breakdown,
        "sample_count": float(actions_tensor.shape[0]),
        "effective_sample_count": float(sample_weights_tensor.sum().item()),
        "average_sample_weight": float(sample_weights_tensor.float().mean().item()),
        "max_action_count": float(action_features_tensor.shape[1]),
        "min_action_count": float(min_action_count or 0),
        "average_action_count": float(total_action_count / max(1, total_samples)),
        "feature_count": float(action_features_tensor.shape[2]),
        "min_feature_count": float(min_feature_count or 0),
        "max_feature_count": float(max_feature_count),
        "average_feature_count": float(total_feature_count / max(1, total_samples)),
        "average_legal_actions": float(legal_actions_per_state.mean().item()),
        "min_legal_actions": float(legal_actions_per_state.min().item()),
        "max_legal_actions": float(legal_actions_per_state.max().item()),
    }
    return action_features_tensor, legal_masks_tensor, actions_tensor, sample_weights_tensor, stats


def make_supervised_loaders(
    action_features: torch.Tensor,
    legal_masks: torch.Tensor,
    actions: torch.Tensor,
    *,
    batch_size: int,
    validation_fraction: float,
    seed: int,
    sample_weights: Optional[torch.Tensor] = None,
) -> tuple[DataLoader, Optional[DataLoader], dict[str, float]]:
    dataset = TensorDataset(action_features, legal_masks, actions)
    total_size = len(dataset)
    generator = torch.Generator().manual_seed(seed)

    if sample_weights is None:
        sample_weights = torch.ones(total_size, dtype=torch.int32)
    else:
        sample_weights = sample_weights.to(dtype=torch.int32)

    def expand_indices(indices: Sequence[int]) -> list[int]:
        expanded: list[int] = []
        for index in indices:
            repeat_count = max(1, int(sample_weights[index].item()))
            expanded.extend([index] * repeat_count)
        return expanded

    if total_size == 1:
        train_indices = [0]
        expanded_train_indices = expand_indices(train_indices)
        train_loader = DataLoader(
            dataset,
            batch_size=1,
            sampler=SubsetRandomSampler(expanded_train_indices, generator=generator),
        )
        split_stats = {
            "train_unique_samples": 1.0,
            "train_effective_samples": float(len(expanded_train_indices)),
            "validation_samples": 0.0,
        }
        return train_loader, None, split_stats

    validation_size = int(total_size * validation_fraction)
    if validation_fraction > 0.0:
        validation_size = max(1, validation_size)
    validation_size = min(validation_size, total_size - 1)
    train_size = total_size - validation_size

    shuffled_indices = torch.randperm(total_size, generator=generator).tolist()
    validation_indices = shuffled_indices[:validation_size]
    train_indices = shuffled_indices[validation_size: validation_size + train_size]
    expanded_train_indices = expand_indices(train_indices)

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(expanded_train_indices, generator=generator),
    )
    validation_loader = DataLoader(
        Subset(dataset, validation_indices),
        batch_size=batch_size,
        shuffle=False,
    )
    split_stats = {
        "train_unique_samples": float(len(train_indices)),
        "train_effective_samples": float(len(expanded_train_indices)),
        "validation_samples": float(len(validation_indices)),
    }
    return train_loader, validation_loader, split_stats


@torch.no_grad()
def evaluate_imitation_loader(
    model: ActionValueNet,
    data_loader: Optional[DataLoader],
    *,
    device: torch.device,
) -> dict[str, float]:
    if data_loader is None:
        return {"loss": 0.0, "accuracy": 0.0, "samples": 0.0}

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_action_features, batch_legal_masks, batch_actions in data_loader:
        batch_action_features = batch_action_features.to(device)
        batch_legal_masks = batch_legal_masks.to(device)
        batch_actions = batch_actions.to(device)

        logits = mask_illegal_q_values(model(batch_action_features), batch_legal_masks)
        loss = F.cross_entropy(logits, batch_actions)
        predictions = logits.argmax(dim=1)

        batch_size = int(batch_actions.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_correct += int((predictions == batch_actions).sum().item())
        total_samples += batch_size

    model.train()
    return {
        "loss": total_loss / max(1, total_samples),
        "accuracy": total_correct / max(1, total_samples),
        "samples": float(total_samples),
    }


def run_behavior_cloning(
    model: ActionValueNet,
    config: TrainingConfig,
    *,
    dataset_paths: Sequence[Path],
    device: torch.device,
    writer: SummaryWriter,
) -> dict[str, Any]:
    action_features, legal_masks, actions, sample_weights, dataset_stats = load_imitation_dataset(
        dataset_paths,
        expected_feature_count=model.input_dim,
    )
    train_loader, validation_loader, split_stats = make_supervised_loaders(
        action_features,
        legal_masks,
        actions,
        batch_size=config.bc_batch_size,
        validation_fraction=config.bc_validation_fraction,
        seed=config.seed,
        sample_weights=sample_weights,
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.bc_learning_rate,
        weight_decay=config.weight_decay,
    )
    best_state: Optional[dict[str, torch.Tensor]] = None
    best_accuracy = -1.0
    global_step = 0

    writer.add_text(
        "imitation/dataset_paths",
        "\n".join(str(dataset_path) for dataset_path in dataset_paths),
    )
    writer.add_text("imitation/dataset_stats", json.dumps(dataset_stats, indent=2))
    writer.add_text("imitation/split_stats", json.dumps(split_stats, indent=2))
    dataset_breakdown = list(dataset_stats["dataset_breakdown"])
    print(
        f"Loaded {int(dataset_stats['dataset_file_count'])} imitation dataset(s) with "
        f"{int(dataset_stats['sample_count'])} unique samples total "
        f"({int(dataset_stats['effective_sample_count'])} effective weighted samples), "
        f"{int(dataset_stats['min_action_count'])}-{int(dataset_stats['max_action_count'])} actions/state, "
        f"{int(dataset_stats['feature_count'])} features/action."
    )
    for dataset_info in dataset_breakdown:
        print(
            f"  - {dataset_info['path']}: {int(dataset_info['sample_count'])} samples, "
            f"repeat={int(dataset_info['repeat_count'])}, "
            f"effective={int(dataset_info['effective_sample_count'])}, "
            f"avg_actions={dataset_info['average_action_count']:.1f}, "
            f"avg_legal={dataset_info['average_legal_actions']:.1f}"
        )
    print(
        f"  -> train unique={int(split_stats['train_unique_samples'])}, "
        f"train effective={int(split_stats['train_effective_samples'])}, "
        f"validation={int(split_stats['validation_samples'])}"
    )

    for epoch in range(1, config.bc_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_samples = 0

        for batch_action_features, batch_legal_masks, batch_actions in train_loader:
            batch_action_features = batch_action_features.to(device)
            batch_legal_masks = batch_legal_masks.to(device)
            batch_actions = batch_actions.to(device)

            logits = mask_illegal_q_values(model(batch_action_features), batch_legal_masks)
            loss = F.cross_entropy(logits, batch_actions)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            predictions = logits.argmax(dim=1)
            batch_size = int(batch_actions.shape[0])
            epoch_loss += float(loss.item()) * batch_size
            epoch_correct += int((predictions == batch_actions).sum().item())
            epoch_samples += batch_size
            global_step += 1

        train_metrics = {
            "loss": epoch_loss / max(1, epoch_samples),
            "accuracy": epoch_correct / max(1, epoch_samples),
        }
        validation_metrics = evaluate_imitation_loader(
            model,
            validation_loader,
            device=device,
        )

        writer.add_scalar("bc/train_loss", train_metrics["loss"], epoch)
        writer.add_scalar("bc/train_accuracy", train_metrics["accuracy"], epoch)
        writer.add_scalar("bc/val_loss", validation_metrics["loss"], epoch)
        writer.add_scalar("bc/val_accuracy", validation_metrics["accuracy"], epoch)
        writer.add_scalar("bc/updates", global_step, epoch)

        print(
            f"bc epoch={epoch:02d}/{config.bc_epochs:02d} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.3f} "
            f"val_loss={validation_metrics['loss']:.4f} "
            f"val_acc={validation_metrics['accuracy']:.3f}"
        )

        score = validation_metrics["accuracy"] if validation_loader is not None else train_metrics["accuracy"]
        if score >= best_accuracy:
            best_accuracy = score
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)

    final_validation = evaluate_imitation_loader(model, validation_loader, device=device)
    result = {
        **dataset_stats,
        **split_stats,
        "best_val_accuracy": best_accuracy,
        "final_val_loss": final_validation["loss"],
        "final_val_accuracy": final_validation["accuracy"],
    }
    writer.add_text("bc/final_metrics", json.dumps(result, indent=2))
    return result


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    config = TrainingConfig()
    if args.episodes is not None:
        config.episodes = args.episodes
    if args.bc_epochs is not None:
        config.bc_epochs = args.bc_epochs
    if args.bc_batch_size is not None:
        config.bc_batch_size = args.bc_batch_size
    if args.bc_learning_rate is not None:
        config.bc_learning_rate = args.bc_learning_rate
    if args.hidden1_size is not None:
        config.hidden1_size = args.hidden1_size
    if args.hidden2_size is not None:
        config.hidden2_size = args.hidden2_size
    if args.hidden3_size is not None:
        config.hidden3_size = args.hidden3_size
    if args.seed is not None:
        config.seed = args.seed
    config.device = args.device

    default_paths = config.resolve(repo_root)
    export_header_path = args.export_header or default_paths["export_header_path"]
    paths = config.resolve(repo_root, export_header_path=export_header_path)
    requested_dataset_paths = args.dataset or list(paths["dataset_paths"])
    export_namespace = args.export_namespace
    for key in ("runs_dir", "checkpoints_dir"):
        paths[key].mkdir(parents=True, exist_ok=True)

    missing_dataset_paths = [path for path in requested_dataset_paths if not path.exists()]
    if args.dataset is not None and missing_dataset_paths:
        missing_text = ", ".join(str(path) for path in missing_dataset_paths)
        raise FileNotFoundError(f"imitation dataset(s) not found: {missing_text}")
    available_dataset_paths = [path for path in requested_dataset_paths if path.exists()]

    set_seed(config.seed)
    torch.set_float32_matmul_precision("high")
    device = resolve_device(config.device)
    run_name = time.strftime("xrz_policy_%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=str(paths["runs_dir"] / run_name))
    writer.add_text("config/json", json.dumps(config.as_dict(), indent=2))
    writer.add_text("features/list", "\n".join(FEATURE_NAMES))
    writer.add_text("device/selected", str(device))
    writer.add_text(
        "export/target",
        json.dumps(
            {
                "header": str(export_header_path),
                "namespace": export_namespace,
                "checkpoint": str(paths["checkpoint_path"]),
                "best_checkpoint": str(paths["best_checkpoint_path"]),
            },
            indent=2,
        ),
    )
    writer.add_text(
        "model/architecture",
        json.dumps(
            {
                "input_dim": len(FEATURE_NAMES),
                "hidden1_size": config.hidden1_size,
                "hidden2_size": config.hidden2_size,
                "hidden3_size": config.hidden3_size,
            },
            indent=2,
        ),
    )

    online_net = ActionValueNet(
        hidden1_size=config.hidden1_size,
        hidden2_size=config.hidden2_size,
        hidden3_size=config.hidden3_size,
    ).to(device)
    target_net = ActionValueNet(
        hidden1_size=config.hidden1_size,
        hidden2_size=config.hidden2_size,
        hidden3_size=config.hidden3_size,
    ).to(device)
    target_net.load_state_dict(online_net.state_dict())
    target_net.eval()

    checkpoint_candidates = [args.checkpoint] if args.checkpoint is not None else [
        paths["checkpoint_path"],
        paths["best_checkpoint_path"],
    ]
    checkpoint_to_use = next(
        (candidate for candidate in checkpoint_candidates if candidate.exists()),
        checkpoint_candidates[0],
    )
    start_episode = 1
    total_steps = 0
    best_win_rate = -1.0

    should_try_load_checkpoint = (
        args.eval_only
        or args.checkpoint is not None
        or args.skip_bc
        or not available_dataset_paths
    )
    if should_try_load_checkpoint and checkpoint_to_use.exists():
        payload = maybe_load_checkpoint(
            checkpoint_to_use,
            online_net,
            optimizer=None,
            strict=args.eval_only or args.checkpoint is not None,
        )
        if payload is not None:
            target_net.load_state_dict(online_net.state_dict())
            start_episode = read_int(payload, "episode", 0) + 1
            total_steps = read_int(payload, "total_steps", 0)
            best_win_rate = read_float(payload, "best_win_rate", -1.0)
            print(f"Loaded checkpoint from {checkpoint_to_use}")
    elif args.eval_only:
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_to_use}")

    if args.eval_only:
        metrics = evaluate_policy(
            online_net,
            config,
            device=device,
            episodes=config.eval_episodes,
        )
        export_cpp_header(
            online_net,
            export_header_path,
            namespace_name=export_namespace,
        )
        print(json.dumps(metrics, indent=2))
        writer.close()
        return

    if not args.skip_bc:
        if available_dataset_paths:
            if args.bc_init is not None:
                warm_start_summary = warm_start_model(online_net, args.bc_init)
                target_net.load_state_dict(online_net.state_dict())
                writer.add_text("bc/warm_start", json.dumps(warm_start_summary, indent=2))
                print(
                    f"Warm-started BC from {warm_start_summary['source_path']} "
                    f"({warm_start_summary['source_type']}, "
                    f"copied {warm_start_summary['copied_parameter_count']} parameter tensors)."
                )
            bc_metrics = run_behavior_cloning(
                online_net,
                config,
                dataset_paths=available_dataset_paths,
                device=device,
                writer=writer,
            )
            print(
                f"Behavior cloning complete. best_val_accuracy={bc_metrics['best_val_accuracy']:.3f} "
                f"final_val_accuracy={bc_metrics['final_val_accuracy']:.3f}"
            )
            export_cpp_header(
                online_net,
                export_header_path,
                namespace_name=export_namespace,
            )
            target_net.load_state_dict(online_net.state_dict())
            start_episode = 1
            total_steps = 0
            best_win_rate = -1.0
        else:
            missing_text = ", ".join(str(path) for path in requested_dataset_paths)
            print(f"Imitation dataset not found in [{missing_text}]; skipping BC pretraining.")

    if args.skip_rl or config.episodes <= 0:
        metrics = evaluate_policy(
            online_net,
            config,
            device=device,
            episodes=config.eval_episodes,
        )
        writer.add_text("eval/final", json.dumps(metrics, indent=2))
        export_cpp_header(
            online_net,
            export_header_path,
            namespace_name=export_namespace,
        )
        print(json.dumps(metrics, indent=2))
        writer.flush()
        writer.close()
        return

    optimizer = optim.AdamW(
        online_net.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    replay_buffer = ReplayBuffer(config.replay_capacity)
    env = LocalGenMiniEnv(
        seed=config.seed,
        board_min_size=config.board_min_size,
        board_max_size=config.board_max_size,
        max_half_turns=config.max_half_turns,
    )
    action_rng = random.Random(config.seed + 99)
    updates = 0

    for episode in range(start_episode, config.episodes + 1):
        observation = env.reset()
        done = False
        episode_return = 0.0
        episode_steps = 0
        last_info: dict[str, object] = {}

        while not done:
            epsilon = linear_epsilon(config, total_steps)
            action = choose_action(
                online_net,
                observation,
                epsilon,
                device=device,
                rng=action_rng,
            )
            next_observation, reward, done, last_info = env.step(action)
            replay_buffer.push(
                Transition(
                    action_features=[list(features) for features in observation.action_features],
                    legal_mask=list(observation.legal_mask),
                    action=action,
                    reward=reward,
                    next_action_features=[list(features) for features in next_observation.action_features],
                    next_legal_mask=list(next_observation.legal_mask),
                    done=done,
                )
            )

            observation = next_observation
            episode_return += reward
            episode_steps += 1
            total_steps += 1

            if total_steps % config.train_interval == 0:
                train_metrics = optimize_model(
                    online_net,
                    target_net,
                    optimizer,
                    replay_buffer,
                    config,
                    device=device,
                )
                if train_metrics is not None:
                    updates += 1
                    writer.add_scalar("train/loss", train_metrics["loss"], updates)
                    writer.add_scalar("train/q_mean", train_metrics["q_mean"], updates)
                    writer.add_scalar("train/target_mean", train_metrics["target_mean"], updates)

            if total_steps % config.target_sync_interval == 0:
                target_net.load_state_dict(online_net.state_dict())

        winner = last_info.get("winner")
        writer.add_scalar("episode/return", episode_return, episode)
        writer.add_scalar("episode/steps", episode_steps, episode)
        writer.add_scalar("episode/epsilon", linear_epsilon(config, total_steps), episode)
        writer.add_scalar("episode/win", 1.0 if winner == 0 else 0.0, episode)
        writer.add_scalar("episode/loss", 1.0 if winner == 1 else 0.0, episode)
        writer.add_scalar("buffer/size", len(replay_buffer), episode)

        if episode % config.log_interval == 0 or episode == 1:
            print(
                f"episode={episode:04d} steps={total_steps:06d} return={episode_return:7.2f} "
                f"winner={winner} epsilon={linear_epsilon(config, total_steps):.3f} buffer={len(replay_buffer)}"
            )

        if episode % config.eval_interval == 0 or episode == config.episodes:
            metrics = evaluate_policy(
                online_net,
                config,
                device=device,
                episodes=config.eval_episodes,
            )
            writer.add_scalar("eval/average_return", metrics["average_return"], episode)
            writer.add_scalar("eval/win_rate", metrics["win_rate"], episode)
            writer.add_scalar("eval/loss_rate", metrics["loss_rate"], episode)
            writer.add_scalar("eval/draw_rate", metrics["draw_rate"], episode)
            writer.add_scalar("eval/average_steps", metrics["average_steps"], episode)
            print(
                f"eval episode={episode:04d} avg_return={metrics['average_return']:.2f} "
                f"win_rate={metrics['win_rate']:.3f} draw_rate={metrics['draw_rate']:.3f}"
            )

            if metrics["win_rate"] >= best_win_rate:
                best_win_rate = metrics["win_rate"]
                save_checkpoint(
                    paths["best_checkpoint_path"],
                    online_net,
                    optimizer,
                    config=config,
                    episode=episode,
                    total_steps=total_steps,
                    best_win_rate=best_win_rate,
                    device_name=str(device),
                )
                export_cpp_header(
                    online_net,
                    export_header_path,
                    namespace_name=export_namespace,
                )

        save_checkpoint(
            paths["checkpoint_path"],
            online_net,
            optimizer,
            config=config,
            episode=episode,
            total_steps=total_steps,
            best_win_rate=best_win_rate,
            device_name=str(device),
        )

    if paths["best_checkpoint_path"].exists():
        maybe_load_checkpoint(paths["best_checkpoint_path"], online_net, optimizer=None, strict=False)
    export_cpp_header(
        online_net,
        export_header_path,
        namespace_name=export_namespace,
    )
    writer.flush()
    writer.close()
    print(f"Training complete. Exported policy header to {export_header_path}")


if __name__ == "__main__":
    main()
