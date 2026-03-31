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
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from localgen_rl import (
    FEATURE_NAMES,
    ActionValueNet,
    LocalGenMiniEnv,
    ReplayBuffer,
    TrainingConfig,
    Transition,
    export_cpp_header,
)
from localgen_rl.model import mask_illegal_q_values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the XrzBot policy with behavior cloning and RL fine-tuning."
    )
    parser.add_argument("--episodes", type=int, default=None, help="Override RL fine-tuning episode count")
    parser.add_argument("--bc-epochs", type=int, default=None, help="Override behavior-cloning epoch count")
    parser.add_argument(
        "--dataset",
        type=Path,
        action="append",
        default=None,
        help="Imitation dataset JSONL path. Repeat the flag to mix multiple files.",
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


def load_imitation_dataset(
    dataset_paths: Sequence[Path],
    *,
    expected_feature_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    if not dataset_paths:
        raise FileNotFoundError("no imitation datasets were provided")

    action_features: list[list[list[float]]] = []
    legal_masks: list[list[bool]] = []
    actions: list[int] = []
    action_counts: list[int] = []
    max_action_count = 0
    dataset_breakdown: list[dict[str, Any]] = []

    for dataset_path in dataset_paths:
        if not dataset_path.exists():
            raise FileNotFoundError(f"imitation dataset not found: {dataset_path}")

        file_samples = 0
        file_action_counts: list[int] = []
        file_legal_counts: list[int] = []

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
                if any(
                    len(feature_row) != expected_feature_count
                    for feature_row in sample_features
                ):
                    raise ValueError(
                        f"dataset {dataset_path} line {line_number}: expected {expected_feature_count} features"
                    )

                const_action_count = len(sample_features)
                max_action_count = max(max_action_count, const_action_count)
                action_counts.append(const_action_count)
                file_action_counts.append(const_action_count)
                file_legal_counts.append(sum(sample_legal_mask))
                action_features.append(sample_features)
                legal_masks.append(sample_legal_mask)
                actions.append(sample_action)
                file_samples += 1

        if file_samples == 0:
            raise RuntimeError(f"imitation dataset is empty: {dataset_path}")

        dataset_breakdown.append(
            {
                "path": str(dataset_path),
                "sample_count": file_samples,
                "average_action_count": sum(file_action_counts) / max(1, file_samples),
                "average_legal_actions": sum(file_legal_counts) / max(1, file_samples),
            }
        )

    if not actions:
        raise RuntimeError("all provided imitation datasets were empty")

    action_features_tensor = torch.zeros(
        (len(actions), max_action_count, expected_feature_count),
        dtype=torch.float32,
    )
    legal_masks_tensor = torch.zeros((len(actions), max_action_count), dtype=torch.bool)
    for sample_index, (sample_features, sample_legal_mask) in enumerate(
        zip(action_features, legal_masks)
    ):
        sample_action_count = len(sample_features)
        action_features_tensor[sample_index, :sample_action_count] = torch.tensor(
            sample_features,
            dtype=torch.float32,
        )
        legal_masks_tensor[sample_index, :sample_action_count] = torch.tensor(
            sample_legal_mask,
            dtype=torch.bool,
        )
    actions_tensor = torch.tensor(actions, dtype=torch.long)
    legal_actions_per_state = legal_masks_tensor.sum(dim=1).float()
    stats = {
        "dataset_file_count": len(dataset_breakdown),
        "dataset_breakdown": dataset_breakdown,
        "sample_count": float(actions_tensor.shape[0]),
        "max_action_count": float(action_features_tensor.shape[1]),
        "min_action_count": float(min(action_counts)),
        "average_action_count": float(sum(action_counts) / max(1, len(action_counts))),
        "feature_count": float(action_features_tensor.shape[2]),
        "average_legal_actions": float(legal_actions_per_state.mean().item()),
        "min_legal_actions": float(legal_actions_per_state.min().item()),
        "max_legal_actions": float(legal_actions_per_state.max().item()),
    }
    return action_features_tensor, legal_masks_tensor, actions_tensor, stats


def make_supervised_loaders(
    action_features: torch.Tensor,
    legal_masks: torch.Tensor,
    actions: torch.Tensor,
    *,
    batch_size: int,
    validation_fraction: float,
    seed: int,
) -> tuple[DataLoader, Optional[DataLoader]]:
    dataset = TensorDataset(action_features, legal_masks, actions)
    total_size = len(dataset)
    if total_size == 1:
        return DataLoader(dataset, batch_size=1, shuffle=True), None

    validation_size = int(total_size * validation_fraction)
    if validation_fraction > 0.0:
        validation_size = max(1, validation_size)
    validation_size = min(validation_size, total_size - 1)
    train_size = total_size - validation_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, validation_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, validation_size],
        generator=generator,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, validation_loader


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
    action_features, legal_masks, actions, dataset_stats = load_imitation_dataset(
        dataset_paths,
        expected_feature_count=model.input_dim,
    )
    train_loader, validation_loader = make_supervised_loaders(
        action_features,
        legal_masks,
        actions,
        batch_size=config.bc_batch_size,
        validation_fraction=config.bc_validation_fraction,
        seed=config.seed,
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
    dataset_breakdown = list(dataset_stats["dataset_breakdown"])
    print(
        f"Loaded {int(dataset_stats['dataset_file_count'])} imitation dataset(s) with "
        f"{int(dataset_stats['sample_count'])} samples total, "
        f"{int(dataset_stats['min_action_count'])}-{int(dataset_stats['max_action_count'])} actions/state, "
        f"{int(dataset_stats['feature_count'])} features/action."
    )
    for dataset_info in dataset_breakdown:
        print(
            f"  - {dataset_info['path']}: {int(dataset_info['sample_count'])} samples, "
            f"avg_actions={dataset_info['average_action_count']:.1f}, "
            f"avg_legal={dataset_info['average_legal_actions']:.1f}"
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
    if args.seed is not None:
        config.seed = args.seed
    config.device = args.device

    paths = config.resolve(repo_root)
    requested_dataset_paths = args.dataset or list(paths["dataset_paths"])
    export_header_path = args.export_header or paths["export_header_path"]
    export_namespace = args.export_namespace
    for key in ("runs_dir", "checkpoints_dir"):
        paths[key].mkdir(parents=True, exist_ok=True)

    deduped_dataset_paths: list[Path] = []
    seen_dataset_paths: set[str] = set()
    for dataset_path in requested_dataset_paths:
        dataset_key = str(dataset_path)
        if dataset_key in seen_dataset_paths:
            continue
        seen_dataset_paths.add(dataset_key)
        deduped_dataset_paths.append(dataset_path)

    missing_dataset_paths = [path for path in deduped_dataset_paths if not path.exists()]
    if args.dataset is not None and missing_dataset_paths:
        missing_text = ", ".join(str(path) for path in missing_dataset_paths)
        raise FileNotFoundError(f"imitation dataset(s) not found: {missing_text}")
    available_dataset_paths = [path for path in deduped_dataset_paths if path.exists()]

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
            },
            indent=2,
        ),
    )

    online_net = ActionValueNet(
        hidden1_size=config.hidden1_size, hidden2_size=config.hidden2_size
    ).to(device)
    target_net = ActionValueNet(
        hidden1_size=config.hidden1_size, hidden2_size=config.hidden2_size
    ).to(device)
    target_net.load_state_dict(online_net.state_dict())
    target_net.eval()

    checkpoint_to_use = args.checkpoint or paths["checkpoint_path"]
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
            missing_text = ", ".join(str(path) for path in deduped_dataset_paths)
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
