# XrzBot reinforcement learning workspace

This folder contains the Python side of the current `XrzBot` training stack.
The workflow is now imitation-first: it can learn from real LocalGen games dumped
from a stronger teacher bot, then optionally run reinforcement-learning
fine-tuning in a lightweight LocalGen-style environment that matches the same
feature encoder used by the C++ bot.

## What it does

- runs behavior cloning from JSONL demonstrations generated from real matches
- optionally performs RL fine-tuning after pretraining
- uses the same 37-feature split-move action encoding as `src/bots/xrzPolicy.h`
- logs behavior-cloning and RL metrics to TensorBoard
- prefers Apple Silicon `mps` automatically when available
- exports the trained policy directly to `src/bots/generated/xrzRlWeights.h`
- stores checkpoints under `rl/checkpoints/`

## Main pieces

- `train_xrz_dqn.py` — training entrypoint for behavior cloning and optional RL
- `localgen_rl/env.py` — LocalGen-inspired mini environment aligned with the
	shared action/feature schema
- `localgen_rl/model.py` — MLP policy/value scorer over per-action features
- `localgen_rl/export.py` — exports PyTorch weights into the C++ header format
- `localgen_rl/constants.py` — canonical action-space and feature definitions
- `../simulator/xrzImitationDump.cpp` — real-game imitation dataset dumper

## Typical workflow

1. Build `LocalGen-bot-imitation-dump` and generate a dataset under
	 `rl/datasets/`.
2. Train with behavior cloning, optionally followed by RL fine-tuning.
3. Let the trainer export `src/bots/generated/xrzRlWeights.h`.
4. Rebuild the C++ project to deploy the updated `XrzBot`.

## Training

From the repository root, the default run uses the configured dataset path and
performs behavior cloning followed by RL fine-tuning when enabled.

Useful variants include:

- BC only: `python rl/train_xrz_dqn.py --skip-rl`
- RL only from an existing checkpoint/header seed: `python rl/train_xrz_dqn.py --skip-bc`
- custom dataset: `python rl/train_xrz_dqn.py --dataset rl/datasets/your_dump.jsonl`
- short smoke run: `python rl/train_xrz_dqn.py --skip-rl --bc-epochs 1`

## TensorBoard

Training logs are written under `rl/runs/`.

Open TensorBoard with:

`tensorboard --logdir rl/runs`

## Output

- latest checkpoint: `rl/checkpoints/xrz_dqn.pt`
- best checkpoint: `rl/checkpoints/xrz_dqn_best.pt`
- exported C++ weights: `src/bots/generated/xrzRlWeights.h`

`src/bots/xrzBot.cpp` consumes the exported header directly, so rebuilding the
project is enough to deploy the most recently exported policy. A 30-feature
older header can still compile because the shared encoder keeps the leading
features backward-compatible, but new training runs export the full 37-feature
model.
