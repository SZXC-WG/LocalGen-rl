# XrzBot reinforcement learning workspace

This folder contains the Python side of the current `XrzBot` training stack.
The workflow is now imitation-first: it can learn from real LocalGen games dumped
from a stronger teacher bot, then optionally run reinforcement-learning
fine-tuning in a lightweight LocalGen-style environment that matches the same
feature encoder used by the C++ bot.

The bot remains fully self-contained at runtime: other bots are only used as
teachers or opponents while generating datasets and evaluating strength, never
as direct delegates inside `XrzBot` itself.

## What it does

- runs behavior cloning from JSONL demonstrations generated from real matches
- optionally performs RL fine-tuning after pretraining
- uses the same 39-feature split-move action encoding as `src/bots/xrzPolicy.h`
- logs behavior-cloning and RL metrics to TensorBoard
- prefers Apple Silicon `mps` automatically when available
- exports trained policies directly to C++ headers under `src/bots/generated/`
- stores checkpoints under `rl/checkpoints/`
- zero-pads older 37-feature datasets automatically so legacy corpora can be
	mixed with newer dumps that include the extra feature channels

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
3. Let the trainer export specialist headers such as
	`src/bots/generated/xrzRlWeightsDuel.h` and
	`src/bots/generated/xrzRlWeightsFfa.h`.
4. Rebuild the C++ project to deploy the updated `XrzBot`.

## Training

From the repository root, the default run uses the configured dataset path and
performs behavior cloning followed by RL fine-tuning when enabled.

Useful variants include:

- BC only: `python rl/train_xrz_dqn.py --skip-rl`
- RL only from an existing checkpoint/header seed: `python rl/train_xrz_dqn.py --skip-bc`
- custom dataset: `python rl/train_xrz_dqn.py --dataset rl/datasets/your_dump.jsonl`
- short smoke run: `python rl/train_xrz_dqn.py --skip-rl --bc-epochs 1`

For the current strongest specialist recipe in this repository, train duel and
FFA separately with the larger default MLP and mixed real-game corpora:

- duel:
	`python rl/train_xrz_dqn.py --skip-rl --device mps --dataset rl/datasets/xrz_allteachers_duel_maps.jsonl --dataset rl/datasets/xrz_strong_duel.jsonl --dataset rl/datasets/xrz_selfpolicy_duel_train.jsonl --export-header src/bots/generated/xrzRlWeightsDuel.h --export-namespace xrz_rl_duel_model`
- free-for-all:
	`python rl/train_xrz_dqn.py --skip-rl --device mps --dataset rl/datasets/xrz_allteachers_ffa_maps.jsonl --dataset rl/datasets/xrz_strong_ffa.jsonl --dataset rl/datasets/xrz_selfpolicy_ffa_train.jsonl --export-header src/bots/generated/xrzRlWeightsFfa.h --export-namespace xrz_rl_ffa_model`

## TensorBoard

Training logs are written under `rl/runs/`.

Open TensorBoard with:

`tensorboard --logdir rl/runs`

## Output

- latest checkpoint: `rl/checkpoints/xrz_dqn.pt`
- best checkpoint: `rl/checkpoints/xrz_dqn_best.pt`
- exported generic C++ weights: `src/bots/generated/xrzRlWeights.h`
- exported duel specialist: `src/bots/generated/xrzRlWeightsDuel.h`
- exported FFA specialist: `src/bots/generated/xrzRlWeightsFfa.h`

`src/bots/xrzBot.cpp` consumes the exported specialist headers directly, so
rebuilding the project is enough to deploy the most recently exported policy.
Older 37-feature datasets and headers can still be reused because the shared
encoder keeps the leading features backward-compatible and the trainer now pads
missing feature slots automatically.
