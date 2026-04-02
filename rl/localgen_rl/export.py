from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Iterable

import torch

from .model import ActionValueNet


_SIZE_RE = re.compile(r"inline constexpr std::size_t\s+(k\w+)\s*=\s*(\d+)\s*;")


def _make_array_re(name: str) -> re.Pattern[str]:
    return re.compile(
        rf"inline constexpr std::array<double,\s*[^>]+>\s+{re.escape(name)}\s*=\s*\{{(.*?)\}};",
        re.DOTALL,
    )


def _format_cpp_array(values: Iterable[float], indent: str = "    ") -> str:
    rendered = [f"{float(value):.8f}" for value in values]
    lines: list[str] = []
    chunk_size = 6
    for i in range(0, len(rendered), chunk_size):
        chunk = ", ".join(rendered[i : i + chunk_size])
        lines.append(f"{indent}{chunk},")
    if not lines:
        lines.append(f"{indent}")
    return "\n".join(lines)


def _cpp_identifier(text: str) -> str:
    identifier = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in text)
    identifier = identifier.strip("_")
    if not identifier:
        raise ValueError("C++ identifier cannot be empty")
    if identifier[0].isdigit():
        identifier = f"_{identifier}"
    return identifier


def _header_guard(output_path: Path) -> str:
    return f"LGEN_BOTS_GENERATED_{_cpp_identifier(output_path.name).upper()}"


def _parse_cpp_array_values(serialized: str) -> list[float]:
    values: list[float] = []
    for token in serialized.replace("\n", " ").split(","):
        stripped = token.strip()
        if not stripped:
            continue
        values.append(float(stripped))
    return values


def _require_cpp_array(
    header_text: str,
    name: str,
    *,
    expected_size: int | None = None,
) -> list[float]:
    match = _make_array_re(name).search(header_text)
    if match is None:
        raise ValueError(f"failed to locate {name} in exported header")
    values = _parse_cpp_array_values(match.group(1))
    if expected_size is not None and len(values) != expected_size:
        raise ValueError(
            f"exported header array {name} has {len(values)} values, expected {expected_size}"
        )
    return values


def infer_model_architecture(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, int]:
    layer1_weight = state_dict.get("backbone.0.weight")
    layer2_weight = state_dict.get("backbone.2.weight")
    if layer1_weight is None or layer2_weight is None:
        raise ValueError("state dict is missing one or more backbone layers")

    hidden1_size, input_dim = layer1_weight.shape
    hidden2_size, _ = layer2_weight.shape
    layer3_weight = state_dict.get("backbone.4.weight")
    hidden3_size = int(layer3_weight.shape[0]) if layer3_weight is not None else 0

    return {
        "input_dim": int(input_dim),
        "hidden1_size": int(hidden1_size),
        "hidden2_size": int(hidden2_size),
        "hidden3_size": hidden3_size,
    }


def load_exported_header(header_path: Path) -> dict[str, Any]:
    header_text = header_path.read_text(encoding="utf-8")
    size_constants = {name: int(value) for name, value in _SIZE_RE.findall(header_text)}

    input_dim = size_constants.get("kInputSize")
    hidden1_size = size_constants.get("kHidden1Size")
    hidden2_size = size_constants.get("kHidden2Size")
    hidden3_size = size_constants.get("kHidden3Size", 0)
    if input_dim is None or hidden1_size is None or hidden2_size is None:
        raise ValueError(f"exported header is missing required size constants: {header_path}")

    output_input_size = hidden3_size if hidden3_size > 0 else hidden2_size

    state_dict: dict[str, torch.Tensor] = {
        "backbone.0.weight": torch.tensor(
            _require_cpp_array(
                header_text,
                "kLayer1Weights",
                expected_size=input_dim * hidden1_size,
            ),
            dtype=torch.float32,
        ).reshape(hidden1_size, input_dim),
        "backbone.0.bias": torch.tensor(
            _require_cpp_array(
                header_text,
                "kLayer1Bias",
                expected_size=hidden1_size,
            ),
            dtype=torch.float32,
        ),
        "backbone.2.weight": torch.tensor(
            _require_cpp_array(
                header_text,
                "kLayer2Weights",
                expected_size=hidden1_size * hidden2_size,
            ),
            dtype=torch.float32,
        ).reshape(hidden2_size, hidden1_size),
        "backbone.2.bias": torch.tensor(
            _require_cpp_array(
                header_text,
                "kLayer2Bias",
                expected_size=hidden2_size,
            ),
            dtype=torch.float32,
        ),
        "q_head.weight": torch.tensor(
            _require_cpp_array(
                header_text,
                "kOutputWeights",
                expected_size=output_input_size,
            ),
            dtype=torch.float32,
        ).reshape(1, output_input_size),
        "q_head.bias": torch.tensor(
            _require_cpp_array(
                header_text,
                "kOutputBias",
                expected_size=1,
            ),
            dtype=torch.float32,
        ),
    }

    if hidden3_size > 0:
        state_dict["backbone.4.weight"] = torch.tensor(
            _require_cpp_array(
                header_text,
                "kLayer3Weights",
                expected_size=hidden2_size * hidden3_size,
            ),
            dtype=torch.float32,
        ).reshape(hidden3_size, hidden2_size)
        state_dict["backbone.4.bias"] = torch.tensor(
            _require_cpp_array(
                header_text,
                "kLayer3Bias",
                expected_size=hidden3_size,
            ),
            dtype=torch.float32,
        )

    return {
        "source_type": "header",
        "source_path": str(header_path),
        "architecture": {
            "input_dim": input_dim,
            "hidden1_size": hidden1_size,
            "hidden2_size": hidden2_size,
            "hidden3_size": hidden3_size,
        },
        "state_dict": state_dict,
    }


def load_model_source(source_path: Path) -> dict[str, Any]:
    if not source_path.exists():
        raise FileNotFoundError(f"model source not found: {source_path}")

    suffix = source_path.suffix.lower()
    if suffix in {".h", ".hh", ".hpp"}:
        return load_exported_header(source_path)

    payload = torch.load(source_path, map_location="cpu")
    source_type = "checkpoint"
    if isinstance(payload, dict) and "model_state_dict" in payload:
        raw_state_dict = payload["model_state_dict"]
    elif isinstance(payload, dict) and payload and all(torch.is_tensor(value) for value in payload.values()):
        raw_state_dict = payload
        source_type = "state_dict"
    else:
        raise ValueError(f"unsupported model source payload in {source_path}")

    state_dict = {
        name: tensor.detach().cpu().to(dtype=torch.float32).clone()
        for name, tensor in raw_state_dict.items()
    }

    return {
        "source_type": source_type,
        "source_path": str(source_path),
        "architecture": infer_model_architecture(state_dict),
        "state_dict": state_dict,
    }


def _copy_tensor_overlap(target_tensor: torch.Tensor, source_tensor: torch.Tensor) -> int:
    if target_tensor.ndim != source_tensor.ndim:
        return 0

    overlap_shape = tuple(min(dst, src) for dst, src in zip(target_tensor.shape, source_tensor.shape))
    if any(size <= 0 for size in overlap_shape):
        return 0

    slices = tuple(slice(0, size) for size in overlap_shape)
    target_tensor[slices] = source_tensor[slices].to(dtype=target_tensor.dtype)
    return math.prod(overlap_shape)


def warm_start_model(model: ActionValueNet, source_path: Path) -> dict[str, Any]:
    loaded = load_model_source(source_path)
    source_state = loaded["state_dict"]
    target_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }

    copied_keys: set[str] = set()
    copied_tensor_elements = 0

    for name, target_tensor in target_state.items():
        source_tensor = source_state.get(name)
        if source_tensor is None:
            continue
        if name == "q_head.weight" and target_tensor.shape != source_tensor.shape:
            continue
        copied = _copy_tensor_overlap(target_tensor, source_tensor)
        if copied > 0:
            copied_keys.add(name)
            copied_tensor_elements += copied

    adaptation_notes: list[str] = []
    source_architecture = dict(loaded["architecture"])
    target_architecture = {
        "input_dim": model.input_dim,
        "hidden1_size": model.hidden1_size,
        "hidden2_size": model.hidden2_size,
        "hidden3_size": model.hidden3_size,
    }

    if model.hidden3_size > 0 and source_architecture["hidden3_size"] == 0:
        layer3_weight = target_state.get("backbone.4.weight")
        layer3_bias = target_state.get("backbone.4.bias")
        source_output_weight = source_state.get("q_head.weight")
        source_output_bias = source_state.get("q_head.bias")
        q_head_weight = target_state.get("q_head.weight")
        q_head_bias = target_state.get("q_head.bias")

        if (
            layer3_weight is not None
            and layer3_bias is not None
            and source_output_weight is not None
            and q_head_weight is not None
            and q_head_bias is not None
        ):
            layer3_weight.zero_()
            layer3_bias.zero_()
            q_head_weight.zero_()
            identity_dim = min(
                int(layer3_weight.shape[0]),
                int(layer3_weight.shape[1]),
                int(source_output_weight.shape[1]),
            )
            if identity_dim > 0:
                diag = torch.arange(identity_dim, dtype=torch.long)
                layer3_weight[diag, diag] = 1.0
                q_head_weight[0, :identity_dim] = source_output_weight[0, :identity_dim].to(
                    dtype=q_head_weight.dtype
                )
                copied_tensor_elements += identity_dim * 2
            if source_output_bias is not None:
                copied_tensor_elements += _copy_tensor_overlap(q_head_bias, source_output_bias)

            copied_keys.update({"backbone.4.weight", "backbone.4.bias", "q_head.weight", "q_head.bias"})
            adaptation_notes.append(
                "Inserted an identity-like third hidden layer so a 2-layer source can warm-start a 3-layer target."
            )

    if not copied_keys:
        raise ValueError(f"no compatible parameters could be loaded from {source_path}")

    model.load_state_dict(target_state)
    return {
        "source_type": loaded["source_type"],
        "source_path": loaded["source_path"],
        "source_architecture": source_architecture,
        "target_architecture": target_architecture,
        "copied_parameter_keys": sorted(copied_keys),
        "copied_parameter_count": len(copied_keys),
        "copied_tensor_elements": copied_tensor_elements,
        "adaptation_notes": adaptation_notes,
    }


def export_cpp_header(
    model: ActionValueNet,
    output_path: Path,
    *,
    namespace_name: str = "xrz_rl_model",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    state = {name: tensor.detach().cpu().flatten().tolist() for name, tensor in model.state_dict().items()}
    input_dim, hidden1_size, hidden2_size, hidden3_size = model.architecture()
    output_weight_size_name = "kHidden3Size" if hidden3_size > 0 else "kHidden2Size"
    namespace_identifier = _cpp_identifier(namespace_name)
    guard = _header_guard(output_path)

    header = f"""/**
 * @file {output_path.name}
 *
 * Auto-generated by rl/train_xrz_dqn.py.
 * Re-run the trainer to refresh these parameters.
 */

#ifndef {guard}
#define {guard}

#include <array>
#include <cstddef>

namespace {namespace_identifier} {{

inline constexpr std::size_t kInputSize = {input_dim};
inline constexpr std::size_t kHidden1Size = {hidden1_size};
inline constexpr std::size_t kHidden2Size = {hidden2_size};
inline constexpr std::size_t kHidden3Size = {hidden3_size};

inline constexpr std::array<double, kInputSize * kHidden1Size> kLayer1Weights = {{
{_format_cpp_array(state['backbone.0.weight'])}
}};

inline constexpr std::array<double, kHidden1Size> kLayer1Bias = {{
{_format_cpp_array(state['backbone.0.bias'])}
}};

inline constexpr std::array<double, kHidden1Size * kHidden2Size> kLayer2Weights = {{
{_format_cpp_array(state['backbone.2.weight'])}
}};

inline constexpr std::array<double, kHidden2Size> kLayer2Bias = {{
{_format_cpp_array(state['backbone.2.bias'])}
}};

inline constexpr std::array<double, kHidden2Size * kHidden3Size> kLayer3Weights = {{
{_format_cpp_array(state.get('backbone.4.weight', []))}
}};

inline constexpr std::array<double, kHidden3Size> kLayer3Bias = {{
{_format_cpp_array(state.get('backbone.4.bias', []))}
}};

inline constexpr std::array<double, {output_weight_size_name}> kOutputWeights = {{
{_format_cpp_array(state['q_head.weight'])}
}};

inline constexpr std::array<double, 1> kOutputBias = {{
{_format_cpp_array(state['q_head.bias'])}
}};

}}  // namespace {namespace_identifier}

#endif  // {guard}
"""

    output_path.write_text(header)
