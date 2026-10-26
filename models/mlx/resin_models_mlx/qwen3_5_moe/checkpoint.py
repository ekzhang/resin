"""Load Hugging Face Qwen3.5-MoE checkpoints into the MLX model."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from resin.backend.loading import read_config
from resin.models.qwen3_5.config import Qwen35Config
from resin_models_mlx.qwen3_5_moe.model import Qwen35MoeModel

TEXT_PREFIX = "model.language_model."
# Qwen stores these RMSNorm weights as offsets from one.
_OFFSET_NORMS = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    ".q_norm.weight",
    ".k_norm.weight",
)
_EXPERT_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def load_config(model_path: str | Path) -> tuple[Qwen35Config, dict]:
    raw_config = read_config(Path(model_path))
    config = Qwen35Config.from_hugging_face(raw_config)
    if not config.is_moe:
        raise ValueError("the MLX Qwen3.5 runner requires a Qwen3.5-MoE checkpoint")
    if raw_config.get("quantization_config"):
        raise ValueError("the MLX Qwen3.5 runner loads unquantized BF16 checkpoints")
    return config, raw_config


def _tensor_files(model_path: Path) -> dict[str, Path]:
    index = model_path / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        return {name: model_path / file for name, file in weight_map.items()}
    single = model_path / "model.safetensors"
    return {name: single for name in _load(single)}


def _load(file: Path) -> dict[str, mx.array]:
    # Safetensors arrays are memory-mapped and read when evaluated.
    return cast(dict[str, mx.array], mx.load(str(file)))


def _text_name(name: str) -> str | None:
    if name == "lm_head.weight":
        return name
    if name.startswith(TEXT_PREFIX):
        return name[len(TEXT_PREFIX) :]
    # Vision tower and MTP head weights are outside this text runner.
    return None


def checkpoint_weights(
    model_path: str | Path, config: Qwen35Config
) -> Iterator[tuple[str, mx.array]]:
    """Yield lazily loaded tensors under the MLX model's parameter names."""

    model_path = Path(model_path)
    files = _tensor_files(model_path)
    shards: dict[Path, dict[str, mx.array]] = {}

    def tensor(name: str) -> mx.array:
        file = files[name]
        if file not in shards:
            shards[file] = _load(file)
        return shards[file][name]

    experts: dict[str, dict[int, str]] = {}
    for name in files:
        text = _text_name(name)
        if text is None:
            continue
        parts = text.split(".")
        if len(parts) == 7 and parts[3] == "experts":
            # layers.{i}.mlp.experts.{e}.{projection}.weight
            layer, expert, projection = parts[1], int(parts[4]), parts[5]
            key = f"layers.{layer}.mlp.switch_mlp.{projection}.weight"
            experts.setdefault(key, {})[expert] = name
            continue
        value = tensor(name)
        if text.endswith(_OFFSET_NORMS) or text == "norm.weight":
            value = value.astype(mx.float32) + 1.0
        elif text.endswith("linear_attn.conv1d.weight"):
            value = value.moveaxis(2, 1)
        yield text, value

    for key, names in experts.items():
        if sorted(names) != list(range(config.num_experts)):
            raise ValueError(f"checkpoint is missing experts for {key}")
        yield key, mx.stack([tensor(names[index]) for index in sorted(names)])


def _quantizable(group_size: int):
    def predicate(path: str, module: nn.Module) -> bool:
        weight = getattr(module, "weight", None)
        return (
            hasattr(module, "to_quantized")
            and not path.endswith("conv1d")
            and weight is not None
            and weight.shape[-1] % group_size == 0
        )

    return predicate


def load_model(
    model_path: str | Path,
    *,
    bits: int | None = 8,
    group_size: int = 64,
) -> tuple[Qwen35MoeModel, Qwen35Config, dict]:
    """Build the text model, load BF16 weights and optionally quantize them."""

    config, raw_config = load_config(model_path)
    model = Qwen35MoeModel(config)
    expected = {name for name, _ in tree_flatten(model.parameters())}
    weights = dict(checkpoint_weights(model_path, config))
    if missing := sorted(expected - set(weights)):
        raise ValueError(f"checkpoint is missing parameters: {missing[:8]}")
    if unexpected := sorted(set(weights) - expected):
        raise ValueError(f"checkpoint has unexpected parameters: {unexpected[:8]}")
    # The DeltaNet decay parameter keeps its checkpoint precision.
    model.load_weights(
        [
            (name, value if name.endswith("A_log") else value.astype(mx.bfloat16))
            for name, value in weights.items()
        ]
    )
    del weights
    if bits is not None:
        nn.quantize(
            model,
            group_size=group_size,
            bits=bits,
            class_predicate=_quantizable(group_size),
        )
    # Materialize one layer at a time so BF16 sources are released as each
    # layer's quantized weights are produced.
    mx.eval(model.embed_tokens.parameters())
    for layer in model.layers:
        mx.eval(layer.parameters())
    mx.eval(model.norm.parameters(), model.lm_head.parameters())
    return model, config, raw_config


__all__ = ["checkpoint_weights", "load_config", "load_model"]
