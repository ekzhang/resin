"""Load BF16 Hugging Face or prequantized MLX Qwen3.5 checkpoints."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import cast

import mlx.core as mx
import mlx.nn as nn

from resin.backend.loading import read_config
from resin.backend.safetensors import indexed_tensor_files
from resin.models.qwen3_5.config import Qwen35Config
from resin_models_mlx.qwen3_5.model import Qwen35Model

# Multimodal exports nest the text model; text-only exports do not.
_TEXT_PREFIXES = ("language_model.model.", "model.language_model.", "model.")
_EXPERT_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
_DECAY_PARAMETERS = ("A_log", "dt_bias")


def load_config(model_path: str | Path) -> tuple[Qwen35Config, dict]:
    raw_config = read_config(Path(model_path))
    config = Qwen35Config.from_hugging_face(raw_config)
    checkpoint_quantization(raw_config)
    return config, raw_config


def checkpoint_quantization(raw_config: dict) -> dict | None:
    """Validate the uniform packed MLX format, without guessing other schemes."""
    configurations = []
    for key in ("quantization", "quantization_config"):
        value = raw_config.get(key)
        if value is None:
            continue
        if not isinstance(value, dict) or set(value) - {"bits", "group_size", "mode"}:
            raise ValueError("expected uniform affine MLX checkpoint quantization")
        bits, group_size, mode = (
            value.get("bits"),
            value.get("group_size"),
            value.get("mode", "affine"),
        )
        if (
            type(bits) is not int
            or bits not in (4, 8)
            or type(group_size) is not int
            or group_size not in (32, 64, 128)
            or mode != "affine"
        ):
            raise ValueError(
                "supported MLX checkpoint quantization is affine, 4 or 8 bits, "
                "with group_size 32, 64, or 128"
            )
        configurations.append(dict(bits=bits, group_size=group_size, mode=mode))
    if not configurations:
        return None
    if any(value != configurations[0] for value in configurations[1:]):
        raise ValueError("conflicting MLX checkpoint quantization metadata")
    return configurations[0]


class _Checkpoint:
    """Share validated shard lookup and lazy MLX mmap loading for text/vision."""

    def __init__(self, path: str | Path):
        self.files = indexed_tensor_files(Path(path))
        self._shards: dict[Path, dict[str, mx.array]] = {}

    def __getitem__(self, name: str) -> mx.array:
        path = self.files[name]
        if path not in self._shards:
            self._shards[path] = cast(dict[str, mx.array], mx.load(str(path)))
        return self._shards[path][name]


def _text_prefix(names) -> str:
    for prefix in _TEXT_PREFIXES:
        if any(name.startswith(prefix + "embed_tokens.") for name in names):
            return prefix
    raise ValueError("checkpoint is missing the Qwen3.5 text embedding")


def checkpoint_weights(
    model_path: str | Path, config: Qwen35Config, *, prequantized: bool = False
) -> Iterator[tuple[str, mx.array]]:
    """Yield lazily loaded tensors under the MLX model's parameter names.

    Experts may be stored one tensor per expert or fused as Transformers saves
    them (``experts.gate_up_proj`` and ``experts.down_proj``). Vision tower and
    MTP weights are outside this text runner and are skipped.
    """

    tensors = _Checkpoint(model_path)
    prefix = _text_prefix(tensors.files)
    experts: dict[str, dict[int, str]] = {}
    for name in tensors.files:
        head = name.removeprefix("language_model.")
        if head.startswith("lm_head."):
            # Tied checkpoints may also serialize the shared output weight.
            # The embedding is the single owned parameter in that case.
            if not config.tie_word_embeddings:
                yield head, tensors[name]
            continue
        if not name.startswith(prefix) or name.startswith("model.visual."):
            continue
        text = name[len(prefix) :]
        parts = text.split(".")
        if len(parts) == 7 and parts[3] == "experts":
            if prequantized:
                raise ValueError(
                    "prequantized per-expert weights require MLX switch_mlp layout"
                )
            # layers.{i}.mlp.experts.{e}.{projection}.weight
            layer, expert, projection = parts[1], int(parts[4]), parts[5]
            if projection not in _EXPERT_PROJECTIONS:
                raise ValueError(f"unexpected expert projection {name!r}")
            key = f"layers.{layer}.mlp.switch_mlp.{projection}.weight"
            experts.setdefault(key, {})[expert] = name
            continue
        if text.endswith("mlp.experts.gate_up_proj"):
            if prequantized:
                raise ValueError(
                    "prequantized fused experts require MLX switch_mlp layout"
                )
            # [experts, 2 * intermediate, hidden]: gate rows, then up rows.
            gate_up = tensors[name]
            middle = gate_up.shape[1] // 2
            base = text.removesuffix("experts.gate_up_proj") + "switch_mlp."
            yield base + "gate_proj.weight", gate_up[:, :middle]
            yield base + "up_proj.weight", gate_up[:, middle:]
            continue
        if text.endswith("mlp.experts.down_proj"):
            if prequantized:
                raise ValueError(
                    "prequantized fused experts require MLX switch_mlp layout"
                )
            base = text.removesuffix("experts.down_proj") + "switch_mlp."
            yield base + "down_proj.weight", tensors[name]
            continue
        value = tensors[name]
        if text.endswith("linear_attn.conv1d.weight"):
            channels, kernel = config.linear_conv_dim, config.linear_conv_kernel_dim
            if value.shape == (channels, 1, kernel):
                value = value.moveaxis(2, 1)
            elif value.shape != (channels, kernel, 1):
                raise ValueError(
                    f"unexpected Qwen3.5 convolution layout: {value.shape}"
                )
        yield text, value

    for key, names in experts.items():
        if sorted(names) != list(range(config.num_experts)):
            raise ValueError(f"checkpoint is missing experts for {key}")
        yield key, mx.stack([tensors[names[index]] for index in sorted(names)])


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
) -> tuple[Qwen35Model, Qwen35Config, dict]:
    """Load packed MLX tensors directly, or optionally quantize BF16 sources.

    ``bits`` and ``group_size`` apply only to unquantized sources. Packed
    checkpoints retain the precision and layout declared in their metadata.
    """

    config, raw_config = load_config(model_path)
    quantization = checkpoint_quantization(raw_config)
    model = Qwen35Model(config, norm_weights_are_offset=quantization is None)
    weights = dict(
        checkpoint_weights(model_path, config, prequantized=quantization is not None)
    )
    if quantization is not None:
        _prepare_packed_modules(model, weights, quantization)
    if quantization is not None:
        # Preserve uint32 payloads, affine scales/biases, and the sanitized
        # MLX normalization weights exactly. No source quantization is run.
        model.load_weights(list(weights.items()))
    else:
        # The DeltaNet decay parameters keep their checkpoint precision.
        model.load_weights(
            [
                (
                    name,
                    value
                    if name.endswith(_DECAY_PARAMETERS)
                    else value.astype(mx.bfloat16),
                )
                for name, value in weights.items()
            ]
        )
    del weights
    if quantization is None and bits is not None:
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
    mx.eval(model.norm.parameters())
    if not config.tie_word_embeddings:
        mx.eval(model.lm_head.parameters())
    model.prepare_for_inference()
    return model, config, raw_config


def _prepare_packed_modules(model: Qwen35Model, weights: dict, quantization: dict):
    """Install packed module types before assigning checkpoint arrays."""
    packed = {
        name.removesuffix(".scales") for name in weights if name.endswith(".scales")
    }
    if not packed:
        raise ValueError("quantized MLX checkpoint contains no affine scales")
    for path in packed:
        weight, scales, biases = (
            weights.get(f"{path}.{suffix}") for suffix in ("weight", "scales", "biases")
        )
        if weight is None or scales is None or biases is None:
            raise ValueError(f"incomplete affine MLX parameters for {path}")
        if weight.dtype != mx.uint32:
            raise ValueError(f"packed MLX weight must have uint32 dtype: {path}")
        if not all(
            mx.issubdtype(value.dtype, mx.floating) for value in (scales, biases)
        ):
            raise ValueError(
                f"affine MLX scales and biases must be floating point: {path}"
            )
    for name, value in weights.items():
        if value.dtype == mx.uint32 and name.removesuffix(".weight") not in packed:
            raise ValueError(f"packed MLX weight is missing affine scales: {name}")

    def predicate(path: str, module: nn.Module) -> bool:
        if path not in packed:
            return False
        if not _quantizable(quantization["group_size"])(path, module):
            raise ValueError(f"unsupported packed MLX module: {path}")
        return True

    # This converts the unevaluated random architecture, before loading any
    # checkpoint weights. Those placeholder arrays are replaced below without
    # evaluation; the checkpoint's packed payloads are never requantized.
    nn.quantize(model, **quantization, class_predicate=predicate)


__all__ = [
    "checkpoint_quantization",
    "checkpoint_weights",
    "load_config",
    "load_model",
]
