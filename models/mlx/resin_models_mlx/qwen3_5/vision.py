"""Qwen3.5 MLX-VLM vision tower over the native recipe's patch rows."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_vlm.models.qwen3_5.config import VisionConfig
from mlx_vlm.models.qwen3_5.vision import VisionModel

from resin.backend.loading import read_config
from resin_models_mlx.qwen3_5.checkpoint import _Checkpoint


class _Config(VisionConfig):
    @property
    def patch_width(self) -> int:
        return self.in_channels * self.temporal_patch_size * self.patch_size**2


class Qwen35VisionModel(VisionModel):
    """Reuse MLX-VLM weights/blocks while encoding each frame independently.

    Its geometry helpers run before compilation. Only the pure per-frame tower
    is compiled; upstream attention receives constant empty frame split points.
    """

    config: _Config

    def __init__(self, config: Mapping[str, Any], *, compiled: bool = False):
        values = dict(config.get("vision_config", config))
        values.setdefault("patch_size", 16)
        if values.get("deepstack_visual_indexes"):
            raise ValueError("Qwen3.5 does not support DeepStack")
        super().__init__(_Config.from_dict(values))
        activation = values.get("hidden_act", "gelu_pytorch_tanh")
        if activation not in ("gelu", "gelu_pytorch_tanh"):
            raise ValueError(f"unsupported Qwen3.5 vision activation: {activation}")
        # MLX-VLM fixes tanh GELU; retain HF's explicit exact-GELU configuration.
        for block in self.blocks:
            block.mlp.act_fn = nn.GELU(
                approx="tanh" if activation == "gelu_pytorch_tanh" else "none"
            )
        self.enable_compile(compiled)

    def enable_compile(self, enabled: bool = True):
        self._encode = mx.compile(self._encode_frame) if enabled else self._encode_frame

    def _encode_frame(self, pixels, positions, rotary):
        hidden = self.patch_embed(pixels.astype(self.patch_embed.proj.weight.dtype))
        hidden = hidden + positions.astype(hidden.dtype)
        for block in self.blocks:
            hidden = block(
                hidden, cu_seqlens=np.zeros(2, dtype=np.int32), rotary_pos_emb=rotary
            )
        return self.merger(hidden)

    def __call__(self, pixel_values: Any, grid_thw: Any) -> mx.array:
        grid = np.asarray(grid_thw)
        if (
            grid.ndim != 2
            or grid.shape[1] != 3
            or not len(grid)
            or not np.issubdtype(grid.dtype, np.integer)
            or (grid <= 0).any()
            or (grid[:, 1:] % self.config.spatial_merge_size).any()
        ):
            raise ValueError(
                "Qwen3.5 vision grid must contain positive [frames,height,width] rows divisible by merge size"
            )
        expected = (int(np.prod(grid, axis=1).sum()), self.config.patch_width)
        if tuple(pixel_values.shape) != expected:
            raise ValueError(f"Qwen3.5 vision patches must have shape {expected}")
        pixels = (
            pixel_values
            if isinstance(pixel_values, mx.array)
            else mx.array(pixel_values)
        )
        outputs, start = [], 0
        for frames, height, width in grid.tolist():
            frame_grid = mx.array([[1, height, width]], mx.int32)
            positions = self.fast_pos_embed_interpolate(frame_grid)
            rotary = self.rot_pos_emb(frame_grid)
            for _ in range(frames):
                end = start + height * width
                outputs.append(self._encode(pixels[start:end], positions, rotary))
                start = end
        return mx.concatenate(outputs)


_PREFIXES = ("model.visual.", "vision_tower.", "visual.")


def sanitize_vision_weights(
    weights: Mapping[str, mx.array], *, mlx_format: bool = False
) -> dict[str, mx.array]:
    """Translate HF/MLX names and the HF Conv3d layout into MLX-VLM's tower."""
    result = {}
    for name, value in weights.items():
        prefix = next((p for p in _PREFIXES if name.startswith(p)), None)
        if prefix is not None:
            name = name[len(prefix) :]
        elif not name.startswith(("patch_embed.", "pos_embed.", "blocks.", "merger.")):
            continue
        if name == "patch_embed.proj.weight" and not mlx_format:
            value = value.transpose(0, 2, 3, 4, 1)
        result[name] = value
    return result


def load_vision_model(
    model_path: str | Path,
    raw_config: Mapping[str, Any] | None = None,
    *,
    compiled: bool = False,
) -> Qwen35VisionModel:
    path = Path(model_path)
    config = read_config(path) if raw_config is None else raw_config
    if not config.get("vision_config"):
        raise ValueError("Qwen3.5 checkpoint does not contain a vision configuration")
    tensors = _Checkpoint(path)
    selected = {
        name: tensors[name] for name in tensors.files if name.startswith(_PREFIXES)
    }
    if any(name.endswith((".scales", ".biases")) for name in selected):
        raise ValueError("packed quantized Qwen3.5 vision weights are not supported")
    weights = sanitize_vision_weights(
        selected, mlx_format=any(name.startswith("vision_tower.") for name in selected)
    )
    if any(not mx.issubdtype(value.dtype, mx.floating) for value in weights.values()):
        raise ValueError("Qwen3.5 vision weights must have a floating-point dtype")
    model = Qwen35VisionModel(config, compiled=compiled)
    model.load_weights(list(weights.items()))
    model.eval()
    mx.eval(model.parameters())
    return model


__all__ = ["Qwen35VisionModel", "load_vision_model", "sanitize_vision_weights"]
