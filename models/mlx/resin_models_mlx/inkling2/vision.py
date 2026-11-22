# Copyright 2026 SGLang contributors. Licensed under the Apache License, Version 2.0.
# Adapted from SGLang e62ac6279646c686d3b328f58d9b5c2c2d5927dc.
"""Inkling 2 vision transformer over the recipe's normalized RGB patches."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from resin.models.inkling2.config import validate_vision_config


def _resize_rows(grid: mx.array, length: int) -> mx.array:
    """Linear interpolation along the first axis using half-pixel centers."""
    size = grid.shape[0]
    if size == length:
        return grid
    coordinate = (mx.arange(length, dtype=mx.float32) + 0.5) * size / length - 0.5
    coordinate = mx.clip(coordinate, 0, size - 1)
    lower = mx.floor(coordinate).astype(mx.int32)
    upper = mx.minimum(lower + 1, size - 1)
    fraction = (coordinate - lower)[:, None, None]
    return (1 - fraction) * grid[lower] + fraction * grid[upper]


def _rotary_angles(height: int, width: int, dim: int) -> tuple[mx.array, mx.array]:
    # Native patches use one-based x/y; the leading class token uses (0, 0).
    indices = mx.arange(height * width)
    xy = mx.stack([indices % width + 1, indices // width + 1], axis=-1)
    xy = mx.concatenate([mx.zeros((1, 2), mx.int32), xy]).astype(mx.float32)
    inverse = 10000.0 ** (-np.arange(0, dim, 4, dtype=np.float32) / dim)
    angles = (xy[..., None] * mx.array(inverse)).reshape(-1, 1, dim // 2)
    return mx.cos(angles), mx.sin(angles)


def _rotate(value: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    paired = value.astype(mx.float32)
    even, odd = paired[..., ::2], paired[..., 1::2]
    rotated = mx.stack([even * cos - odd * sin, odd * cos + even * sin], axis=-1)
    return rotated.reshape(value.shape).astype(value.dtype)


def _gelu(value: mx.array) -> mx.array:
    return nn.gelu(value.astype(mx.float32)).astype(value.dtype)


class _Attention(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.heads = heads
        self.q_proj = nn.Linear(width, width)
        self.k_proj = nn.Linear(width, width)
        self.v_proj = nn.Linear(width, width)
        self.out_proj = nn.Linear(width, width)

    def __call__(self, hidden, cos, sin):
        tokens, width = hidden.shape
        dim = width // self.heads
        shape = (tokens, self.heads, dim)
        q = _rotate(self.q_proj(hidden).reshape(shape), cos, sin)
        k = _rotate(self.k_proj(hidden).reshape(shape), cos, sin)
        v = self.v_proj(hidden).reshape(shape)
        attended = mx.fast.scaled_dot_product_attention(
            q.transpose(1, 0, 2)[None],
            k.transpose(1, 0, 2)[None],
            v.transpose(1, 0, 2)[None],
            scale=dim**-0.5,
        )
        return self.out_proj(attended[0].transpose(1, 0, 2).reshape(tokens, width))


class _MLP(nn.Module):
    def __init__(self, width: int, intermediate: int):
        super().__init__()
        self.c_fc = nn.Linear(width, intermediate)
        self.c_proj = nn.Linear(intermediate, width)

    def __call__(self, hidden):
        return self.c_proj(_gelu(self.c_fc(hidden)))


class _LayerScale(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.gamma = mx.ones((width,))

    def __call__(self, hidden):
        return hidden * self.gamma


class _Layer(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        width, eps = config["hidden_size"], config["layer_norm_eps"]
        self.ln_1 = nn.LayerNorm(width, eps=eps)
        self.self_attn = _Attention(width, config["num_attention_heads"])
        self.ln_2 = nn.LayerNorm(width, eps=eps)
        self.mlp = _MLP(width, int(width * config["mlp_ratio"]))
        self.ls_1 = _LayerScale(width)
        self.ls_2 = _LayerScale(width)

    def __call__(self, hidden, cos, sin):
        hidden = hidden + self.ls_1(self.self_attn(self.ln_1(hidden), cos, sin))
        return hidden + self.ls_2(self.mlp(self.ln_2(hidden)))


class _Embeddings(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.grid_size = config["posemb_grid_size"]
        width = config["hidden_size"]
        self.patch_embedding = nn.Linear(
            3 * config["native_patch_size"] ** 2, width, bias=False
        )
        self.position_embedding = mx.zeros((self.grid_size**2, width))
        self.class_embedding = mx.zeros((width,))
        self.cls_position_embedding = mx.zeros((width,))

    def __call__(self, patches, height, width):
        grid = self.position_embedding.astype(mx.float32).reshape(
            self.grid_size, self.grid_size, -1
        )
        # Interpolate across x first and then y, keeping both blends in FP32.
        across = _resize_rows(grid.transpose(1, 0, 2), width).transpose(1, 0, 2)
        positions = _resize_rows(across, height).reshape(height * width, -1)
        hidden = self.patch_embedding(patches)
        cls = self.class_embedding + self.cls_position_embedding
        return mx.concatenate([cls[None], hidden + positions.astype(hidden.dtype)])


class _Backbone(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.embeddings = _Embeddings(config)
        self.ln_pre = nn.LayerNorm(config["hidden_size"], eps=config["layer_norm_eps"])
        self.layers = [_Layer(config) for _ in range(config["num_hidden_layers"])]
        self.head_dim = config["hidden_size"] // config["num_attention_heads"]

    def __call__(self, patches, height, width):
        hidden = self.ln_pre(self.embeddings(patches, height, width))
        cos, sin = _rotary_angles(height, width, self.head_dim)
        for layer in self.layers:
            hidden = layer(hidden, cos, sin)
        return hidden[1:]


class Inkling2VisionEncoder(nn.Module):
    """Encode each image independently, returning one BF16 vector per outer patch."""

    def __init__(self, vision_config: dict, text_hidden_size: int):
        super().__init__()
        validate_vision_config(vision_config, text_hidden_size)
        self.patch_size = vision_config["native_patch_size"]
        self.merge_size = vision_config["adapter_merge_size"]
        self.backbone = _Backbone(vision_config)
        projection = vision_config["projection_hidden_dim"]
        bias = vision_config["projector_bias"]
        self.proj_in = nn.Linear(
            self.merge_size**2 * vision_config["hidden_size"], projection, bias=bias
        )
        self.proj_out = nn.Linear(projection, text_hidden_size, bias=bias)
        self.set_dtype(mx.bfloat16)

    def _encode_image(self, pixels: mx.array, height: int, width: int) -> mx.array:
        m, p = self.merge_size, self.patch_size
        # Split outer patches into a row-major grid of native HWC patches.
        native = pixels.reshape(height, width, m, p, m, p, 3)
        native = native.transpose(0, 2, 1, 4, 3, 5, 6).reshape(-1, p * p * 3)
        hidden = self.backbone(native, height * m, width * m)
        # The projector consumes the m x m native vectors of each outer patch.
        merged = hidden.reshape(height, m, width, m, -1).transpose(0, 2, 1, 3, 4)
        return self.proj_out(_gelu(self.proj_in(merged.reshape(height * width, -1))))

    def __call__(self, pixel_values: mx.array, image_grid_thw: np.ndarray) -> mx.array:
        side = self.patch_size * self.merge_size
        if pixel_values.dtype != mx.bfloat16 or pixel_values.shape[1:] != (
            side,
            side,
            3,
        ):
            raise ValueError(
                f"Inkling 2 expects BF16 image patches of shape [N, {side}, {side}, 3]"
            )
        grid = np.asarray(image_grid_thw)
        if (
            grid.ndim != 2
            or grid.shape[1] != 3
            or grid.dtype not in (np.int32, np.int64)
        ):
            raise ValueError("Inkling 2 image grid must have integer shape [images, 3]")
        sizes = grid.tolist()
        if any(t != 1 or h <= 0 or w <= 0 for t, h, w in sizes):
            raise ValueError(
                "Inkling 2 image grids must have one frame and positive dimensions"
            )
        if sum(h * w for _, h, w in sizes) != pixel_values.shape[0]:
            raise ValueError("Inkling 2 image grids do not match the number of patches")
        outputs, start = [], 0
        for _, height, width in sizes:
            end = start + height * width
            outputs.append(self._encode_image(pixel_values[start:end], height, width))
            start = end
        return (
            mx.concatenate(outputs)
            if outputs
            else mx.zeros((0, self.proj_out.weight.shape[0]), mx.bfloat16)
        )
