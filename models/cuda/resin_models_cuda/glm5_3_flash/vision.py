"""GLM image encoder and placement of features into chunked language prefill.

Adapted from Hugging Face Transformers (Apache-2.0), revision
50e35cc0ca18ec19b80653690de553e6decf74f5, models/glm5_next/modeling_glm5_next.py.
The tower retains the checkpoint's BF16 weights; NVFP4 applies to the text MoE.
"""

from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, fields
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel


@dataclass(frozen=True)
class GlmVisionConfig:
    hidden_size: int = 1024
    intermediate_size: int = 4096
    depth: int = 24
    num_heads: int = 16
    out_hidden_size: int = 4096
    projection_intermediate_size: int = 10240
    patch_size: int = 14
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2
    in_channels: int = 3
    attention_bias: bool = True
    rms_norm_eps: float = 1e-5
    swiglu_limit: float = 10.0
    rope_theta: float = 10000.0

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "GlmVisionConfig":
        if raw.get("hidden_act", "silu") != "silu":
            raise ValueError("GLM vision requires silu activation")
        rope = raw.get("rope_parameters") or {}
        if rope.get("rope_type", "axial") != "axial":
            raise ValueError("GLM vision requires axial RoPE")
        values = {f.name: raw[f.name] for f in fields(cls) if f.name in raw}
        values["rope_theta"] = rope.get("rope_theta", 10000.0)
        return cls(**values)

    def __post_init__(self):
        if any(
            getattr(self, name) <= 0
            for name in (
                "hidden_size",
                "intermediate_size",
                "depth",
                "num_heads",
                "out_hidden_size",
                "projection_intermediate_size",
                "patch_size",
                "temporal_patch_size",
                "spatial_merge_size",
                "in_channels",
            )
        ) or self.hidden_size % (4 * self.num_heads):
            raise ValueError("invalid GLM vision dimensions or axial head size")


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = x.float() * torch.rsqrt(
            x.float().square().mean(-1, keepdim=True) + self.eps
        )
        return self.weight * normalized.to(x.dtype)


def vision_rope(
    height: int, width: int, config: GlmVisionConfig, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Axial RoPE in processor merge-block-major patch order."""
    merge = config.spatial_merge_size
    y, x = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    positions = torch.stack((y, x), -1).reshape(
        height // merge, merge, width // merge, merge, 2
    )
    positions = positions.permute(0, 2, 1, 3, 4).reshape(-1, 2)
    dim = config.hidden_size // config.num_heads // 2
    inv = config.rope_theta ** (
        -torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim
    )
    angles = (positions.float().unsqueeze(-1) * inv).flatten(1)
    angles = torch.cat((angles, angles), -1).unsqueeze(1)
    return angles.cos(), angles.sin()


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    a, b = x.float().chunk(2, -1)
    return (x.float() * cos + torch.cat((-b, a), -1) * sin).to(x.dtype)


class VisionMLP(nn.Module):
    def __init__(self, dim: int, intermediate: int, limit: float, bias: bool):
        super().__init__()
        self.gate_proj = nn.Linear(dim, intermediate, bias=bias)
        self.up_proj = nn.Linear(dim, intermediate, bias=bias)
        self.down_proj = nn.Linear(intermediate, dim, bias=bias)
        self.limit = limit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x).clamp(max=self.limit)
        up = self.up_proj(x).clamp(min=-self.limit, max=self.limit)
        return self.down_proj(F.silu(gate) * up)


class VisionAttention(nn.Module):
    def __init__(self, c: GlmVisionConfig):
        super().__init__()
        self.heads = c.num_heads
        self.qkv = nn.Linear(c.hidden_size, 3 * c.hidden_size, bias=c.attention_bias)
        self.proj = nn.Linear(c.hidden_size, c.hidden_size, bias=c.attention_bias)
        self.q_norm = RMSNorm(c.hidden_size // c.num_heads, c.rms_norm_eps)
        self.k_norm = RMSNorm(c.hidden_size // c.num_heads, c.rms_norm_eps)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        q, k, v = self.qkv(x).reshape(x.shape[0], 3, self.heads, -1).unbind(1)
        q = apply_rotary(self.q_norm(q), cos, sin)
        k = apply_rotary(self.k_norm(k), cos, sin)
        # Keep the image batch axis explicit so CUDA can select fused SDPA.
        # Avoid cuDNN's first-use plan build for each new image geometry.
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION) if x.is_cuda else nullcontext():
            out = F.scaled_dot_product_attention(
                q.transpose(0, 1).unsqueeze(0),
                k.transpose(0, 1).unsqueeze(0),
                v.transpose(0, 1).unsqueeze(0),
            )
        return self.proj(out.squeeze(0).transpose(0, 1).reshape(x.shape[0], -1))


class VisionBlock(nn.Module):
    def __init__(self, c: GlmVisionConfig):
        super().__init__()
        self.norm1 = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.norm2 = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.attn = VisionAttention(c)
        self.mlp = VisionMLP(
            c.hidden_size, c.intermediate_size, c.swiglu_limit, c.attention_bias
        )

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.mlp(self.norm2(x))


class PatchEmbed(nn.Module):
    def __init__(self, c: GlmVisionConfig):
        super().__init__()
        self.shape = (c.in_channels, c.temporal_patch_size, c.patch_size, c.patch_size)
        self.proj = nn.Conv3d(
            c.in_channels,
            c.hidden_size,
            kernel_size=self.shape[1:],
            stride=self.shape[1:],
        )

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.proj(
            pixels.reshape(-1, *self.shape).to(self.proj.weight.dtype)
        ).flatten(1)


class PatchMerger(VisionMLP):
    def __init__(self, c: GlmVisionConfig):
        super().__init__(
            c.out_hidden_size, c.projection_intermediate_size, c.swiglu_limit, False
        )
        self.proj = nn.Linear(c.out_hidden_size, c.out_hidden_size, bias=False)
        # The checkpoint uses LayerNorm with its default 1e-5, not RMSNorm.
        self.post_projection_norm = nn.LayerNorm(c.out_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(F.gelu(self.post_projection_norm(self.proj(x))))


class GlmVisionModel(nn.Module):
    """Prefill-only image encoder; no context/sampling-specialized compilation.

    ``weights`` use names stripped of ``model.visual.``. Image geometry is host
    metadata, so no GPU readbacks occur in forward. Each image attends only to
    its own patches. Forward is CUDA-graphable for fixed image geometry.
    """

    def __init__(
        self, config: GlmVisionConfig, weights: Mapping[str, torch.Tensor] | None = None
    ):
        super().__init__()
        self.config = config
        with torch.device("meta" if weights is not None else "cpu"):
            self.patch_embed = PatchEmbed(config)
            self.blocks = nn.ModuleList(
                VisionBlock(config) for _ in range(config.depth)
            )
            self.post_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
            self.downsample = nn.Conv2d(
                config.hidden_size,
                config.out_hidden_size,
                config.spatial_merge_size,
                stride=config.spatial_merge_size,
            )
            self.merger = PatchMerger(config)
        if weights is not None:
            self.load_state_dict(dict(weights), strict=True, assign=True)
        self.eval().requires_grad_(False)

    def forward(
        self, pixel_values: torch.Tensor, grid_thw: Sequence[Sequence[int]]
    ) -> torch.Tensor:
        c, merge = self.config, self.config.spatial_merge_size
        if not grid_thw:
            raise ValueError("GLM vision requires at least one image")
        rows = 0
        for t, h, w in grid_thw:
            if t != 1 or h <= 0 or w <= 0 or h % merge or w % merge:
                raise ValueError(
                    "GLM image grids must be [1, H, W] with positive merge-aligned spatial dimensions"
                )
            rows += h * w
        if tuple(pixel_values.shape) != (
            rows,
            c.in_channels * c.temporal_patch_size * c.patch_size**2,
        ):
            raise ValueError(
                "GLM pixel values do not match image grids and patch configuration"
            )
        x = self.patch_embed(pixel_values)
        images = []
        offset = 0
        for _, h, w in grid_thw:
            image = x[offset : offset + h * w]
            cos, sin = vision_rope(h, w, c, x.device)
            for block in self.blocks:
                image = block(image, cos, sin)
            images.append(image)
            offset += h * w
        x = self.post_layernorm(torch.cat(images))
        x = x.reshape(-1, merge, merge, c.hidden_size).permute(0, 3, 1, 2)
        return self.merger(self.downsample(x).flatten(1))

    def encode(
        self, pixel_values: torch.Tensor, grid_thw: Sequence[Sequence[int]]
    ) -> torch.Tensor:
        return self(pixel_values, grid_thw)
