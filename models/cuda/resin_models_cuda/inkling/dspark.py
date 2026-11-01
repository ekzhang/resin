"""Inkling-Small six-layer DSpark semi-autoregressive drafter."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from functools import lru_cache
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from resin.backend.layers import RMSNorm
from resin.models.inkling.config import (
    InklingConfig,
    InklingDSparkConfig,
    YarnConfig,
)
from resin_models_cuda.inkling.attention import (
    PagedAttentionBatch,
    append_kv,
    paged_attention,
    prepare_paged_attention_batch,
)
from resin_models_cuda.inkling.cache import DRAFT_GROUP, DRAFT_LAYERS, DeviceCache
from resin_models_cuda.inkling.model import InklingTarget


@lru_cache(8)
def precompute_yarn_rope(
    head_dim: int, max_sequence_length: int, rope: YarnConfig
) -> tuple[Tensor, Tensor]:
    frequencies = 1.0 / (
        rope.theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    if rope.factor > 1:

        def correction(rotations: float) -> float:
            return (
                head_dim
                * math.log(rope.original_context_length / (rotations * 2 * math.pi))
                / (2 * math.log(rope.theta))
            )

        low = max(math.floor(correction(rope.beta_fast)), 0)
        high = min(math.ceil(correction(rope.beta_slow)), head_dim - 1)
        if low == high:
            high += 0.001
        ramp = (
            (torch.arange(head_dim // 2, dtype=torch.float32) - low) / (high - low)
        ).clamp(0, 1)
        smooth = 1 - ramp
        frequencies = frequencies / rope.factor * (1 - smooth) + frequencies * smooth
    angles = torch.outer(
        torch.arange(max_sequence_length, dtype=torch.float32), frequencies
    )
    return angles.cos() * rope.attention_factor, angles.sin() * rope.attention_factor


def apply_neox_rope(
    hidden: Tensor, positions: Tensor, cos: Tensor, sin: Tensor
) -> Tensor:
    half = hidden.shape[-1] // 2
    selected_cos = cos[positions].unsqueeze(2)
    selected_sin = sin[positions].unsqueeze(2)
    left, right = hidden.float().split(half, dim=-1)
    return torch.cat(
        (
            left * selected_cos - right * selected_sin,
            right * selected_cos + left * selected_sin,
        ),
        dim=-1,
    ).to(hidden.dtype)


class DSparkAttention(nn.Module):
    cos: Tensor
    sin: Tensor

    def __init__(
        self,
        config: InklingDSparkConfig,
        layer_id: int,
        cos: Tensor,
        sin: Tensor,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        hidden = config.hidden_size
        self.q_proj = nn.Linear(
            hidden, self.heads * self.head_dim, bias=False, dtype=torch.bfloat16
        )
        self.k_proj = nn.Linear(
            hidden, self.kv_heads * self.head_dim, bias=False, dtype=torch.bfloat16
        )
        self.v_proj = nn.Linear(
            hidden, self.kv_heads * self.head_dim, bias=False, dtype=torch.bfloat16
        )
        self.o_proj = nn.Linear(
            self.heads * self.head_dim, hidden, bias=False, dtype=torch.bfloat16
        )
        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def project_kv(self, hidden: Tensor, positions: Tensor) -> tuple[Tensor, Tensor]:
        batch, tokens, _ = hidden.shape
        keys = self.k_proj(hidden).view(batch, tokens, self.kv_heads, self.head_dim)
        values = self.v_proj(hidden).view(batch, tokens, self.kv_heads, self.head_dim)
        keys = apply_neox_rope(self.k_norm(keys), positions, self.cos, self.sin)
        return keys, values

    def inject(self, hidden: Tensor, positions: Tensor, cache: DeviceCache) -> None:
        keys, values = self.project_kv(hidden, positions)
        append_kv(
            cache,
            group_index=DRAFT_GROUP,
            layer=self.layer_id,
            layer_count=DRAFT_LAYERS,
            positions=positions,
            keys=keys,
            values=values,
        )

    def forward(
        self,
        hidden: Tensor,
        positions: Tensor,
        cache: DeviceCache,
        sequence_lengths: Tensor,
        max_sequence_length: int,
        attention_batch: PagedAttentionBatch,
    ) -> Tensor:
        batch, tokens, _ = hidden.shape
        queries = self.q_proj(hidden).view(batch, tokens, self.heads, self.head_dim)
        queries = apply_neox_rope(self.q_norm(queries), positions, self.cos, self.sin)
        keys, values = self.project_kv(hidden, positions)
        append_kv(
            cache,
            group_index=DRAFT_GROUP,
            layer=self.layer_id,
            layer_count=DRAFT_LAYERS,
            positions=positions,
            keys=keys,
            values=values,
        )
        output = paged_attention(
            queries,
            cache=cache,
            group_index=DRAFT_GROUP,
            layer=self.layer_id,
            layer_count=DRAFT_LAYERS,
            positions=positions,
            sequence_lengths=sequence_lengths,
            max_sequence_length=max_sequence_length,
            softmax_scale=self.head_dim**-0.5,
            prepared_batch=attention_batch,
        )
        return self.o_proj(output.reshape(batch, tokens, -1))


class DSparkMlp(nn.Module):
    def __init__(self, config: InklingDSparkConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            dtype=torch.bfloat16,
        )
        self.up = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            dtype=torch.bfloat16,
        )
        self.down = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            dtype=torch.bfloat16,
        )

    def forward(self, hidden: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(hidden)) * self.up(hidden))


class DSparkLayer(nn.Module):
    def __init__(
        self,
        config: InklingDSparkConfig,
        layer_id: int,
        cos: Tensor,
        sin: Tensor,
    ) -> None:
        super().__init__()
        self.input_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attn = DSparkAttention(config, layer_id, cos, sin)
        self.post_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = DSparkMlp(config)

    def forward(
        self,
        hidden: Tensor,
        positions: Tensor,
        cache: DeviceCache,
        sequence_lengths: Tensor,
        max_sequence_length: int,
        attention_batch: PagedAttentionBatch,
    ) -> Tensor:
        hidden = hidden + self.attn(
            self.input_norm(hidden),
            positions,
            cache,
            sequence_lengths,
            max_sequence_length,
            attention_batch,
        )
        return hidden + self.mlp(self.post_norm(hidden))


class DSparkDraft(nn.Module):
    """External gamma-five drafter sharing the target embedding and head."""

    gamma = 5

    def __init__(
        self,
        config: InklingDSparkConfig,
        target_config: InklingConfig,
        target: InklingTarget,
        max_sequence_length: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.target_config = target_config
        self.target = target
        cos, sin = precompute_yarn_rope(
            config.head_dim, max_sequence_length, config.yarn
        )
        self.layers = nn.ModuleList(
            DSparkLayer(config, index, cos, sin)
            for index in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.fc = nn.Linear(
            len(config.target_layer_ids) * config.hidden_size,
            config.hidden_size,
            bias=False,
            dtype=torch.bfloat16,
        )
        self.hidden_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.markov_w1 = nn.Embedding(
            config.vocab_size, config.markov_rank, dtype=torch.bfloat16
        )
        self.markov_w2 = nn.Linear(
            config.markov_rank,
            config.vocab_size,
            bias=False,
            dtype=torch.bfloat16,
        )
        self.confidence = nn.Linear(
            config.hidden_size + config.markov_rank,
            1,
            bias=True,
            dtype=torch.float32,
        )

    def project_target_hidden(self, taps: Tensor) -> Tensor:
        return self.hidden_norm(self.fc(taps))

    def inject_target(
        self, taps: Tensor, positions: Tensor, cache: DeviceCache
    ) -> None:
        projected = self.project_target_hidden(taps)
        for layer in cast(Sequence[DSparkLayer], self.layers):
            layer.attn.inject(projected, positions, cache)

    def forward(
        self,
        input_ids: Tensor,
        positions: Tensor,
        cache: DeviceCache,
        sequence_lengths: Tensor,
        max_sequence_length: int,
    ) -> tuple[Tensor, Tensor]:
        hidden = self.target.embed_tokens(input_ids)
        attention_batch = prepare_paged_attention_batch(
            cache,
            sequence_lengths,
            max_sequence_length,
            batch=input_ids.shape[0],
            tokens=input_ids.shape[1],
            layer_counts={DRAFT_GROUP: DRAFT_LAYERS},
        )
        for layer in self.layers:
            hidden = layer(
                hidden,
                positions,
                cache,
                sequence_lengths,
                max_sequence_length,
                attention_batch,
            )
        hidden = self.norm(hidden)
        base_logits = self.target.unembed(
            hidden / self.target_config.logits_mup_width_multiplier
        )[..., : self.target_config.unpadded_vocab_size]
        return base_logits, hidden

    def propose(
        self,
        anchor_ids: Tensor,
        positions: Tensor,
        cache: DeviceCache,
        max_sequence_length: int,
        sampler: Callable[[Tensor, int], Tensor],
    ) -> tuple[Tensor, Tensor]:
        batch = anchor_ids.shape[0]
        input_ids = torch.full(
            (batch, self.gamma),
            self.config.mask_token_id,
            dtype=torch.long,
            device=anchor_ids.device,
        )
        input_ids[:, 0] = anchor_ids
        proposal_positions = positions[:, None] + torch.arange(
            self.gamma, device=positions.device
        )
        base_logits, hidden = self(
            input_ids,
            proposal_positions,
            cache,
            positions + self.gamma,
            max_sequence_length,
        )
        previous = anchor_ids
        drafts = []
        corrected = []
        for step in range(self.gamma):
            logits = (
                base_logits[:, step]
                + self.markov_w2(self.markov_w1(previous))[
                    ..., : self.target_config.unpadded_vocab_size
                ]
            )
            previous = sampler(logits, step)
            drafts.append(previous)
            corrected.append(logits)
        return torch.stack(drafts, dim=1), torch.stack(corrected, dim=1)


__all__ = [
    "DSparkAttention",
    "DSparkDraft",
    "DSparkLayer",
    "DSparkMlp",
    "apply_neox_rope",
    "precompute_yarn_rope",
]
