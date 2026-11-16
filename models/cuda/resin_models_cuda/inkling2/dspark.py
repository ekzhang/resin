"""Bundled Inkling2 DSpark: one parallel block, then seven recurrent samples.

Adapted from SGLang's Apache-2.0 Inkling2 MTP implementation at
e62ac6279646c686d3b328f58d9b5c2c2d5927dc (models/inkling2_mtp.py).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from resin.backend.cuda.parallel import TensorParallelContext
from resin.backend.layers import RMSNorm
from resin.models.inkling2.dspark import Inkling2DSparkConfig
from resin_models_cuda.inkling2.attention import (
    PagedAttentionBatch,
    paged_attention,
    prepare_paged_attention_batch,
)
from resin_models_cuda.inkling2.cache import (
    DRAFT_GLOBAL_GROUP,
    DRAFT_LOCAL_GROUP,
    DeviceCache,
)

if TYPE_CHECKING:
    from resin.models.inkling2.config import Inkling2Config
    from resin_models_cuda.inkling2.model import Inkling2Target


def mix_rmsnorm(targets: Tensor, alpha: Tensor, norm: RMSNorm) -> Tensor:
    """Accumulate the learned target mixture in FP32 before normalization."""
    if targets.is_cuda:
        from resin_models_cuda.inkling2.dspark_kernels import mix_rmsnorm as fused

        return fused(targets, alpha, norm.weight, norm.eps)
    mixed = (targets.float() * alpha.float()[None, :, None]).sum(dim=1)
    return F.rms_norm(mixed, (mixed.shape[-1],), norm.weight.float(), norm.eps).to(
        targets.dtype
    )


def apply_neox_rope(hidden: Tensor, positions: Tensor, rope: Tensor) -> Tensor:
    cos, sin = rope[positions].unsqueeze(-2).chunk(2, dim=-1)
    left, right = hidden.float().chunk(2, dim=-1)
    return torch.cat((left * cos - right * sin, right * cos + left * sin), -1).to(
        hidden.dtype
    )


def norm_neox_rope(
    hidden: Tensor, positions: Tensor, rope: Tensor, norm: RMSNorm
) -> Tensor:
    if hidden.is_cuda:
        from resin_models_cuda.inkling2.dspark_kernels import norm_neox_rope as fused

        return fused(hidden, positions, rope, norm.weight, norm.eps)
    return apply_neox_rope(norm(hidden), positions, rope)


class DSparkAttention(nn.Module):
    def __init__(
        self, config: Inkling2DSparkConfig, layer: int, parallel: TensorParallelContext
    ) -> None:
        super().__init__()
        self.parallel = parallel
        self.allreduce: Callable[[Tensor], Tensor] = parallel.sum
        self.heads = parallel.layout.even_shard(
            config.num_attention_heads, name="DSpark query heads"
        ).size
        self.kv_heads = parallel.layout.even_shard(
            config.num_key_value_heads, name="DSpark KV heads"
        ).size
        self.head_dim = config.head_dim
        self.q_width = self.heads * config.head_dim
        self.kv_width = self.kv_heads * config.head_dim
        local = layer in config.local_layer_ids
        group_layers = config.local_layer_ids if local else config.global_layer_ids
        self.group = DRAFT_LOCAL_GROUP if local else DRAFT_GLOBAL_GROUP
        self.layer, self.layer_count = group_layers.index(layer), len(group_layers)
        self.window = config.sliding_window_size if local else None
        self.qkv_proj = nn.Linear(
            config.hidden_size,
            self.q_width + 2 * self.kv_width,
            bias=False,
            dtype=torch.bfloat16,
        )
        self.kv_proj_ctx = nn.Linear(
            config.hidden_size, 2 * self.kv_width, bias=False, dtype=torch.bfloat16
        )
        self.o_proj = nn.Linear(
            self.q_width, config.hidden_size, bias=False, dtype=torch.bfloat16
        )
        self.q_norm = RMSNorm(config.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(config.head_dim, config.rms_norm_eps)
        self.k_norm_ctx = RMSNorm(config.head_dim, config.rms_norm_eps)

    def store(
        self, keys: Tensor, values: Tensor, positions: Tensor, cache: DeviceCache
    ) -> None:
        cache.store_kv(
            keys,
            values,
            positions,
            group_index=self.group,
            layer=self.layer,
            layer_count=self.layer_count,
        )

    def inject(
        self, context: Tensor, positions: Tensor, cache: DeviceCache, rope: Tensor
    ) -> None:
        keys, values = (
            self.kv_proj_ctx(context)
            .view(*positions.shape, 2 * self.kv_heads, self.head_dim)
            .chunk(2, dim=-2)
        )
        keys = norm_neox_rope(keys, positions, rope, self.k_norm_ctx)
        self.store(keys, values, positions, cache)

    def forward(
        self,
        hidden: Tensor,
        positions: Tensor,
        cache: DeviceCache,
        sequence_lengths: Tensor,
        max_sequence_length: int,
        attention_batch: PagedAttentionBatch,
        rope: Tensor,
    ) -> Tensor:
        q, k, v = self.qkv_proj(hidden).split(
            (self.q_width, self.kv_width, self.kv_width), dim=-1
        )
        q = q.view(*positions.shape, self.heads, self.head_dim)
        k = k.view(*positions.shape, self.kv_heads, self.head_dim)
        v = v.view_as(k)
        q = norm_neox_rope(q, positions, rope, self.q_norm)
        k = norm_neox_rope(k, positions, rope, self.k_norm)
        self.store(k, v, positions, cache)
        result = paged_attention(
            q,
            cache=cache,
            group_index=self.group,
            layer=self.layer,
            layer_count=self.layer_count,
            positions=positions,
            sequence_lengths=sequence_lengths,
            max_sequence_length=max_sequence_length,
            softmax_scale=self.head_dim**-0.5,
            prepared_batch=attention_batch,
            causal=False,
            window_left=self.window,
            window_anchor_at_start=self.window is not None,
        )
        assert isinstance(result, Tensor)
        return self.allreduce(self.o_proj(result.flatten(-2)))


class DSparkLayer(nn.Module):
    def __init__(
        self, config: Inkling2DSparkConfig, layer: int, parallel: TensorParallelContext
    ) -> None:
        super().__init__()
        self.parallel = parallel
        self.allreduce: Callable[[Tensor], Tensor] = parallel.sum
        self.dflare_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = DSparkAttention(config, layer, parallel)
        width = parallel.layout.even_shard(
            config.intermediate_size, name="DSpark MLP intermediate"
        ).size
        self.gate_up_proj = nn.Linear(
            config.hidden_size, 2 * width, bias=False, dtype=torch.bfloat16
        )
        self.down_proj = nn.Linear(
            width, config.hidden_size, bias=False, dtype=torch.bfloat16
        )

    def forward(self, hidden: Tensor, *attention_args) -> Tensor:
        hidden = hidden + self.self_attn(self.input_layernorm(hidden), *attention_args)
        gate, up = self.gate_up_proj(self.post_attention_layernorm(hidden)).chunk(2, -1)
        return hidden + self.allreduce(self.down_proj(F.silu(gate) * up))


class DSparkMarkovHead(nn.Module):
    fold_weight: Tensor
    fold_bias: Tensor

    def __init__(self, config: Inkling2DSparkConfig) -> None:
        super().__init__()
        rank, hidden = config.markov_rank, config.hidden_size
        self.markov_w1 = nn.Embedding(config.vocab_size, rank, dtype=torch.bfloat16)
        self.rnn_proj = nn.Linear(2 * rank + 2 * hidden, 3 * rank, dtype=torch.bfloat16)
        self.hidden_out = nn.Linear(rank, hidden, dtype=torch.bfloat16)
        self.dflare_norm = RMSNorm(hidden, config.rms_norm_eps)
        self.register_buffer(
            "fold_weight",
            torch.empty(config.vocab_size, rank, dtype=torch.bfloat16),
            persistent=False,
        )
        self.register_buffer(
            "fold_bias",
            torch.empty(config.vocab_size, dtype=torch.bfloat16),
            persistent=False,
        )

    @torch.no_grad()
    def prepare(self, head: Tensor) -> None:
        """Fold the target head once: every recurrent step projects rank 256."""
        weight, bias = self.hidden_out.weight.double(), self.hidden_out.bias.double()
        for start in range(0, head.shape[0], 4096):
            rows = slice(start, start + 4096)
            self.fold_weight[rows].copy_(head[rows].double() @ weight)
            self.fold_bias[rows].copy_(head[rows].double() @ bias)

    def step(
        self, state: Tensor, previous: Tensor, hidden: Tensor, f_rnn: Tensor
    ) -> tuple[Tensor, Tensor]:
        if hidden.is_cuda:
            from resin_models_cuda.inkling2.dspark_kernels import rnn_gates, rnn_inputs

            inputs = rnn_inputs(state, previous, self.markov_w1.weight, hidden, f_rnn)
            state, latent = rnn_gates(self.rnn_proj(inputs), state)
        else:
            inputs = torch.cat((state, self.markov_w1(previous), hidden, f_rnn), -1)
            gate, candidate, output = self.rnn_proj(inputs).chunk(3, -1)
            gate = gate.sigmoid()
            state = gate * state + (1 - gate) * candidate.tanh()
            latent = output.tanh()
        return F.linear(latent, self.fold_weight, self.fold_bias), state


class DSparkDraft(nn.Module):
    """Teacher-forced context KV and a fresh rank-256 RNN per proposal block."""

    gamma = 7
    target: Inkling2Target
    rope: Tensor
    fuse_alpha: Tensor

    def __init__(
        self,
        config: Inkling2DSparkConfig,
        target_config: Inkling2Config,
        target: Inkling2Target,
        max_sequence_length: int,
        tensor_parallel: TensorParallelContext | None = None,
    ) -> None:
        super().__init__()
        config.validate_target(target_config)
        self.config, self.target_config = config, target_config
        # The target owns these weights; do not register it twice in the module tree.
        object.__setattr__(self, "target", target)
        self.parallel = tensor_parallel or TensorParallelContext()
        self.gamma = config.gamma
        self.layers = nn.ModuleList(
            DSparkLayer(config, layer, self.parallel)
            for layer in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.fuse_weight = nn.Parameter(
            torch.empty(
                config.num_hidden_layers + 1,
                len(config.target_layer_ids),
                dtype=torch.bfloat16,
            ),
            requires_grad=False,
        )
        self.mask_embed = nn.Parameter(
            torch.empty(config.hidden_size, dtype=torch.bfloat16), requires_grad=False
        )
        self.markov_head = DSparkMarkovHead(config)
        self.register_buffer(
            "fuse_alpha", torch.empty_like(self.fuse_weight), persistent=False
        )
        self.register_buffer(
            "rope",
            torch.empty(max_sequence_length, config.head_dim, dtype=torch.float32),
            persistent=False,
        )

    @torch.no_grad()
    def prepare(self) -> None:
        """Initialize derived constants after loading and before graph capture."""
        self.fuse_alpha.copy_(self.fuse_weight.float().softmax(-1))
        frequencies = self.config.rope_theta ** (
            -torch.arange(0, self.config.head_dim, 2, device=self.rope.device).float()
            / self.config.head_dim
        )
        angles = torch.outer(
            torch.arange(self.rope.shape[0], device=self.rope.device).float(),
            frequencies,
        )
        self.rope.copy_(torch.cat((angles.cos(), angles.sin()), -1))
        self.markov_head.prepare(self.target.unembed.weight)

    def inject_target(
        self, taps: Tensor, positions: Tensor, cache: DeviceCache
    ) -> Tensor:
        targets = taps.reshape(
            -1, len(self.config.target_layer_ids), self.config.hidden_size
        )
        for index, layer in enumerate(cast(Sequence[DSparkLayer], self.layers)):
            context = mix_rmsnorm(targets, self.fuse_alpha[index], layer.dflare_norm)
            layer.self_attn.inject(
                context.view(*positions.shape, -1), positions, cache, self.rope
            )
        features = mix_rmsnorm(
            targets, self.fuse_alpha[-1], self.markov_head.dflare_norm
        )
        return features.view(*taps.shape[:-1], self.config.hidden_size)

    def forward(
        self,
        anchor_ids: Tensor,
        positions: Tensor,
        cache: DeviceCache,
        max_sequence_length: int,
    ) -> Tensor:
        hidden = self.mask_embed.expand(
            anchor_ids.shape[0], self.config.block_size, -1
        ).clone()
        hidden[:, 0] = self.target.embed_norm(self.target.embed_tokens(anchor_ids))
        lengths = positions[:, -1] + 1
        layer_counts = {
            group: count
            for group, count in (
                (DRAFT_GLOBAL_GROUP, self.config.global_layer_count),
                (DRAFT_LOCAL_GROUP, self.config.local_layer_count),
            )
            if count
        }
        attention_batch = prepare_paged_attention_batch(
            cache,
            lengths,
            max_sequence_length,
            batch=anchor_ids.shape[0],
            tokens=self.config.block_size,
            layer_counts=layer_counts,
        )
        for layer in self.layers:
            hidden = layer(
                hidden,
                positions,
                cache,
                lengths,
                max_sequence_length,
                attention_batch,
                self.rope,
            )
        # The anchor participates in attention but is never a sampled draft row.
        return self.norm(hidden[:, 1:].contiguous())

    def propose(
        self,
        anchor_ids: Tensor,
        positions: Tensor,
        cache: DeviceCache,
        max_sequence_length: int,
        sampler: Callable[[Tensor, int], Tensor],
        *,
        f_rnn: Tensor,
    ) -> Tensor:
        proposal_positions = positions[:, None] + torch.arange(
            self.config.block_size, device=positions.device
        )
        hidden = self(anchor_ids, proposal_positions, cache, max_sequence_length)
        vocab = self.target_config.unpadded_vocab_size
        base = self.target.unembed(hidden)[..., :vocab]
        state = hidden.new_zeros(anchor_ids.shape[0], self.config.markov_rank)
        previous, drafts = anchor_ids, []
        softcap = self.target_config.final_logit_softcapping
        for step in range(self.gamma):
            bias, state = self.markov_head.step(state, previous, hidden[:, step], f_rnn)
            if hidden.is_cuda:
                from resin_models_cuda.inkling2.dspark_kernels import corrected_logits

                logits = corrected_logits(base[:, step], bias[:, :vocab], softcap)
            else:
                logits = base[:, step] + bias[:, :vocab]
                logits = (torch.tanh(logits.float() / softcap) * softcap).to(
                    logits.dtype
                )
            previous = sampler(logits, step)
            drafts.append(previous)
        return torch.stack(drafts, 1)
