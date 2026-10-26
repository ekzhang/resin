# Portions are adapted from Hugging Face Transformers (Apache-2.0).

"""LFM2 modules and the three pieces of request-owned model state."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from resin_models_torch.lfm2.config import Lfm2Config


class FullAttentionState:
    """All keys and values produced by one full-attention layer."""

    def __init__(self) -> None:
        self.keys: Tensor | None = None
        self.values: Tensor | None = None

    @property
    def length(self) -> int:
        return 0 if self.keys is None else self.keys.shape[-2]

    def append(self, keys: Tensor, values: Tensor) -> tuple[Tensor, Tensor]:
        if self.keys is None:
            self.keys, self.values = keys, values
        else:
            assert self.values is not None
            self.keys = torch.cat((self.keys, keys), dim=-2)
            self.values = torch.cat((self.values, values), dim=-2)
        return self.keys, self.values


class RecurrentState:
    """The fixed-width history consumed by one short-convolution layer."""

    def __init__(self) -> None:
        self.value: Tensor | None = None


LayerState = FullAttentionState | RecurrentState


class Lfm2State:
    """One request's ordered full-attention and recurrent layer state."""

    def __init__(self, layer_types: tuple[str, ...]) -> None:
        self.layers: list[LayerState] = [
            FullAttentionState() if layer_type == "full_attention" else RecurrentState()
            for layer_type in layer_types
        ]
        self.length = 0


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, hidden: Tensor) -> Tensor:
        dtype = hidden.dtype
        normalized = hidden.float()
        normalized = normalized * torch.rsqrt(
            normalized.square().mean(-1, keepdim=True) + self.eps
        )
        return self.weight * normalized.to(dtype)


def _rotate_half(hidden: Tensor) -> Tensor:
    first, second = hidden.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class Attention(nn.Module):
    def __init__(self, config: Lfm2Config) -> None:
        super().__init__()
        width = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.kv_groups = self.num_heads // self.num_kv_heads
        self.head_dim = width // self.num_heads
        self.scale = self.head_dim**-0.5
        self.rope_theta = config.rope_theta
        self.q_layernorm = RMSNorm(self.head_dim, config.norm_eps)
        self.k_layernorm = RMSNorm(self.head_dim, config.norm_eps)
        self.q_proj = nn.Linear(width, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(width, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(width, self.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, width, bias=False)
        self.register_buffer(
            "inv_freq", self._frequencies(self.q_proj.weight.device), persistent=False
        )

    def _frequencies(self, device) -> Tensor:
        exponents = torch.arange(
            0, self.head_dim, 2, device=device, dtype=torch.float32
        )
        return 1.0 / self.rope_theta ** (exponents / self.head_dim)

    def materialize(self, device) -> None:
        self.inv_freq = self._frequencies(device)

    def _project(
        self, hidden: Tensor, positions: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch, length, _ = hidden.shape
        queries = self.q_proj(hidden).view(batch, length, self.num_heads, self.head_dim)
        keys = self.k_proj(hidden).view(batch, length, self.num_kv_heads, self.head_dim)
        values = self.v_proj(hidden).view(
            batch, length, self.num_kv_heads, self.head_dim
        )
        queries = self.q_layernorm(queries).transpose(1, 2)
        keys = self.k_layernorm(keys).transpose(1, 2)
        values = values.transpose(1, 2)

        angles = torch.einsum("bl,d->bld", positions.float(), self.inv_freq.float())
        angles = torch.cat((angles, angles), dim=-1)
        cos = angles.cos().to(queries.dtype).unsqueeze(1)
        sin = angles.sin().to(queries.dtype).unsqueeze(1)
        queries = queries * cos + _rotate_half(queries) * sin
        keys = keys * cos + _rotate_half(keys) * sin
        return queries, keys, values

    def forward(
        self, hidden: Tensor, positions: Tensor, state: FullAttentionState
    ) -> Tensor:
        batch, length, _ = hidden.shape
        queries, keys, values = self._project(hidden, positions)

        previous = state.length
        keys, values = state.append(keys, values)
        keys = keys.repeat_interleave(self.kv_groups, dim=1)
        values = values.repeat_interleave(self.kv_groups, dim=1)
        mask = None
        if length > 1:
            query_positions = torch.arange(
                previous, previous + length, device=hidden.device
            )
            key_positions = torch.arange(keys.shape[-2], device=hidden.device)
            mask = key_positions[None, :] <= query_positions[:, None]
        output = F.scaled_dot_product_attention(
            queries, keys, values, attn_mask=mask, dropout_p=0.0, scale=self.scale
        )
        output = output.transpose(1, 2).contiguous().view(batch, length, -1)
        return self.out_proj(output)

    def forward_batch(
        self,
        hidden: Tensor,
        positions: Tensor,
        states: Sequence[FullAttentionState],
    ) -> Tensor:
        """Decode one token for each independently cached sequence."""

        batch, length, _ = hidden.shape
        if length != 1 or len(states) != batch:
            raise ValueError("batched attention requires one state and token per row")
        queries, new_keys, new_values = self._project(hidden, positions)
        for index, state in enumerate(states):
            state.append(new_keys[index : index + 1], new_values[index : index + 1])

        max_length = max(state.length for state in states)
        keys = []
        values = []
        for state in states:
            if state.keys is None or state.values is None:
                raise RuntimeError("attention cache was not populated")
            padding = max_length - state.length
            keys.append(
                F.pad(state.keys, (0, 0, 0, padding)) if padding else state.keys
            )
            values.append(
                F.pad(state.values, (0, 0, 0, padding)) if padding else state.values
            )
        keys_tensor = torch.cat(keys).repeat_interleave(self.kv_groups, dim=1)
        values_tensor = torch.cat(values).repeat_interleave(self.kv_groups, dim=1)
        mask = None
        if any(state.length != max_length for state in states):
            cache_lengths = torch.tensor(
                [state.length for state in states], device=hidden.device
            )
            key_positions = torch.arange(max_length, device=hidden.device)
            mask = (
                key_positions[None, None, None, :] < cache_lengths[:, None, None, None]
            )
        output = F.scaled_dot_product_attention(
            queries,
            keys_tensor,
            values_tensor,
            attn_mask=mask,
            dropout_p=0.0,
            scale=self.scale,
        )
        output = output.transpose(1, 2).contiguous().view(batch, length, -1)
        return self.out_proj(output)


class ShortConv(nn.Module):
    def __init__(self, config: Lfm2Config) -> None:
        super().__init__()
        self.width = config.hidden_size
        self.state_length = config.conv_cache_length - 1
        self.conv = nn.Conv1d(
            self.width,
            self.width,
            kernel_size=config.conv_cache_length,
            groups=self.width,
            bias=config.conv_bias,
        )
        self.in_proj = nn.Linear(self.width, 3 * self.width, bias=config.conv_bias)
        self.out_proj = nn.Linear(self.width, self.width, bias=config.conv_bias)

    def forward(self, hidden: Tensor, state: RecurrentState) -> Tensor:
        gate_b, gate_c, value = self.in_proj(hidden).chunk(3, dim=-1)
        gated = (gate_b * value).transpose(1, 2)
        previous = state.value
        if previous is None:
            previous = gated.new_zeros(
                gated.shape[0], gated.shape[1], self.state_length
            )
        convolution_input = torch.cat((previous, gated), dim=-1)
        state.value = (
            convolution_input[..., -self.state_length :]
            if self.state_length
            else convolution_input[..., :0]
        )
        output = F.conv1d(
            convolution_input, self.conv.weight, self.conv.bias, groups=self.width
        )
        return self.out_proj(gate_c * output.transpose(1, 2))

    def forward_batch(self, hidden: Tensor, states: Sequence[RecurrentState]) -> Tensor:
        """Decode one token for each independently cached sequence."""

        batch, length, _ = hidden.shape
        if length != 1 or len(states) != batch:
            raise ValueError("batched convolution requires one state and token per row")
        gate_b, gate_c, value = self.in_proj(hidden).chunk(3, dim=-1)
        gated = (gate_b * value).transpose(1, 2)
        previous = torch.cat(
            [
                state.value
                if state.value is not None
                else gated.new_zeros(1, self.width, self.state_length)
                for state in states
            ]
        )
        convolution_input = torch.cat((previous, gated), dim=-1)
        next_state = (
            convolution_input[..., -self.state_length :]
            if self.state_length
            else convolution_input[..., :0]
        )
        for index, state in enumerate(states):
            state.value = next_state[index : index + 1]
        output = F.conv1d(
            convolution_input, self.conv.weight, self.conv.bias, groups=self.width
        )
        return self.out_proj(gate_c * output.transpose(1, 2))


class MLP(nn.Module):
    def __init__(self, config: Lfm2Config) -> None:
        super().__init__()
        size = config.intermediate_size
        if config.auto_adjust_ff_dim:
            size = int(config.ffn_dim_multiplier * int(2 * size / 3))
            size = config.multiple_of * (
                (size + config.multiple_of - 1) // config.multiple_of
            )
        self.w1 = nn.Linear(config.hidden_size, size, bias=False)
        self.w3 = nn.Linear(config.hidden_size, size, bias=False)
        self.w2 = nn.Linear(size, config.hidden_size, bias=False)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(hidden)) * self.w3(hidden))


class DecoderLayer(nn.Module):
    def __init__(self, config: Lfm2Config, layer_type: str) -> None:
        super().__init__()
        self.is_attention_layer = layer_type == "full_attention"
        if self.is_attention_layer:
            self.self_attn = Attention(config)
        else:
            self.conv = ShortConv(config)
        self.feed_forward = MLP(config)
        self.operator_norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, config.norm_eps)

    def forward(self, hidden: Tensor, positions: Tensor, state: LayerState) -> Tensor:
        normalized = self.operator_norm(hidden)
        if self.is_attention_layer:
            hidden = hidden + self.self_attn(normalized, positions, state)
        else:
            hidden = hidden + self.conv(normalized, state)
        return hidden + self.feed_forward(self.ffn_norm(hidden))

    def forward_batch(
        self, hidden: Tensor, positions: Tensor, states: list[LayerState]
    ) -> Tensor:
        normalized = self.operator_norm(hidden)
        if self.is_attention_layer:
            hidden = hidden + self.self_attn.forward_batch(
                normalized, positions, cast(Sequence[FullAttentionState], states)
            )
        else:
            hidden = hidden + self.conv.forward_batch(
                normalized, cast(Sequence[RecurrentState], states)
            )
        return hidden + self.feed_forward(self.ffn_norm(hidden))


class Lfm2Model(nn.Module):
    def __init__(self, config: Lfm2Config) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.layers = nn.ModuleList(
            DecoderLayer(config, layer_type) for layer_type in config.layer_types
        )
        self.embedding_norm = RMSNorm(config.hidden_size, config.norm_eps)

    def forward(self, tokens: Tensor, positions: Tensor, state: Lfm2State) -> Tensor:
        hidden = self.embed_tokens(tokens)
        for layer, layer_state in zip(self.layers, state.layers, strict=True):
            hidden = layer(hidden, positions, layer_state)
        return self.embedding_norm(hidden)

    def forward_batch(
        self, tokens: Tensor, positions: Tensor, states: list[Lfm2State]
    ) -> Tensor:
        hidden = self.embed_tokens(tokens)
        for index, layer in enumerate(cast(Sequence[DecoderLayer], self.layers)):
            layer_states = [state.layers[index] for state in states]
            hidden = layer.forward_batch(hidden, positions, layer_states)
        return self.embedding_norm(hidden)


class Lfm2ForCausalLM(nn.Module):
    def __init__(self, config: Lfm2Config) -> None:
        super().__init__()
        self.config = config
        self.args = config
        self.model = Lfm2Model(config)

    def make_cache(self) -> Lfm2State:
        return Lfm2State(self.config.layer_types)

    def materialize_buffers(self, device) -> None:
        for module in self.modules():
            if isinstance(module, Attention):
                module.materialize(device)

    def forward(
        self,
        tokens: Tensor,
        state: Lfm2State | None = None,
        positions: Tensor | None = None,
        *,
        logits_to_keep: int | None = None,
        return_hidden: bool = False,
    ) -> Tensor:
        if tokens.ndim != 2 or tokens.shape[0] != 1:
            raise ValueError("tokens must contain exactly one sequence")
        state = self.make_cache() if state is None else state
        if positions is None:
            positions = torch.arange(
                state.length,
                state.length + tokens.shape[1],
                dtype=torch.long,
                device=tokens.device,
            ).unsqueeze(0)
        if positions.shape != tokens.shape:
            raise ValueError("positions must have the same shape as tokens")
        hidden = self.model(tokens, positions, state)
        state.length += tokens.shape[1]
        if logits_to_keep is not None:
            if logits_to_keep <= 0:
                raise ValueError("logits_to_keep must be positive")
            hidden = hidden[:, -logits_to_keep:]
        return hidden if return_hidden else self.logits(hidden)

    def logits(self, hidden: Tensor) -> Tensor:
        return F.linear(hidden, self.model.embed_tokens.weight)

    def forward_batch(
        self,
        tokens: Tensor,
        states: list[Lfm2State],
        positions: Tensor | None = None,
    ) -> Tensor:
        """Decode one token for each independently cached sequence."""

        if tokens.ndim != 2 or tokens.shape[1] != 1:
            raise ValueError("batched decode requires one token per sequence")
        if tokens.shape[0] != len(states) or not states:
            raise ValueError("batched decode requires one state per sequence")
        if positions is None:
            positions = torch.tensor(
                [[state.length] for state in states],
                dtype=torch.long,
                device=tokens.device,
            )
        if positions.shape != tokens.shape:
            raise ValueError("positions must have the same shape as tokens")
        hidden = self.model.forward_batch(tokens, positions, states)
        for state in states:
            state.length += 1
        return F.linear(hidden, self.model.embed_tokens.weight)
