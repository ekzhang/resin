"""Inkling 2 target: block attention residuals, bounded attention, and sparse MLPs."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from resin.backend.cuda.parallel import TensorParallelContext
from resin.models.inkling2.config import Inkling2Config
from resin_models_cuda.inkling2.attention import (
    gated_attention,
    log_scaling_tau,
    prepare_paged_attention_batch,
)
from resin_models_cuda.inkling2.cache import GLOBAL_GROUP, LOCAL_GROUP, DeviceCache
from resin_models_cuda.inkling2.fa4_bias import (
    DECODE_TOKENS,
    packed_decode_batch_supported,
    packed_decode_group,
)
from resin_models_cuda.inkling2.kernels import (
    block_attention_residual,
    convolution_residual,
    l2_clip,
    residual_add,
    rms_norm,
    route_experts,
    short_convolution,
)

if TYPE_CHECKING:
    from resin.backend.topology import ParallelGroups


def linear(x: Tensor, weight: Tensor, *, fp32: bool = False) -> Tensor:
    if fp32 and x.is_cuda:
        return torch.mm(
            x.reshape(-1, x.shape[-1]), weight.t(), out_dtype=torch.float32
        ).view(*x.shape[:-1], weight.shape[0])
    return F.linear(x.float(), weight.float()) if fp32 else F.linear(x, weight)


class RMSNorm(nn.Module):
    def __init__(self, width: int, epsilon: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.ones(width, dtype=torch.bfloat16), requires_grad=False
        )
        self.eps = epsilon

    def forward(self, x: Tensor) -> Tensor:
        return rms_norm(x, self.weight, self.eps, dtype=torch.bfloat16)


class ShortConvolution(nn.Module):
    def __init__(self, width: int, *, heads: int = 1, dynamic_input: int = 0) -> None:
        super().__init__()
        self.heads = heads
        if dynamic_input:
            self.static_w = nn.Parameter(
                torch.empty(4, width, dtype=torch.float32), requires_grad=False
            )
            self.weight_proj = nn.Parameter(
                torch.empty(heads * 4, dynamic_input, dtype=torch.bfloat16),
                requires_grad=False,
            )
        else:
            self.weight = nn.Parameter(
                torch.empty(width, 1, 4, dtype=torch.float32), requires_grad=False
            )

    def forward(
        self,
        x: Tensor,
        state: Tensor,
        source: Tensor | None = None,
        *,
        state_out: Tensor | None = None,
        inputs_out: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if source is None:
            weight, dynamic = self.weight.view(-1, 4), None
        else:
            weight = self.static_w
            dynamic = linear(source, self.weight_proj).view(*x.shape[:2], self.heads, 4)
        return short_convolution(
            x, state, weight, dynamic, state_out=state_out, inputs_out=inputs_out
        )


class Inkling2Attention(nn.Module):
    def __init__(
        self,
        config: Inkling2Config,
        layer_id: int,
        tensor_parallel: TensorParallelContext,
    ) -> None:
        super().__init__()
        self.config, self.tensor_parallel = config, tensor_parallel
        self.allreduce: Callable[[Tensor], Tensor] = tensor_parallel.sum
        self.fused_prologue = False
        self.local = layer_id in config.local_layer_ids
        heads = (
            config.swa_num_attention_heads if self.local else config.num_attention_heads
        )
        kv_heads = (
            config.swa_num_key_value_heads if self.local else config.num_key_value_heads
        )
        self.heads = tensor_parallel.layout.even_shard(heads, name="query heads").size
        self.kv_heads = tensor_parallel.layout.even_shard(
            kv_heads, name="KV heads"
        ).size
        self.head_dim = config.swa_head_dim if self.local else config.head_dim
        h, kv, d, r = self.heads, self.kv_heads, self.head_dim, config.d_rel
        self.split_sizes = (h * d, kv * d, kv * d, h * r, h * d, h)
        # A local layer shares its source; a global layer has independent Q/K/V residual streams.
        self.qkvr = nn.Linear(
            config.hidden_size,
            sum(self.split_sizes) if self.local else h * (2 * d + r + 1),
            bias=False,
            dtype=torch.bfloat16,
        )
        if not self.local:
            self.wk_dv = nn.Linear(
                config.hidden_size, kv * d, bias=False, dtype=torch.bfloat16
            )
            self.wv_dv = nn.Linear(
                config.hidden_size, kv * d, bias=False, dtype=torch.bfloat16
            )
        self.kv_input_weight: Tensor | None = None
        self.wo_ud = nn.Linear(
            h * d, config.hidden_size, bias=False, dtype=torch.bfloat16
        )
        self.q_scale = nn.Parameter(
            torch.empty(h, d, dtype=torch.bfloat16), requires_grad=False
        )
        self.rel_scale = nn.Parameter(
            torch.empty(h, r, dtype=torch.bfloat16), requires_grad=False
        )
        self.rel_extent = (
            config.sliding_window_size if self.local else config.rel_extent
        )
        self.native_bias_padded_group = (
            packed_decode_group(h, kv)
            if d == 128 and r == 16 and self.rel_extent == 1024
            else None
        )
        self.rel_proj = nn.Parameter(
            torch.empty(r, self.rel_extent, dtype=torch.bfloat16), requires_grad=False
        )
        conv = dict(heads=kv, dynamic_input=0 if self.local else config.hidden_size)
        self.k_sconv, self.v_sconv = (
            ShortConvolution(kv * d, **conv),
            ShortConvolution(kv * d, **conv),
        )
        self.group = LOCAL_GROUP if self.local else GLOBAL_GROUP
        self.cache_layers = (
            config.execution_local_layer_count
            if self.local
            else config.execution_global_layer_count
        )

    def prepare_input_projection(self) -> None:
        if self.local or self.config.loop is None or self.kv_input_weight is not None:
            return
        self.kv_input_weight = torch.stack(
            (
                torch.cat((self.wk_dv.weight, self.k_sconv.weight_proj)),
                torch.cat((self.wv_dv.weight, self.v_sconv.weight_proj)),
            )
        ).transpose(1, 2)

    def forward(
        self,
        sources,
        positions,
        sequence_lengths,
        max_sequence_length,
        cache,
        attention_batch,
        state,
        *,
        cache_layer,
        state_out=None,
        inputs_out=None,
    ):
        stacked = sources if isinstance(sources, Tensor) else None
        q_source, k_source, v_source = sources if stacked is None else stacked.unbind(0)
        batch, tokens, _ = q_source.shape
        width = self.kv_heads * self.head_dim
        k_dynamic = v_dynamic = None
        if self.local:
            if stacked is None:
                q_projection = self.qkvr(q_source)
            else:
                # All streams share the projection weights, so one GEMM can
                # reuse them and expose the Q, K, and V slices for each input.
                q_projection, k_projection, v_projection = (
                    self.qkvr(stacked.flatten(0, 2))
                    .view(3, batch, tokens, -1)
                    .unbind(0)
                )
            q, k_input, v_input, relative, gate, sinks = q_projection.split(
                self.split_sizes, dim=-1
            )
            if stacked is not None:
                q_size, k_size, v_size = self.split_sizes[:3]
                k_input = k_projection[..., q_size : q_size + k_size]
                v_input = v_projection[..., q_size + k_size : q_size + k_size + v_size]
            elif k_source is not q_source:
                q_size, k_size, v_size = self.split_sizes[:3]
                k_weight, v_weight = self.qkvr.weight[
                    q_size : q_size + k_size + v_size
                ].split((k_size, v_size), dim=0)
                k_input, v_input = (
                    linear(k_source, k_weight),
                    linear(v_source, v_weight),
                )
        else:
            q, relative, gate, sinks = self.qkvr(q_source).split(
                (self.split_sizes[0], *self.split_sizes[3:]), dim=-1
            )
            if (
                stacked is not None
                and self.kv_input_weight is not None
                and self.fused_prologue
            ):
                k_projection, v_projection = (
                    torch.bmm(stacked[1:].flatten(1, 2), self.kv_input_weight)
                    .view(2, batch, tokens, -1)
                    .unbind(0)
                )
                k_input, k_coefficients = k_projection.split(
                    (width, self.kv_heads * 4), dim=-1
                )
                v_input, v_coefficients = v_projection.split(
                    (width, self.kv_heads * 4), dim=-1
                )
                k_dynamic = k_coefficients.view(batch, tokens, self.kv_heads, 4)
                v_dynamic = v_coefficients.view(batch, tokens, self.kv_heads, 4)
            else:
                k_input, v_input = self.wk_dv(k_source), self.wv_dv(v_source)
        tau = None if self.local else attention_batch.log_tau
        native_group = self.native_bias_padded_group
        projected = (
            self.fused_prologue
            and q.is_cuda
            and tokens in DECODE_TOKENS
            and native_group is not None
            and packed_decode_batch_supported(native_group, batch, inputs_padded=True)
        )
        if self.fused_prologue and q.is_cuda:
            from resin_models_cuda.inkling2.attention_prologue import attention_prologue

            if self.local:
                k_weight = self.k_sconv.weight.view(-1, 4)
                v_weight = self.v_sconv.weight.view(-1, 4)
            else:
                k_weight, v_weight = self.k_sconv.static_w, self.v_sconv.static_w
                if k_dynamic is None:
                    k_dynamic = linear(k_source, self.k_sconv.weight_proj).view(
                        batch, tokens, self.kv_heads, 4
                    )
                    v_dynamic = linear(v_source, self.v_sconv.weight_proj).view(
                        batch, tokens, self.kv_heads, 4
                    )
            q, relative, k_state, v_state = attention_prologue(
                q,
                k_input,
                v_input,
                relative,
                state[..., :width],
                state[..., width : 2 * width],
                k_weight,
                v_weight,
                q_scale=self.q_scale,
                rel_scale=self.rel_scale,
                rel_proj=self.rel_proj if projected else None,
                seqused_k=attention_batch.used_lengths[self.group]
                if projected
                else None,
                local=self.local,
                qk_threshold=self.config.qk_l2_clip_threshold,
                rel_threshold=self.config.rel_l2_clip_threshold,
                rms_eps=self.config.rms_norm_eps,
                cache=cache,
                positions=positions,
                group_index=self.group,
                layer=cache_layer,
                layer_count=self.cache_layers,
                k_dynamic=k_dynamic,
                v_dynamic=v_dynamic,
                tau=tau,
                k_state_out=None if state_out is None else state_out[..., :width],
                v_state_out=None
                if state_out is None
                else state_out[..., width : 2 * width],
                k_inputs_out=None if inputs_out is None else inputs_out[..., :width],
                v_inputs_out=None
                if inputs_out is None
                else inputs_out[..., width : 2 * width],
            )
        else:
            q, relative, k_state, v_state = self._prepare_attention(
                q,
                k_input,
                v_input,
                relative,
                k_source,
                v_source,
                state,
                cache,
                positions,
                tau,
                cache_layer=cache_layer,
                state_out=state_out,
                inputs_out=inputs_out,
            )
        if not projected:
            relative = relative.reshape(batch * tokens, self.heads, -1) @ self.rel_proj
        attended = gated_attention(
            q,
            gate,
            sinks,
            cache=cache,
            group_index=self.group,
            layer=cache_layer,
            layer_count=self.cache_layers,
            positions=positions,
            sequence_lengths=sequence_lengths,
            max_sequence_length=max_sequence_length,
            softmax_scale=self.head_dim**-0.5,
            window_left=self.config.sliding_window_size - 1 if self.local else None,
            relative_logits=relative,
            inputs_padded=projected,
            prepared_batch=attention_batch,
        )
        output = self.allreduce(self.wo_ud(attended.reshape(batch, tokens, -1)))
        return output, k_state, v_state, k_input, v_input

    def _prepare_attention(
        self,
        q,
        k_input,
        v_input,
        relative,
        k_source,
        v_source,
        state,
        cache,
        positions,
        tau,
        *,
        cache_layer,
        state_out=None,
        inputs_out=None,
    ):
        batch, tokens, _ = q.shape
        width = self.kv_heads * self.head_dim
        k, k_state = self.k_sconv(
            k_input,
            state[..., :width],
            None if self.local else k_source,
            state_out=None if state_out is None else state_out[..., :width],
            inputs_out=None if inputs_out is None else inputs_out[..., :width],
        )
        v, v_state = self.v_sconv(
            v_input,
            state[..., width : 2 * width],
            None if self.local else v_source,
            state_out=None if state_out is None else state_out[..., width : 2 * width],
            inputs_out=None
            if inputs_out is None
            else inputs_out[..., width : 2 * width],
        )
        q = l2_clip(
            q.reshape(batch, tokens, self.heads, self.head_dim),
            self.q_scale,
            self.config.qk_l2_clip_threshold,
            tau,
        )
        k = l2_clip(
            k.reshape(batch, tokens, self.kv_heads, self.head_dim),
            None,
            self.config.qk_l2_clip_threshold,
        )
        v = rms_norm(
            v.reshape(batch, tokens, self.kv_heads, self.head_dim),
            None,
            self.config.rms_norm_eps,
        )
        relative = l2_clip(
            relative.reshape(batch, tokens, self.heads, self.config.d_rel),
            self.rel_scale,
            self.config.rel_l2_clip_threshold,
            tau,
        )
        cache.store_kv(
            k,
            v,
            positions,
            group_index=self.group,
            layer=cache_layer,
            layer_count=self.cache_layers,
        )
        return q, relative, k_state, v_state


class SparseMlp(nn.Module):
    def __init__(
        self,
        config,
        *,
        precision,
        parallel_groups,
        expert_backend="triton",
    ):
        super().__init__()
        from resin_models_cuda.inkling2.experts import Inkling2Experts

        self.config = config
        self.gate = nn.Linear(
            config.hidden_size,
            config.num_routed_experts,
            bias=False,
            dtype=torch.bfloat16,
        )
        self.gate_bias = nn.Parameter(
            torch.empty(config.num_routed_experts, dtype=torch.float32),
            requires_grad=False,
        )
        self.experts = Inkling2Experts(
            config,
            precision=precision,
            parallel_groups=parallel_groups,
            expert_backend=expert_backend,
        )

    def forward(self, x, *, defer_rounding=False):
        flat = x.reshape(-1, x.shape[-1])
        ids, weights = route_experts(
            linear(flat, self.gate.weight, fp32=True),
            self.gate_bias,
            self.config.experts_per_token,
            self.config.route_scale,
        )
        # route_experts selects distinct expert IDs in each input row, so
        # no expert can receive more routes than the number of input rows.
        result = self.experts(
            flat,
            ids,
            weights,
            defer_rounding=defer_rounding,
            max_rows_per_expert=flat.shape[0],
        )
        return (result if defer_rounding else result.to(x.dtype)).view(x.shape)


class MessageController(nn.Module):
    def __init__(self, width: int, rank: int) -> None:
        super().__init__()
        self.down_weight = nn.Parameter(
            torch.empty(rank, 2 * width, dtype=torch.bfloat16), requires_grad=False
        )
        self.gain_weight = nn.Parameter(
            torch.empty(width, rank, dtype=torch.bfloat16), requires_grad=False
        )
        self.forget_weight = nn.Parameter(
            torch.empty(width, rank, dtype=torch.bfloat16), requires_grad=False
        )

    def forward(self, destination: Tensor, message: Tensor) -> Tensor:
        pair = torch.stack((destination, message), dim=-2)
        normalized = rms_norm(pair, None, 1e-6, dtype=torch.bfloat16).flatten(-2)
        hidden = linear(normalized, self.down_weight)
        gain = linear(hidden, self.gain_weight, fp32=True).tanh()
        forget = linear(hidden, self.forget_weight, fp32=True).tanh()
        return (1 - forget) * destination + gain * message


class DecoderLayer(nn.Module):
    attn_norm: RMSNorm
    q_input_norm: RMSNorm
    k_input_norm: RMSNorm
    v_input_norm: RMSNorm
    mlp_norm: RMSNorm
    prefill_ffn: Callable[..., Tensor] | None

    def __init__(
        self,
        config,
        layer_id,
        tensor_parallel,
        *,
        precision,
        parallel_groups,
        expert_backend="triton",
    ):
        super().__init__()
        self.local = layer_id in config.local_layer_ids
        self.attn = Inkling2Attention(config, layer_id, tensor_parallel)
        names = (
            ("attn_norm",)
            if self.local
            else ("q_input_norm", "k_input_norm", "v_input_norm")
        )
        for name in (*names, "mlp_norm"):
            setattr(self, name, RMSNorm(config.hidden_size, config.rms_norm_eps))
        self.attn_sconv, self.mlp_sconv = (
            ShortConvolution(config.hidden_size),
            ShortConvolution(config.hidden_size),
        )
        loop = config.loop
        self.controllers = (
            nn.ModuleDict(
                {
                    name: MessageController(config.hidden_size, loop.gate_rank)
                    for name in ("attn", "k", "v", "mlp")
                }
            )
            if loop is not None and loop.core_start <= layer_id < loop.core_end
            else None
        )
        self.controller_weights: tuple[Tensor, ...] | None = None
        self.mlp = SparseMlp(
            config,
            precision=precision,
            parallel_groups=parallel_groups,
            expert_backend=expert_backend,
        )
        self.prefill_ffn = None

    def feed_forward(self, hidden, token_ids=None):
        return self.mlp(hidden)

    def prepare_controllers(self) -> None:
        if self.controllers is None or self.controller_weights is not None:
            return
        packed = []
        for name in ("down_weight", "gain_weight", "forget_weight"):
            parameters = [
                module.get_parameter(name) for module in self.controllers.values()
            ]
            weight = torch.stack(tuple(parameters))
            for parameter, row in zip(parameters, weight, strict=True):
                parameter.data = row
            packed.append(weight)
        self.controller_weights = tuple(packed)

    def mix_loop_messages(self, streams, messages, *, mlp=False):
        assert self.controllers is not None
        shape = streams[0].shape
        # Wide controller tiles cover up to 64 eight-position verify blocks.
        max_rows = 512 if shape[1] in (1, 8) else 16
        if (
            self.controller_weights is not None
            and streams[0].is_cuda
            and 0 < shape[0] * shape[1] <= max_rows
            and shape[-1] == 2304
            and self.controller_weights[0].shape[1] == 64
        ):
            from resin_models_cuda.inkling2.triton import mix_loop_messages

            selected = slice(3, 4) if mlp else slice(3)
            output = mix_loop_messages(
                tuple(value.flatten(0, 1) for value in streams),
                tuple(value.flatten(0, 1) for value in messages),
                *(weight[selected] for weight in self.controller_weights),
            )
            return output.view(-1, *shape).unbind(0)
        names = ("mlp",) if mlp else ("attn", "k", "v")
        return tuple(
            self.controllers[name](stream, message)
            for name, stream, message in zip(names, streams, messages, strict=True)
        )


class AttentionResidual(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.hidden_size
        self.attn_queries = nn.ParameterList(
            nn.Parameter(
                torch.zeros(
                    width if i in config.local_layer_ids else (3, width),
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            for i in range(1, config.num_hidden_layers)
        )
        self.mlp_queries = nn.ParameterList(
            nn.Parameter(torch.zeros(width, dtype=torch.float32), requires_grad=False)
            for _ in range(config.num_hidden_layers)
        )
        self.output_query = nn.Parameter(
            torch.zeros(width, dtype=torch.float32), requires_grad=False
        )
        self.normalize_values = config.attn_res_normalize_values

    def mix(self, sources, query, partial_addend=None):
        return block_attention_residual(
            sources,
            query,
            normalize_values=self.normalize_values,
            partial_addend=partial_addend,
        )


class Inkling2Target(nn.Module):
    def __init__(
        self,
        config: Inkling2Config,
        tap_layer_ids: tuple[int, ...],
        *,
        tensor_parallel: TensorParallelContext | None = None,
        parallel_groups: ParallelGroups | None = None,
        precision: dict[str, str] | None = None,
        expert_backend: str = "triton",
    ) -> None:
        super().__init__()
        self.config, self.tap_layer_ids = config, tap_layer_ids
        self.tensor_parallel = tensor_parallel or TensorParallelContext()
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, dtype=torch.bfloat16
        )
        self.embed_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.layers = nn.ModuleList(
            DecoderLayer(
                config,
                i,
                self.tensor_parallel,
                precision={
                    key: precision[f"model.llm.layers.{i}.mlp.experts.{key}_weight"]
                    for key in ("w13", "w2")
                }
                if precision
                else None,
                parallel_groups=parallel_groups,
                expert_backend=expert_backend,
            )
            for i in range(config.num_hidden_layers)
        )
        self.attn_res = AttentionResidual(config)
        self.register_parameter("middle_cycle_cross_pass_queries", None)
        self.register_parameter("middle_cycle_cross_pass_kv_queries", None)
        self.loop_message_queries: Tensor | None = None
        if config.loop is not None:
            core_length = config.loop.core_end - config.loop.core_start
            shape = (2 * core_length, config.hidden_size)
            self.middle_cycle_cross_pass_queries = nn.Parameter(
                torch.empty(shape, dtype=torch.float32), requires_grad=False
            )
            self.middle_cycle_cross_pass_kv_queries = nn.Parameter(
                torch.empty(shape, dtype=torch.float32), requires_grad=False
            )
        layer_order = config.layer_order
        cache_layers = {}
        for group in (
            config.execution_local_layer_ids,
            config.execution_global_layer_ids,
        ):
            cache_layers.update((visit, slot) for slot, visit in enumerate(group))
        self.layer_visits = tuple(
            (layer_id, cache_layers[visit])
            for visit, layer_id in enumerate(layer_order)
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.unembed = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            dtype=torch.bfloat16,
        )

    @torch.no_grad()
    def prepare(self):
        for layer in cast(Sequence[DecoderLayer], self.layers):
            layer.mlp.experts.prepare()
            if self.embed_tokens.weight.is_cuda:
                layer.attn.prepare_input_projection()
                layer.prepare_controllers()
        if (
            self.middle_cycle_cross_pass_queries is not None
            and self.embed_tokens.weight.is_cuda
        ):
            queries = cast(Tensor, self.middle_cycle_cross_pass_queries)
            kv_queries = cast(Tensor, self.middle_cycle_cross_pass_kv_queries)
            self.loop_message_queries = torch.stack(
                (queries[::2], kv_queries[::2], kv_queries[1::2], queries[1::2]), dim=1
            )

    def forward_experts_empty(self):
        hidden = self.embed_tokens.weight.new_empty((0, self.config.hidden_size))
        layers = cast(Sequence[DecoderLayer], self.layers)
        for layer_id, _ in self.layer_visits:
            layer = layers[layer_id]
            callback = layer.prefill_ffn or layer.feed_forward
            callback(hidden)

    def logits(self, hidden):
        values = linear(hidden, self.unembed.weight, fp32=True)
        cap = self.config.final_logit_softcapping
        values = values[..., : self.config.unpadded_vocab_size]
        if values.is_cuda:
            from resin_models_cuda.inkling2.dspark_kernels import softcap_logits

            return softcap_logits(values, cap)
        return cap * torch.tanh(values / cap)

    def forward(
        self,
        token_ids: Tensor,
        positions: Tensor,
        cache: DeviceCache,
        state: Tensor,
        sequence_lengths: Tensor,
        max_sequence_length: int,
        *,
        full_logits: bool,
        prefill: bool = False,
        input_embeddings: Tensor | None = None,
        state_inputs_out: Tensor | None = None,
        return_hidden: bool = False,
    ):
        batch, tokens = token_ids.shape
        config = self.config
        rows = batch * tokens
        fused_tail = (
            token_ids.is_cuda
            and not prefill
            and (
                rows <= 16
                or (config.loop is not None and tokens in DECODE_TOKENS and rows <= 512)
            )
            and state.dtype == torch.bfloat16
        )
        width = config.state_width(self.tensor_parallel.size)
        if state.shape != (batch, config.execution_layer_count, 3, width):
            raise ValueError("Inkling 2 target received malformed convolution state")
        if state_inputs_out is not None and state_inputs_out.shape != (
            batch,
            tokens,
            config.execution_layer_count,
            width,
        ):
            raise ValueError("Inkling 2 convolution input buffer has the wrong shape")
        partial = (
            self.embed_norm(self.embed_tokens(token_ids))
            if input_embeddings is None
            else input_embeddings
        )
        blocks: list[Tensor] = []
        taps: list[Tensor] = []
        loop = config.loop
        prelude: tuple[list[Tensor], Tensor] | None = None
        message_bank: list[Tensor] = []
        # Each producer fills its own channel range without mutating the state
        # needed to roll back to a partially accepted speculative prefix.
        next_state = torch.empty_like(state, memory_format=torch.contiguous_format)
        attention_batch = prepare_paged_attention_batch(
            cache,
            sequence_lengths,
            max_sequence_length,
            batch=batch,
            tokens=tokens,
            layer_counts={
                GLOBAL_GROUP: config.execution_global_layer_count,
                LOCAL_GROUP: config.execution_local_layer_count,
            },
            global_prefill_relative_extent=config.rel_extent if prefill else None,
            # The scale is exactly one below the floor. Keep the same global
            # attention specialization on both sides of that boundary.
            log_tau=log_scaling_tau(
                positions,
                n_floor=config.log_scaling_n_floor,
                alpha=config.log_scaling_alpha,
            ),
        )
        layers = cast(Sequence[DecoderLayer], self.layers)
        kv_width = layers[0].attn.kv_heads * config.head_dim
        for visit, (i, cache_layer) in enumerate(self.layer_visits):
            layer = layers[i]
            if loop is not None:
                if i == loop.core_start:
                    if prelude is None:
                        prelude = (blocks.copy(), partial)
                    else:
                        prelude_blocks, prelude_partial = prelude
                        # The first core layer closes the prelude's last block.
                        # Only blocks produced inside the previous pass carry messages.
                        message_bank = [*blocks[len(prelude_blocks) + 1 :], partial]
                        blocks, partial = prelude_blocks.copy(), prelude_partial
                elif i == loop.core_end:
                    message_bank = []
            if visit in self.tap_layer_ids:
                taps.append(partial)
            streams = (
                self.attn_res.mix([*blocks, partial], self.attn_res.attn_queries[i - 1])
                if blocks
                else (partial,) * (1 if layer.local else 3)
            )
            messages = None
            if message_bank:
                assert loop is not None and layer.controllers is not None
                if self.loop_message_queries is None:
                    queries = cast(Tensor, self.middle_cycle_cross_pass_queries)
                    kv_queries = cast(Tensor, self.middle_cycle_cross_pass_kv_queries)
                    start = 2 * (i - loop.core_start)
                    query = torch.cat(
                        (
                            queries[start : start + 1],
                            kv_queries[start : start + 2],
                            queries[start + 1 : start + 2],
                        )
                    )
                else:
                    query = self.loop_message_queries[i - loop.core_start]
                messages = block_attention_residual(
                    message_bank, query, normalize_values=True
                )
                streams = layer.mix_loop_messages(
                    streams * 3 if layer.local else streams, messages[:3]
                )
            if layer.local and messages is None:
                q = layer.attn_norm(streams[0])
                sources = (q, q, q)
            else:
                norms = (
                    (layer.attn_norm,) * 3
                    if layer.local
                    else (layer.q_input_norm, layer.k_input_norm, layer.v_input_norm)
                )
                if (
                    loop is not None
                    and not prefill
                    and layer.attn.fused_prologue
                    and token_ids.is_cuda
                ):
                    from resin_models_cuda.inkling2.triton import attention_input_norm

                    sources = attention_input_norm(
                        streams,
                        tuple(norm.weight for norm in norms),
                        config.rms_norm_eps,
                    )
                else:
                    sources = tuple(
                        norm(x) for norm, x in zip(norms, streams, strict=True)
                    )
            layer_state = next_state[:, visit]
            layer_inputs = (
                None if state_inputs_out is None else state_inputs_out[:, :, visit]
            )
            attention, _, _, _, _ = layer.attn(
                sources,
                positions,
                sequence_lengths,
                max_sequence_length,
                cache,
                attention_batch,
                state[:, visit],
                cache_layer=cache_layer,
                state_out=layer_state,
                inputs_out=layer_inputs,
            )
            attention_slice = slice(2 * kv_width, 2 * kv_width + config.hidden_size)
            attention, _ = layer.attn_sconv(
                attention,
                state[:, visit, :, attention_slice],
                state_out=layer_state[..., attention_slice],
                inputs_out=None
                if layer_inputs is None
                else layer_inputs[..., attention_slice],
            )
            boundary = i % config.n_layers_per_attn_res_block == 0
            mlp_sources = (
                [*blocks, partial, attention] if boundary else [*blocks, partial]
            )
            mlp_input = self.attn_res.mix(
                mlp_sources,
                self.attn_res.mlp_queries[i],
                None if boundary else attention,
            )[0]
            if messages is not None:
                (mlp_input,) = layer.mix_loop_messages(
                    (mlp_input,), messages[3:], mlp=True
                )
            callback = layer.prefill_ffn or layer.feed_forward
            normalized = layer.mlp_norm(mlp_input).reshape(-1, config.hidden_size)
            # Existing prefill callbacks keep their output dtype. Direct decode
            # defers its BF16 round to the fused convolution's input loads.
            mlp_input = (
                layer.mlp(normalized, defer_rounding=True)
                if fused_tail
                else callback(normalized)
            ).view(batch, tokens, config.hidden_size)
            history = state[:, visit, :, -config.hidden_size :]
            destination = layer_state[..., -config.hidden_size :]
            raw = (
                None
                if layer_inputs is None
                else layer_inputs[..., -config.hidden_size :]
            )
            if boundary:
                blocks.append(partial)
            if fused_tail:
                partial, _ = convolution_residual(
                    mlp_input,
                    history,
                    layer.mlp_sconv.weight.view(-1, 4),
                    attention,
                    None if boundary else partial,
                    state_out=destination,
                    inputs_out=raw,
                )
            else:
                mlp, _ = layer.mlp_sconv(
                    mlp_input,
                    history,
                    state_out=destination,
                    inputs_out=raw,
                )
                partial = (
                    residual_add(attention, mlp)
                    if boundary
                    else residual_add(partial, attention, mlp)
                )
        hidden = self.norm(
            self.attn_res.mix([*blocks, partial], self.attn_res.output_query)[0]
        )
        if config.execution_layer_count in self.tap_layer_ids:
            taps.append(hidden)
        selected = hidden if full_logits else hidden[:, -1:]
        result = selected if return_hidden else self.logits(selected)
        return (
            result,
            torch.cat(taps, dim=-1),
            next_state,
            state_inputs_out,
        )


__all__ = [
    "Inkling2Target",
    "Inkling2Attention",
    "RMSNorm",
    "ShortConvolution",
]
