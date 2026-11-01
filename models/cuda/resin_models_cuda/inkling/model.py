"""Native text target for Thinking Machines' Inkling-Small NVFP4 checkpoint."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from resin.backend.layers import RMSNorm
from resin.models.inkling.config import InklingConfig
from resin_models_cuda.inkling.attention import (
    PagedAttentionBatch,
    log_scaling_tau,
    paged_attention,
    prepare_paged_attention_batch,
)
from resin_models_cuda.inkling.cache import (
    GLOBAL_GROUP,
    GLOBAL_LAYERS,
    LOCAL_GROUP,
    LOCAL_LAYERS,
    STATE_LENGTH,
    STATE_WIDTH_PER_LAYER,
    DeviceCache,
)
from resin_models_cuda.nvfp4 import prepare_nvfp4_moe_weights, quantize_nvfp4


class ShortConvolution(nn.Module):
    """Depthwise causal width-four convolution over an explicit state window."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width
        self.weight = nn.Parameter(
            torch.empty(width, 1, STATE_LENGTH + 1, dtype=torch.float32)
        )

    def forward(self, hidden: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        if hidden.ndim != 3 or state.shape != (
            hidden.shape[0],
            STATE_LENGTH,
            self.width,
        ):
            raise ValueError("Inkling short-convolution state has the wrong shape")
        if hidden.is_cuda and hidden.dtype == torch.bfloat16 and self.width % 128 == 0:
            from resin_models_cuda.inkling.cutile import short_convolution

            return short_convolution(hidden, state, self.weight)
        combined = torch.cat((state, hidden), dim=1)
        convolved = F.conv1d(
            combined.transpose(1, 2).float(),
            self.weight,
            groups=self.width,
        ).transpose(1, 2)
        output = (convolved + hidden.float()).to(hidden.dtype)
        next_state = combined[:, -STATE_LENGTH:].to(state.dtype)
        return output, next_state


def select_convolution_state(
    initial: Tensor,
    inputs: Tensor,
    processed: Tensor,
    *,
    out: Tensor | None = None,
) -> Tensor:
    """Select the width-three state after each row's accepted token count."""

    from resin_models_cuda.inkling.cutile import (
        select_convolution_state as persistent_select_convolution_state,
    )

    return persistent_select_convolution_state(
        initial,
        inputs,
        processed,
        out=out,
    )


def _deinterleave_w13(weight: Tensor, *, up_first: bool) -> Tensor:
    if weight.shape[-2] % 2:
        raise ValueError("Inkling fused gate/up rows must be even")
    grouped = weight.reshape(
        *weight.shape[:-2], weight.shape[-2] // 2, 2, weight.shape[-1]
    )
    if up_first:
        grouped = grouped.flip(-2)
    return grouped.transpose(-3, -2).reshape_as(weight).contiguous()


class DenseMlp(nn.Module):
    def __init__(self, hidden: int, intermediate: int) -> None:
        super().__init__()
        self.w13 = nn.Parameter(
            torch.empty(2 * intermediate, hidden, dtype=torch.bfloat16)
        )
        self.w2 = nn.Parameter(torch.empty(hidden, intermediate, dtype=torch.bfloat16))
        self.global_scale = nn.Parameter(torch.ones(1, dtype=torch.bfloat16))

    def forward(self, hidden: Tensor) -> Tensor:
        from resin_models_cuda.inkling.cutile import dense_swiglu

        projected = F.linear(hidden, self.w13)
        return F.linear(dense_swiglu(projected), self.w2) * self.global_scale


class Router(nn.Module):
    def __init__(self, config: InklingConfig) -> None:
        super().__init__()
        self.routed_experts = config.num_routed_experts
        self.shared_experts = config.num_shared_experts
        self.top_k = config.experts_per_token
        self.route_scale = config.route_scale
        self.weight = nn.Parameter(
            torch.empty(
                config.num_routed_experts + config.num_shared_experts,
                config.hidden_size,
                dtype=torch.bfloat16,
            )
        )
        self.bias = nn.Parameter(
            torch.empty(config.num_routed_experts, dtype=torch.bfloat16)
        )
        self.global_scale = nn.Parameter(torch.ones(1, dtype=torch.bfloat16))

    def forward(self, hidden: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        from resin_models_cuda.inkling.cutile import (
            router_choice,
            router_weights,
        )

        flat = hidden.reshape(-1, hidden.shape[-1])
        logits = F.linear(flat, self.weight)
        choice = router_choice(logits, self.bias)
        indices = torch.topk(choice, self.top_k, dim=-1, sorted=False).indices
        weights, shared_weights = router_weights(
            logits,
            indices,
            self.global_scale,
            self.route_scale,
            routed_experts=self.routed_experts,
            shared_experts=self.shared_experts,
        )
        return indices, weights, shared_weights


class SharedExperts(nn.Module):
    def __init__(self, config: InklingConfig) -> None:
        super().__init__()
        experts = config.num_shared_experts
        intermediate = config.moe_intermediate_size
        hidden = config.hidden_size
        self.experts = experts
        self.w13 = nn.Parameter(
            torch.empty(experts, 2 * intermediate, hidden, dtype=torch.bfloat16)
        )
        self.w2 = nn.Parameter(
            torch.empty(experts, hidden, intermediate, dtype=torch.bfloat16)
        )

    def forward(
        self,
        hidden: Tensor,
        gammas: Tensor,
        routed: Tensor,
    ) -> Tensor:
        from resin_models_cuda.inkling.cutile import (
            shared_reduce_add,
            shared_swiglu,
        )

        shape = hidden.shape
        flat = hidden.reshape(-1, shape[-1])
        expanded = flat.unsqueeze(0).expand(self.experts, -1, -1)
        projected = torch.bmm(expanded, self.w13.transpose(1, 2))
        activated = shared_swiglu(projected, gammas)
        output = torch.bmm(activated, self.w2.transpose(1, 2))
        return shared_reduce_add(output, routed).view(shape)


class Bf16Experts(nn.Module):
    def __init__(self, config: InklingConfig) -> None:
        super().__init__()
        experts = config.num_routed_experts
        intermediate = config.moe_intermediate_size
        hidden = config.hidden_size
        self.num_experts = experts
        self.intermediate = intermediate
        self.hidden = hidden
        self.w13 = nn.Parameter(
            torch.empty(experts, 2 * intermediate, hidden, dtype=torch.bfloat16)
        )
        self.w2 = nn.Parameter(
            torch.empty(experts, hidden, intermediate, dtype=torch.bfloat16)
        )
        self.prepared = False

    def prepare(self) -> None:
        if self.prepared:
            return
        from flashinfer import reorder_rows_for_gated_act_gemm, shuffle_matrix_a
        from flashinfer.fused_moe import convert_to_block_layout

        with torch.no_grad():
            for expert in range(self.num_experts):
                w13 = _deinterleave_w13(self.w13[expert], up_first=True)
                w13 = reorder_rows_for_gated_act_gemm(w13)
                w13 = convert_to_block_layout(
                    shuffle_matrix_a(w13.view(torch.uint8), 64), 128
                ).view(torch.bfloat16)
                w2 = convert_to_block_layout(
                    shuffle_matrix_a(self.w2[expert].view(torch.uint8), 64), 128
                ).view(torch.bfloat16)
                self.w13[expert].view(-1).copy_(w13.view(-1))
                self.w2[expert].view(-1).copy_(w2.view(-1))
        self.w13.data = self.w13.data.view(
            self.num_experts, self.hidden // 64, 2 * self.intermediate, 64
        )
        self.w2.data = self.w2.data.view(
            self.num_experts, self.intermediate // 64, self.hidden, 64
        )
        self.prepared = True

    def forward(self, hidden: Tensor, ids: Tensor, weights: Tensor) -> Tensor:
        if not self.prepared:
            raise RuntimeError("Inkling BF16 experts must be prepared before serving")
        from flashinfer.fused_moe import trtllm_bf16_routed_moe
        from flashinfer.tllm_enums import ActivationType, WeightLayout

        packed_ids = (ids.to(torch.int32) << 16) | weights.view(torch.int16).to(
            torch.int32
        )
        output = trtllm_bf16_routed_moe(
            topk_ids=packed_ids,
            hidden_states=hidden,
            gemm1_weights=self.w13,
            gemm2_weights=self.w2,
            num_experts=self.num_experts,
            top_k=ids.shape[1],
            n_group=None,
            topk_group=None,
            intermediate_size=self.intermediate,
            local_expert_offset=0,
            local_num_experts=self.num_experts,
            routed_scaling_factor=None,
            routing_method_type=5,
            use_shuffled_weight=True,
            weight_layout=WeightLayout.BlockMajorK,
            do_finalize=True,
            activation_type=ActivationType.Swiglu.value,
            tune_max_num_tokens=max(1, 1 << (hidden.shape[0] - 1).bit_length()),
            enable_pdl=True,
        )
        return output[0] if isinstance(output, (tuple, list)) else output


class Nvfp4Experts(nn.Module):
    w13: Tensor
    w13_scales: Tensor
    w2: Tensor
    w2_scales: Tensor
    w13_input_amax: Tensor | None
    w2_input_amax: Tensor | None
    w13_scale2: Tensor | None
    w2_scale2: Tensor | None
    input_scale_quant: Tensor
    g1_scale_c: Tensor
    g1_alphas: Tensor
    g2_alphas: Tensor

    def __init__(self, config: InklingConfig) -> None:
        super().__init__()
        experts = config.num_routed_experts
        intermediate = config.moe_intermediate_size
        hidden = config.hidden_size
        self.num_experts = experts
        self.intermediate = intermediate
        self.hidden = hidden
        self.register_buffer(
            "w13",
            torch.empty(experts, 2 * intermediate, hidden // 2, dtype=torch.uint8),
        )
        self.register_buffer(
            "w13_scales",
            torch.empty(
                experts,
                2 * intermediate,
                hidden // 16,
                dtype=torch.float8_e4m3fn,
            ),
        )
        self.register_buffer(
            "w2",
            torch.empty(experts, hidden, intermediate // 2, dtype=torch.uint8),
        )
        self.register_buffer(
            "w2_scales",
            torch.empty(
                experts,
                hidden,
                intermediate // 16,
                dtype=torch.float8_e4m3fn,
            ),
        )
        self.register_buffer("w13_input_amax", torch.empty(1, dtype=torch.float32))
        self.register_buffer("w2_input_amax", torch.empty(1, dtype=torch.float32))
        self.register_buffer("w13_scale2", torch.empty(experts, dtype=torch.float32))
        self.register_buffer("w2_scale2", torch.empty(experts, dtype=torch.float32))
        self.register_buffer("input_scale_quant", torch.empty((), dtype=torch.float32))
        self.register_buffer("g1_scale_c", torch.empty(experts, dtype=torch.float32))
        self.register_buffer("g1_alphas", torch.empty(experts, dtype=torch.float32))
        self.register_buffer("g2_alphas", torch.empty(experts, dtype=torch.float32))
        self.prepared = False

    def prepare(self) -> None:
        if self.prepared:
            return
        with torch.no_grad():
            for expert in range(self.num_experts):
                self.w13[expert].copy_(
                    _deinterleave_w13(self.w13[expert], up_first=True)
                )
                self.w13_scales[expert].copy_(
                    _deinterleave_w13(self.w13_scales[expert], up_first=True)
                )
        assert self.w13_input_amax is not None and self.w2_input_amax is not None
        assert self.w13_scale2 is not None and self.w2_scale2 is not None
        w13_input = self.w13_input_amax.float().max() / (448.0 * 6.0)
        w2_input = self.w2_input_amax.float().max() / (448.0 * 6.0)
        up_alpha = w13_input * self.w13_scale2
        gate_alpha = w13_input * self.w13_scale2
        self.input_scale_quant.copy_(w13_input.reciprocal())
        self.g1_alphas.copy_(gate_alpha)
        self.g1_scale_c.copy_(w2_input.reciprocal() * up_alpha)
        self.g2_alphas.copy_(w2_input * self.w2_scale2)
        prepare_nvfp4_moe_weights(self.w13, self.w13_scales, self.w2, self.w2_scales)
        self.w13_input_amax = None
        self.w2_input_amax = None
        self.w13_scale2 = None
        self.w2_scale2 = None
        self.prepared = True

    def forward(self, hidden: Tensor, ids: Tensor, weights: Tensor) -> Tensor:
        if not self.prepared:
            raise RuntimeError("Inkling NVFP4 experts must be prepared before serving")
        from flashinfer.fused_moe import trtllm_fp4_block_scale_routed_moe
        from flashinfer.tllm_enums import ActivationType

        packed, scales = quantize_nvfp4(hidden, self.input_scale_quant)
        output = trtllm_fp4_block_scale_routed_moe(
            topk_ids=(ids.to(torch.int32), weights),
            routing_bias=None,
            hidden_states=packed,
            hidden_states_scale=scales,
            gemm1_weights=self.w13,
            gemm1_weights_scale=self.w13_scales,
            gemm1_bias=None,
            gemm1_alpha=None,
            gemm1_beta=None,
            gemm1_clamp_limit=None,
            gemm2_weights=self.w2,
            gemm2_weights_scale=self.w2_scales,
            gemm2_bias=None,
            output1_scale_scalar=self.g1_scale_c,
            output1_scale_gate_scalar=self.g1_alphas,
            output2_scale_scalar=self.g2_alphas,
            num_experts=self.num_experts,
            top_k=ids.shape[1],
            n_group=0,
            topk_group=0,
            intermediate_size=self.intermediate,
            local_expert_offset=0,
            local_num_experts=self.num_experts,
            routed_scaling_factor=None,
            routing_method_type=1,
            do_finalize=True,
            activation_type=ActivationType.Swiglu.value,
            tune_max_num_tokens=max(1, 1 << (hidden.shape[0] - 1).bit_length()),
            enable_pdl=True,
        )
        return output[0]


class SparseMlp(nn.Module):
    def __init__(self, config: InklingConfig, *, bf16: bool) -> None:
        super().__init__()
        self.router = Router(config)
        self.shared = SharedExperts(config)
        self.routed: Bf16Experts | Nvfp4Experts = (
            Bf16Experts(config) if bf16 else Nvfp4Experts(config)
        )

    def prepare(self) -> None:
        self.routed.prepare()

    def forward(self, hidden: Tensor) -> Tensor:
        shape = hidden.shape
        flat = hidden.reshape(-1, shape[-1])
        ids, weights, gammas = self.router(flat)
        routed = self.routed(flat, ids, weights).view(shape)
        return self.shared(hidden, gammas, routed.view(-1, shape[-1]))


class InklingAttention(nn.Module):
    def __init__(self, config: InklingConfig, layer_id: int) -> None:
        super().__init__()
        hidden = config.hidden_size
        heads = config.num_attention_heads
        kv_heads = config.num_key_value_heads
        dim = config.head_dim
        self.layer_id = layer_id
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = dim
        self.local = layer_id in config.local_layer_ids
        layer_ids = config.local_layer_ids if self.local else config.global_layer_ids
        self.cache_layer = layer_ids.index(layer_id)
        self.cache_layers = LOCAL_LAYERS if self.local else GLOBAL_LAYERS
        self.group = LOCAL_GROUP if self.local else GLOBAL_GROUP
        self.rel_extent = (
            config.sliding_window_size if self.local else config.rel_extent
        )
        self.window_left = config.sliding_window_size - 1 if self.local else None
        self.scale = 1.0 / dim
        self.log_floor = config.log_scaling_n_floor
        self.log_alpha = config.log_scaling_alpha
        self.wq = nn.Linear(hidden, heads * dim, bias=False, dtype=torch.bfloat16)
        self.wk = nn.Linear(hidden, kv_heads * dim, bias=False, dtype=torch.bfloat16)
        self.wv = nn.Linear(hidden, kv_heads * dim, bias=False, dtype=torch.bfloat16)
        self.wr = nn.Linear(
            hidden, heads * config.d_rel, bias=False, dtype=torch.bfloat16
        )
        self.wo = nn.Linear(heads * dim, hidden, bias=False, dtype=torch.bfloat16)
        self.q_norm = RMSNorm(dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(dim, config.rms_norm_eps)
        self.k_sconv = ShortConvolution(kv_heads * dim)
        self.v_sconv = ShortConvolution(kv_heads * dim)
        self.rel_proj = nn.Parameter(
            torch.empty(config.d_rel, self.rel_extent, dtype=torch.bfloat16)
        )

    def forward(
        self,
        hidden: Tensor,
        positions: Tensor,
        sequence_lengths: Tensor,
        max_sequence_length: int,
        cache: DeviceCache,
        attention_batch: PagedAttentionBatch,
        k_state: Tensor,
        v_state: Tensor,
        *,
        prefill: bool,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        batch, tokens, _ = hidden.shape
        q = self.wq(hidden).view(batch, tokens, self.heads, self.head_dim)
        k_input = self.wk(hidden)
        v_input = self.wv(hidden)
        if prefill:
            k, k_state = self.k_sconv(k_input, k_state)
            v, v_state = self.v_sconv(v_input, v_state)
            k = self.k_norm(k.view(batch, tokens, self.kv_heads, self.head_dim))
            v = v.view(batch, tokens, self.kv_heads, self.head_dim)
            pages, offsets = cache.page_locations(
                group_index=self.group,
                layer=self.cache_layer,
                layer_count=self.cache_layers,
                positions=positions,
            )
            cache.keys[pages, offsets] = k.to(cache.keys.dtype)
            cache.values[pages, offsets] = v.to(cache.values.dtype)
        else:
            from resin_models_cuda.inkling.cutile import kv_sconv_norm_store

            group = cache.groups[self.group]
            k_state, v_state = kv_sconv_norm_store(
                k_input,
                v_input,
                k_state,
                v_state,
                self.k_sconv.weight,
                self.v_sconv.weight,
                self.k_norm.weight,
                cache.keys,
                cache.values,
                group.slab_ids,
                group.slotnums,
                group.first_blocks,
                positions,
                page_tokens=cache.page_tokens,
                slab_units=cache.slab_units,
                layer_count=self.cache_layers,
                layer=self.cache_layer,
                epsilon=self.k_norm.eps,
            )
        q = self.q_norm(q)
        relative = self.wr(hidden).view(batch * tokens, self.heads, -1)
        relative = relative @ self.rel_proj
        if not self.local and max_sequence_length > self.log_floor:
            tau = attention_batch.log_tau
            if tau is None:
                tau = log_scaling_tau(
                    positions, n_floor=self.log_floor, alpha=self.log_alpha
                )
            q = (q.float() * tau[..., None, None]).to(q.dtype)
            relative = (relative.float() * tau.reshape(-1, 1, 1)).to(relative.dtype)
        attended = paged_attention(
            q,
            cache=cache,
            group_index=self.group,
            layer=self.cache_layer,
            layer_count=self.cache_layers,
            positions=positions,
            sequence_lengths=sequence_lengths,
            max_sequence_length=max_sequence_length,
            softmax_scale=self.scale,
            window_left=self.window_left,
            relative_logits=relative,
            prepared_batch=attention_batch,
        )
        return (
            self.wo(attended.reshape(batch, tokens, -1)),
            k_state,
            v_state,
            k_input,
            v_input,
        )


class DecoderLayer(nn.Module):
    def __init__(self, config: InklingConfig, layer_id: int) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attn = InklingAttention(config, layer_id)
        self.attn_sconv = ShortConvolution(config.hidden_size)
        self.mlp_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp: DenseMlp | SparseMlp = (
            DenseMlp(config.hidden_size, config.dense_intermediate_size)
            if layer_id < config.dense_layer_count
            else SparseMlp(config, bf16=layer_id == config.dense_layer_count)
        )
        self.mlp_sconv = ShortConvolution(config.hidden_size)
        self.prefill_ffn: Callable | None = None

    def prepare(self) -> None:
        if isinstance(self.mlp, SparseMlp):
            self.mlp.prepare()

    def feed_forward(self, hidden: Tensor, _ids: Tensor | None = None) -> Tensor:
        return self.mlp(self.mlp_norm(hidden))

    def forward(
        self,
        hidden: Tensor,
        positions: Tensor,
        sequence_lengths: Tensor,
        max_sequence_length: int,
        cache: DeviceCache,
        attention_batch: PagedAttentionBatch,
        state: Tensor,
        *,
        prefill: bool,
        record_state_inputs: bool,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
        k_state, v_state, attn_state, mlp_state = torch.split(
            state, [1024, 1024, 4096, 4096], dim=-1
        )
        residual = hidden
        attended, k_state, v_state, k_input, v_input = self.attn(
            self.attn_norm(hidden),
            positions,
            sequence_lengths,
            max_sequence_length,
            cache,
            attention_batch,
            k_state,
            v_state,
            prefill=prefill,
        )
        attn_input = attended
        if prefill:
            attended, attn_state = self.attn_sconv(attn_input, attn_state)
            hidden = residual + attended
        else:
            from resin_models_cuda.inkling.cutile import sconv_residual

            hidden, attn_state = sconv_residual(
                attn_input,
                attn_state,
                self.attn_sconv.weight,
                residual,
            )
        residual = hidden
        mlp_hidden = (self.prefill_ffn or self.feed_forward)(hidden)
        tap = residual + mlp_hidden
        mlp_input = mlp_hidden
        if prefill:
            mlp_hidden, mlp_state = self.mlp_sconv(mlp_input, mlp_state)
            hidden = residual + mlp_hidden
        else:
            hidden, mlp_state = sconv_residual(
                mlp_input,
                mlp_state,
                self.mlp_sconv.weight,
                residual,
            )
        new_state = torch.cat((k_state, v_state, attn_state, mlp_state), dim=-1)
        state_inputs = None
        if record_state_inputs:
            state_inputs = torch.cat((k_input, v_input, attn_input, mlp_input), dim=-1)
        return hidden, tap, new_state, state_inputs


class InklingVisionEncoderLayer(nn.Module):
    """One fold-to-depth projection in Inkling's image HMLP."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        temporal_fold: int,
        spatial_fold: int,
        add_norm: bool,
        eps: float,
    ) -> None:
        super().__init__()
        self.temporal_fold = temporal_fold
        self.spatial_fold = spatial_fold
        self.projection = nn.Linear(
            input_dim, output_dim, bias=False, dtype=torch.bfloat16
        )
        self.norm = RMSNorm(output_dim, eps) if add_norm else None

    def forward(self, hidden: Tensor) -> Tensor:
        batch, time, height, width, channels = hidden.shape
        if (
            time % self.temporal_fold
            or height % self.spatial_fold
            or width % self.spatial_fold
        ):
            raise ValueError("Inkling vision input cannot be folded evenly")
        if self.temporal_fold > 1 or self.spatial_fold > 1:
            hidden = hidden.reshape(
                batch,
                time // self.temporal_fold,
                self.temporal_fold,
                height // self.spatial_fold,
                self.spatial_fold,
                width // self.spatial_fold,
                self.spatial_fold,
                channels,
            )
            hidden = hidden.permute(0, 1, 3, 5, 2, 4, 6, 7)
            hidden = hidden.reshape(
                batch,
                time // self.temporal_fold,
                height // self.spatial_fold,
                width // self.spatial_fold,
                (self.temporal_fold * self.spatial_fold * self.spatial_fold * channels),
            )
        hidden = self.projection(hidden)
        if self.norm is not None:
            hidden = F.gelu(self.norm(hidden))
        return hidden


class InklingVisionModel(nn.Module):
    """Four-layer patch HMLP producing one decoder embedding per image patch."""

    def __init__(self, config: InklingConfig) -> None:
        super().__init__()
        scales = config.vision_scales
        layers = []
        for index, (start, end) in enumerate(zip(scales, scales[1:])):
            temporal_fold = end[0] // start[0]
            spatial_fold = end[1] // start[1]
            fold_size = temporal_fold * spatial_fold * spatial_fold
            output_dim = (
                config.hidden_size
                if index == config.vision_num_hidden_layers - 1
                else end[3]
            )
            layers.append(
                InklingVisionEncoderLayer(
                    start[3] * fold_size,
                    output_dim,
                    temporal_fold=temporal_fold,
                    spatial_fold=spatial_fold,
                    add_norm=index != config.vision_num_hidden_layers - 1,
                    eps=config.rms_norm_eps,
                )
            )
        self.layers = nn.ModuleList(layers)
        self.final_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.input_shape = (
            config.vision_temporal_patch_size,
            config.vision_patch_size,
            config.vision_patch_size,
            config.vision_num_channels,
        )

    def forward(self, pixel_values: Tensor) -> Tensor:
        if pixel_values.ndim != 5 or tuple(pixel_values.shape[1:]) != self.input_shape:
            raise ValueError(
                "Inkling pixel_values must have shape "
                f"[patches, {', '.join(map(str, self.input_shape))}]"
            )
        hidden = pixel_values
        for layer in self.layers:
            hidden = layer(hidden)
        return self.final_norm(hidden).reshape(pixel_values.shape[0], -1)


class InklingTarget(nn.Module):
    def __init__(self, config: InklingConfig, tap_layer_ids: tuple[int, ...]) -> None:
        super().__init__()
        self.config = config
        self.tap_layer_ids = tap_layer_ids
        self.embed = nn.Embedding(
            config.vocab_size, config.hidden_size, dtype=torch.bfloat16
        )
        self.embed_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.layers = nn.ModuleList(
            DecoderLayer(config, index) for index in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.unembed = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False, dtype=torch.bfloat16
        )

    def prepare(self) -> None:
        for layer in cast(Sequence[DecoderLayer], self.layers):
            layer.prepare()

    def embed_tokens(self, token_ids: Tensor) -> Tensor:
        """Apply the shared embedding exactly as target training defines it."""

        return self.embed_norm(self.embed(token_ids))

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
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
        if state.shape != (
            token_ids.shape[0],
            self.config.num_hidden_layers,
            STATE_LENGTH,
            STATE_WIDTH_PER_LAYER,
        ):
            raise ValueError("Inkling target received malformed convolution state")
        if input_embeddings is None:
            hidden = self.embed_tokens(token_ids)
        else:
            expected = (*token_ids.shape, self.config.hidden_size)
            if tuple(input_embeddings.shape) != expected:
                raise ValueError("Inkling input embeddings have the wrong shape")
            hidden = input_embeddings
        taps = []
        next_states = []
        attention_batch = prepare_paged_attention_batch(
            cache,
            sequence_lengths,
            max_sequence_length,
            batch=token_ids.shape[0],
            tokens=token_ids.shape[1],
            layer_counts={GLOBAL_GROUP: GLOBAL_LAYERS, LOCAL_GROUP: LOCAL_LAYERS},
            log_tau=(
                log_scaling_tau(
                    positions,
                    n_floor=self.config.log_scaling_n_floor,
                    alpha=self.config.log_scaling_alpha,
                )
                if max_sequence_length > self.config.log_scaling_n_floor
                else None
            ),
        )
        if state_inputs_out is not None and state_inputs_out.shape != (
            token_ids.shape[0],
            token_ids.shape[1],
            self.config.num_hidden_layers,
            STATE_WIDTH_PER_LAYER,
        ):
            raise ValueError("Inkling convolution input buffer has the wrong shape")
        for index, layer in enumerate(self.layers):
            hidden, tap, layer_state, layer_inputs = layer(
                hidden,
                positions,
                sequence_lengths,
                max_sequence_length,
                cache,
                attention_batch,
                state[:, index],
                prefill=prefill,
                record_state_inputs=state_inputs_out is not None,
            )
            next_states.append(layer_state)
            if state_inputs_out is not None:
                if layer_inputs is None:
                    raise RuntimeError("Inkling layer omitted its convolution inputs")
                state_inputs_out[:, :, index].copy_(layer_inputs)
            if index in self.tap_layer_ids:
                taps.append(tap)
        normalized = self.norm(hidden)
        selected = normalized if full_logits else normalized[:, -1:]
        logits = selected / self.config.logits_mup_width_multiplier
        logits = (
            logits
            if return_hidden
            else self.unembed(logits)[..., : self.config.unpadded_vocab_size]
        )
        target_taps = torch.cat(taps, dim=-1)
        return (
            logits,
            target_taps,
            torch.stack(next_states, dim=1),
            state_inputs_out,
        )


class InklingForCausalLM(nn.Module):
    def __init__(self, config: InklingConfig, tap_layer_ids: tuple[int, ...]) -> None:
        super().__init__()
        self.target = InklingTarget(config, tap_layer_ids)
        self.vision = InklingVisionModel(config)

    def prepare(self) -> None:
        self.target.prepare()


__all__ = [
    "Bf16Experts",
    "DenseMlp",
    "InklingForCausalLM",
    "InklingTarget",
    "InklingVisionEncoderLayer",
    "InklingVisionModel",
    "Nvfp4Experts",
    "Router",
    "ShortConvolution",
    "SparseMlp",
    "select_convolution_state",
]
