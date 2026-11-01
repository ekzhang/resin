# Checkpoint expert storage adapted from DeepSeek-V4-Flash-0731 (MIT).
"""Prepare V4.1 checkpoint tensors for the selected complete FFN backend."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from functools import partial
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from resin.backend.cuda.parallel import TensorParallelContext, TensorParallelLayout
from resin.backend.expert_parallel import ExpertParallelDispatch
from resin.backend.topology import ParallelGroups
from resin_models_cuda.deepseek_v41.gemm import fp4_fused_moe
from resin_models_cuda.deepseek_v41.kernels import act_quant
from resin_models_cuda.deepseek_v41.linear import weight_scale_rows
from resin_models_cuda.mega_moe import MegaMoe, prepare_weights


class ExpertProjection(nn.Module):
    """Checkpoint-shaped dense or packed FP4 expert projection."""

    def __init__(self, input_width, output_width, *, dtype=None, storage=None):
        super().__init__()
        dtype = dtype or torch.bfloat16
        packed = dtype == torch.float4_e2m1fn_x2
        shape = (output_width, input_width // 2 if packed else input_width)
        weight = torch.empty(shape, dtype=dtype) if storage is None else storage[0]
        if tuple(weight.shape) != shape or weight.dtype != dtype:
            raise ValueError("invalid shared expert weight storage")
        self.weight = nn.Parameter(weight)
        if packed:
            scale_shape = (output_width, input_width // 32)
            scale = (
                torch.empty(scale_shape, dtype=torch.float8_e8m0fnu)
                if storage is None
                else storage[1]
            )
            if tuple(scale.shape) != scale_shape or scale.dtype != torch.float8_e8m0fnu:
                raise ValueError("invalid shared FP4 scale storage")
            self.scale = nn.Parameter(scale)
            setattr(self.weight, "scale", self.scale)
        else:
            self.register_parameter("scale", None)

    def forward(self, hidden):
        if self.scale is not None:
            raise RuntimeError(
                "packed expert weights must be prepared before inference"
            )
        return F.linear(hidden.to(self.weight.dtype), self.weight)


class Expert(nn.Module):
    """Named checkpoint projections backed by shared routed-expert buffers."""

    def __init__(self, dim, intermediate, *, dtype=None, swiglu_limit=0, storage=None):
        super().__init__()
        blocks = (None, None, None) if storage is None else storage
        self.w1 = ExpertProjection(dim, intermediate, dtype=dtype, storage=blocks[0])
        self.w2 = ExpertProjection(intermediate, dim, dtype=dtype, storage=blocks[1])
        self.w3 = ExpertProjection(dim, intermediate, dtype=dtype, storage=blocks[2])
        self.swiglu_limit = swiglu_limit

    def forward(self, hidden, weights=None):
        gate, up = self.w1(hidden).float(), self.w3(hidden).float()
        if self.swiglu_limit:
            gate = gate.clamp(max=self.swiglu_limit)
            up = up.clamp(-self.swiglu_limit, self.swiglu_limit)
        value = F.silu(gate) * up
        if weights is not None:
            value = weights * value
        return self.w2(value.to(hidden.dtype))


class MoEWeights(nn.Module):
    """Checkpoint-shaped gate, shared expert and rank-local routed storage."""

    _fused_w13_weight: torch.Tensor
    _fused_w13_scale: torch.Tensor
    _fused_w2_weight: torch.Tensor
    _fused_w2_scale: torch.Tensor
    _fused_output_scales: torch.Tensor
    _fused_clamp_limits: torch.Tensor

    def __init__(
        self,
        layer_id: int,
        args,
        gate,
        shared_experts,
        *,
        expert_parallel: TensorParallelLayout | None = None,
        expert_tensor_parallel: TensorParallelLayout | None = None,
    ):
        super().__init__()
        expert_parallel = expert_parallel or TensorParallelContext().layout
        expert_tensor_parallel = expert_tensor_parallel or TensorParallelLayout()
        self.layer_id = layer_id
        self.dim = args.dim
        owned = expert_parallel.even_shard(
            args.n_routed_experts, name="n_routed_experts"
        )
        self.expert_tensor_parallel = expert_tensor_parallel
        self.expert_intermediate_shard = args.expert_intermediate_shard(
            expert_tensor_parallel
        )
        self.routed_intermediate_size = args.moe_inter_dim
        intermediate = self.expert_intermediate_shard.size
        self.n_routed_experts = args.n_routed_experts
        self.n_local_experts = owned.size
        self.n_activated_experts = args.n_activated_experts
        self.swiglu_limit = args.swiglu_limit
        self.decode_overlap_stream: torch.cuda.Stream | None = None
        self.experts_start_idx = owned.start
        self.experts_end_idx = owned.stop
        self.gate = gate
        self.gate.expert_parallel_size = expert_parallel.size
        expert_dtype = torch.float4_e2m1fn_x2 if args.expert_dtype == "fp4" else None
        self._fused_fp4 = expert_dtype == torch.float4_e2m1fn_x2
        if self._fused_fp4:
            self.register_buffer(
                "_fused_w13_weight",
                torch.empty(
                    self.n_local_experts,
                    2 * intermediate,
                    args.dim // 2,
                    dtype=expert_dtype,
                ),
                persistent=False,
            )
            self.register_buffer(
                "_fused_w13_scale",
                torch.empty(
                    self.n_local_experts,
                    2 * intermediate,
                    args.dim // 32,
                    dtype=torch.float8_e8m0fnu,
                ),
                persistent=False,
            )
            self.register_buffer(
                "_fused_w2_weight",
                torch.empty(
                    self.n_local_experts,
                    args.dim,
                    intermediate // 2,
                    dtype=expert_dtype,
                ),
                persistent=False,
            )
            self.register_buffer(
                "_fused_w2_scale",
                torch.empty(
                    self.n_local_experts,
                    args.dim,
                    intermediate // 32,
                    dtype=torch.float8_e8m0fnu,
                ),
                persistent=False,
            )
            self.register_buffer(
                "_fused_output_scales",
                torch.ones(self.n_local_experts, dtype=torch.float32),
                persistent=False,
            )
            self.register_buffer(
                "_fused_clamp_limits",
                torch.full(
                    (self.n_local_experts,),
                    args.swiglu_limit,
                    dtype=torch.float32,
                ),
                persistent=False,
            )

        def expert_storage(expert_id: int):
            if not self._fused_fp4:
                return None
            local = expert_id - self.experts_start_idx
            split = intermediate
            return (
                (
                    self._fused_w13_weight[local, split:],
                    self._fused_w13_scale[local, split:],
                ),
                (
                    self._fused_w2_weight[local],
                    self._fused_w2_scale[local],
                ),
                (
                    self._fused_w13_weight[local, :split],
                    self._fused_w13_scale[local, :split],
                ),
            )

        # ModuleList supports None placeholders at runtime for unowned experts.
        self.experts = cast(
            Sequence[Expert | None],
            nn.ModuleList(
                cast(
                    Iterable[nn.Module],
                    [
                        Expert(
                            args.dim,
                            intermediate,
                            dtype=expert_dtype,
                            swiglu_limit=args.swiglu_limit,
                            storage=expert_storage(i),
                        )
                        if self.experts_start_idx <= i < self.experts_end_idx
                        else None
                        for i in range(self.n_routed_experts)
                    ],
                )
            ),
        )
        assert args.n_shared_experts == 1
        self.shared_experts = shared_experts

    def forward(self, *args, **kwargs):
        raise RuntimeError("routed expert weights must be prepared before inference")


class MoE(nn.Module):
    """Architectural gating, routed execution and shared-expert composition."""

    def __init__(self, source: MoEWeights, routed: nn.Module):
        super().__init__()
        self.dim = source.dim
        self.gate = source.gate
        self.shared_experts = source.shared_experts
        self.routed = routed
        self.decode_overlap_stream = source.decode_overlap_stream

    def forward_empty(self):
        hidden = self.gate.weight.new_empty((0, self.dim))
        self.routed(
            hidden,
            torch.empty((0, self.gate.topk), dtype=torch.int32, device=hidden.device),
            hidden.new_empty((0, self.gate.topk)),
        )

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor | None) -> torch.Tensor:
        shape = x.size()
        x = x.view(-1, self.dim)
        main_stream = None
        shared = None
        if self.decode_overlap_stream is not None:
            # The shared expert does not depend on routing, so decode can hide
            # it behind the routed experts and their tensor-parallel reduction.
            main_stream = torch.cuda.current_stream(x.device)
            self.decode_overlap_stream.wait_stream(main_stream)
            with torch.cuda.stream(self.decode_overlap_stream):
                shared = self.shared_experts(x)

        weights, indices = self.gate(
            x, None if input_ids is None else input_ids.flatten()
        )
        y = self.routed(x, indices, weights)
        if main_stream is None:
            shared = self.shared_experts(x)
        else:
            assert self.decode_overlap_stream is not None
            main_stream.wait_stream(self.decode_overlap_stream)
        assert shared is not None
        y += shared
        return y.type_as(x).view(shape)


class RoutedExperts(nn.Module):
    """Execute a prepared expert shard and return complete local token rows.

    Replicated rows reduce expert-shard outputs. The
    explicit mesh dispatches local rows to their expert owner and all of its
    expert-TP lanes, then combines contributions back into complete origin rows.
    """

    w13_weight: Tensor
    w13_scale: Tensor
    w2_weight: Tensor
    w2_scale: Tensor
    output_scales: Tensor
    clamp_limits: Tensor

    def __init__(
        self,
        source: MoEWeights,
        *,
        reduce: Callable[[Tensor], Tensor],
        block_size: int,
        scale_fmt: str | None,
        scale_dtype: torch.dtype,
        groups: ParallelGroups | None = None,
    ) -> None:
        super().__init__()
        self.num_experts = source.n_routed_experts
        self.start = source.experts_start_idx
        self.stop = source.experts_end_idx
        self.reduce = reduce
        self.block_size = block_size
        self.scale_fmt = scale_fmt
        self.scale_dtype = scale_dtype
        self.fused = source._fused_fp4
        self.dispatch = (
            ExpertParallelDispatch(
                self.local,
                num_experts=self.num_experts,
                groups=groups,
                # MLA gathers CP query outputs before FFN, so these rows are
                # identical across every rank of one attention-DP replica.
                replication=groups.device_replica,
            )
            if groups is not None
            else None
        )
        if self.fused:
            from resin_models_cuda.deepseek_v41.gemm import (
                prepare_fp4_fused_moe_weights,
            )

            # Adopt the loaded storage; checkpoint Expert wrappers are aliases
            # into these buffers and need not survive runtime preparation.
            for name in ("w13_weight", "w13_scale", "w2_weight", "w2_scale"):
                self.register_buffer(name, getattr(source, f"_fused_{name}"))
            prepare_fp4_fused_moe_weights(
                self.w13_weight, self.w13_scale, self.w2_weight, self.w2_scale
            )
            self.register_buffer("output_scales", source._fused_output_scales)
            self.register_buffer("clamp_limits", source._fused_clamp_limits)
        else:
            self.experts = nn.ModuleList(
                cast("Expert", expert)
                for expert in source.experts[self.start : self.stop]
            )

    def forward(self, hidden: Tensor, indices: Tensor, weights: Tensor) -> Tensor:
        if self.dispatch is not None:
            return self.dispatch(hidden, indices, weights)
        return self.reduce(self.local(hidden, indices, weights))

    def local(self, hidden, indices, weights):
        if not hidden.shape[0]:
            return torch.zeros_like(
                hidden, dtype=torch.float32 if self.dispatch else hidden.dtype
            )
        quantized, scales = act_quant(hidden, 32, "ue8m0", torch.float8_e8m0fnu)
        partial = fp4_fused_moe(
            quantized,
            scales.view(torch.float8_e4m3fn),
            indices,
            # FlashInfer's routed-MoE ABI requires BF16 routing weights;
            # keep the model/transport contract in FP32 until this boundary.
            weights.bfloat16(),
            self.w13_weight,
            self.w13_scale,
            self.w2_weight,
            self.w2_scale,
            self.output_scales,
            self.clamp_limits,
            num_experts=self.num_experts,
            local_expert_offset=self.start,
            scale_dtype=torch.float8_e8m0fnu,
        )
        return partial.float() if self.dispatch else partial


def mega_gate(gate, hidden, input_ids, active, out, *, ep_rank):
    """Fuse the released gate into the FFN workspace, retaining BF16 weights."""
    import deep_gemm

    routing = gate.balanced_routing
    image_mask = (
        input_ids == gate.image_token_id
        if input_ids is not None and gate.bias_vl is not None
        else None
    )
    deep_gemm.bf16_mega_gate(
        hidden,
        gate.weight,
        gate.topk,
        False,
        0,
        gate.route_scale,
        ep_rank,
        mask=active,
        bias=gate.bias,
        image_bias=gate.bias_vl if image_mask is not None else None,
        image_token_mask=image_mask,
        fix_routing_mask=(
            torch.ones(len(hidden), dtype=torch.bool, device=hidden.device)
            if routing is not None
            else None
        ),
        unmapped_topk_idx=(
            routing.indices(
                gate.weight[:, 0].expand(len(hidden), -1),
                gate.topk,
            )
            if routing is not None
            else None
        ),
        out=out,
    )


@torch.no_grad()
def prepare_mega_moe(source, collective):
    import deep_gemm

    gate = source.gate
    if (
        gate.temperature != 1
        or gate.score_func != "sqrtsoftplus"
        or not gate.normalized
        or gate.topk < 2
    ):
        raise ValueError(
            "DeepGEMM requires the normalized sqrtsoftplus gate with temperature 1 and top-k > 1"
        )
    intermediate = source.routed_intermediate_size
    # The checkpoint container shares FlashInfer's [up | gate] backing buffer.
    # DeepGEMM accepts [gate | up] before its eight-row interleaving transform.
    routed_weight = torch.cat(
        (
            source._fused_w13_weight[:, intermediate:].view(torch.int8),
            source._fused_w13_weight[:, :intermediate].view(torch.int8),
        ),
        dim=1,
    )
    routed_scale = torch.cat(
        (
            source._fused_w13_scale[:, intermediate:].float(),
            source._fused_w13_scale[:, :intermediate].float(),
        ),
        dim=1,
    )
    routed = cast(
        tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]],
        deep_gemm.transform_weights_for_mega_moe(
            prepare_weights(routed_weight, routed_scale),
            prepare_weights(
                source._fused_w2_weight.view(torch.int8), source._fused_w2_scale
            ),
        ),
    )
    shared = source.shared_experts
    shared_weight = torch.cat((shared.w1.weight, shared.w3.weight))
    shared_scale = torch.cat(
        (weight_scale_rows(shared.w1), weight_scale_rows(shared.w3))
    )
    shared_weights = cast(
        tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]],
        deep_gemm.transform_weights_for_mega_moe(
            prepare_weights(shared_weight, shared_scale),
            prepare_weights(shared.w2.weight, weight_scale_rows(shared.w2)),
        ),
    )
    return MegaMoe(
        source.gate,
        collective,
        lambda hidden: act_quant(hidden, 32, "ue8m0", torch.float8_e8m0fnu),
        route=partial(mega_gate, gate, ep_rank=collective.group.local_rank),
        gate_up=routed[0],
        down=routed[1],
        shared_gate_up=shared_weights[0],
        shared_down=shared_weights[1],
        experts=source.n_routed_experts,
        topk=source.n_activated_experts,
        hidden=source.dim,
        intermediate=intermediate,
        clamp=source.swiglu_limit,
    )
