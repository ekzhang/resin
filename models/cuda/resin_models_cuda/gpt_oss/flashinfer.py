"""GPT-OSS FlashInfer Hopper MXFP4 fused-MoE kernel adapter.

Checkpoint preparation and execution are specific to GPT-OSS: MXFP4 group-32
weights, paired gate/up rows, biased projections, and capped SwiGLU.

FlashInfer's SM90 W4A16 scale layout groups four consecutive MXFP4 groups.
For a width of 2880 (90 groups), preparation pads the hidden and intermediate
dimensions to 2944 (92 groups). Gate/up rows are padded in paired order from
5760 to 5888. The semantic interface stores paired ``[gate, up]`` rows, while the
CUTLASS activation consumes ``[all up, all gate]``; preparation applies that
row permutation consistently to weights, scales, and biases.  FlashInfer's
fused runner uses one hidden size for both input and output, so the second
projection's output rows are padded too; ``forward`` slices the 2944-wide
kernel result back to 2880.  These cold-path details are invisible at the
model-facing kernel interface.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import torch
from torch import Tensor

KERNEL_ID = "flashinfer_cutlass_sm90_mxfp4"
_MXFP4_GROUP_SIZE = 32
_SM90_SCALE_ALIGNMENT = 4
_K_ALIGNMENT = _MXFP4_GROUP_SIZE * _SM90_SCALE_ALIGNMENT
_MAX_TOKENS = 8192


@dataclass(frozen=True, slots=True, kw_only=True)
class MXFP4Config:
    """GPT-OSS expert dimensions after EP and intermediate-channel TP sharding."""

    top_k: int
    num_experts: int
    hidden_size: int
    intermediate_size: int
    swiglu_limit: float

    def __post_init__(self) -> None:
        for name in ("top_k", "num_experts", "hidden_size", "intermediate_size"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.top_k > self.num_experts:
            raise ValueError("top_k cannot exceed num_experts")
        if self.hidden_size % 32 or self.intermediate_size % 32:
            raise ValueError(
                "MXFP4 hidden and intermediate sizes must be divisible by 32"
            )
        if (
            isinstance(self.swiglu_limit, bool)
            or not isinstance(self.swiglu_limit, (int, float))
            or not math.isfinite(self.swiglu_limit)
            or self.swiglu_limit <= 0
        ):
            raise ValueError("swiglu_limit must be finite and positive")

    @classmethod
    def from_model(cls, args):
        return cls(
            top_k=args.num_experts_per_tok,
            num_experts=args.num_local_experts,
            hidden_size=args.hidden_size,
            intermediate_size=args.intermediate_size,
            swiglu_limit=args.swiglu_limit,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class MXFP4Weights:
    """Checkpoint tensors: paired gate/up rows and two input values per byte."""

    gate_up: Tensor
    down: Tensor
    gate_up_scales: Tensor
    down_scales: Tensor
    gate_up_bias: Tensor
    down_bias: Tensor


@dataclass(frozen=True, slots=True)
class PreparedFlashInferMXFP4Weights:
    """One layer's interleaved tensors and dimensions for FlashInfer execution."""

    fc1_weights: Tensor
    fc2_weights: Tensor
    fc1_scales: Tensor
    fc2_scales: Tensor
    fc1_bias: Tensor
    fc2_bias: Tensor
    swiglu_alpha: Tensor
    swiglu_beta: Tensor
    swiglu_limit: Tensor
    padded_hidden_size: int
    padded_intermediate_size: int
    config: MXFP4Config
    fc2_zero_bias: Tensor


def _round_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _shape(tensor: Any) -> tuple[int, ...]:
    try:
        return tuple(tensor.shape)
    except (AttributeError, TypeError) as error:
        raise TypeError("MoE weights and routing values must be tensor-like") from error


def _reinterpret_interleaved_scales(tensor: Any, field: str):
    """Keep FlashInfer's folded (or legacy flat) layout through its ABI view."""

    shape = _shape(tensor)
    if len(shape) not in {3, 5} or shape[-1] % 4:
        raise RuntimeError(
            f"{field} interleaver returned incompatible shape {shape}; "
            "the last dimension must be divisible by four"
        )
    if tensor.dtype != torch.uint8:
        raise RuntimeError(
            f"{field} interleaver returned dtype {tensor.dtype}, expected uint8"
        )
    return tensor.view(torch.int32)


def _copy_into_padded(tensor: Any, shape: tuple[int, ...], fill: int):
    if _shape(tensor) == shape:
        return tensor.contiguous()
    padded = torch.full(
        shape,
        fill_value=fill,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    source_shape = _shape(tensor)
    destination = padded[tuple(slice(0, size) for size in source_shape)]
    destination.copy_(tensor)
    return padded


def _reorder_and_pad_gate_up(tensor: Any, shape: tuple[int, ...], fill: int):
    """Convert paired ``[gate, up]`` rows to CUTLASS ``[up..., gate...]``."""

    source_shape = _shape(tensor)
    if len(source_shape) not in {2, 3} or len(source_shape) != len(shape):
        raise ValueError("gate/up tensors must be rank two or three")
    source_rows = source_shape[1]
    target_rows = shape[1]
    if source_rows % 2 or target_rows % 2:
        raise ValueError("gate/up row counts must be even")
    intermediate = source_rows // 2
    padded_intermediate = target_rows // 2

    padded = torch.full(
        shape,
        fill_value=fill,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    trailing = tuple(slice(0, size) for size in source_shape[2:])
    source_trailing = (slice(None),) * (len(source_shape) - 2)
    up_destination = (slice(None), slice(0, intermediate), *trailing)
    gate_destination = (
        slice(None),
        slice(padded_intermediate, padded_intermediate + intermediate),
        *trailing,
    )
    up_source = (slice(None), slice(1, None, 2), *source_trailing)
    gate_source = (slice(None), slice(0, None, 2), *source_trailing)
    padded[up_destination].copy_(tensor[up_source])
    padded[gate_destination].copy_(tensor[gate_source])
    return padded


class FlashInferSM90MXFP4Kernel:
    """Adapter for ``flashinfer.fused_moe.cutlass_fused_moe`` on H100."""

    max_tokens = _MAX_TOKENS

    def validate(self, config: MXFP4Config, *, num_tokens: int | None = None) -> None:
        if config.top_k != 4:
            raise ValueError("GPT-OSS FlashInfer requires top_k=4")
        if not 4 <= config.num_experts <= 128:
            raise ValueError("GPT-OSS FlashInfer requires 4 to 128 local experts")
        if num_tokens is not None and (
            type(num_tokens) is not int or not 1 <= num_tokens <= self.max_tokens
        ):
            raise ValueError(f"num_tokens must be between 1 and {self.max_tokens}")

    def validate_device(self, device) -> None:
        if device.type != "cuda":
            raise ValueError("packed GPT-OSS MXFP4 execution requires a CUDA device")
        capability = tuple(torch.cuda.get_device_capability(device))
        if capability != (9, 0):
            raise ValueError(
                f"kernel {KERNEL_ID!r} requires SM90, got sm{capability[0]}{capability[1]}"
            )

    def _get_fused_moe(self):
        if not hasattr(self, "_fused_moe"):
            from flashinfer import fused_moe

            self._fused_moe = fused_moe

        required = (
            "ActivationType",
            "cutlass_fused_moe",
            "interleave_moe_scales_for_sm90_mixed_gemm",
            "interleave_moe_weights_for_sm90_mixed_gemm",
        )
        missing = [name for name in required if not hasattr(self._fused_moe, name)]
        if missing:
            raise RuntimeError(
                "FlashInfer does not expose the pinned SM90 MXFP4 API: "
                + ", ".join(missing)
            )
        return self._fused_moe

    def _validate_sm90_tensor(self, tensor: Any, field: str) -> None:
        if not bool(getattr(tensor, "is_cuda", False)):
            raise ValueError(f"{field} must be a CUDA tensor")
        self.validate_device(tensor.device)

    def _expected_shapes(self, config: MXFP4Config) -> dict[str, tuple[int, ...]]:
        experts = config.num_experts
        hidden = config.hidden_size
        intermediate = config.intermediate_size
        group = _MXFP4_GROUP_SIZE
        return {
            "gate_up": (experts, 2 * intermediate, hidden // 2),
            "down": (experts, hidden, intermediate // 2),
            "gate_up_scales": (experts, 2 * intermediate, hidden // group),
            "down_scales": (experts, hidden, intermediate // group),
            "gate_up_bias": (experts, 2 * intermediate),
            "down_bias": (experts, hidden),
        }

    def _validate_weight_field(
        self, field: str, tensor: Any, config: MXFP4Config
    ) -> None:
        expected = self._expected_shapes(config)[field]
        actual = _shape(tensor)
        if actual != expected:
            raise ValueError(f"{field} must have shape {expected}, got {actual}")
        if field in {"gate_up", "down", "gate_up_scales", "down_scales"}:
            if tensor.dtype != torch.uint8:
                raise ValueError(f"{field} must have dtype uint8, got {tensor.dtype}")
        else:
            allowed = {torch.float16, torch.bfloat16, torch.float32}
            if tensor.dtype not in allowed:
                raise ValueError(
                    f"{field} must have a floating dtype, got {tensor.dtype}"
                )
        self._validate_sm90_tensor(tensor, field)

    def _validate_weight_devices(self, tensors: list[tuple[str, Any]]) -> None:
        if not tensors:
            return
        expected = tensors[0][1].device
        for field, tensor in tensors[1:]:
            if tensor.device != expected:
                raise ValueError(
                    f"all expert tensors must share one device; {field} is on "
                    f"{tensor.device}, expected {expected}"
                )

    def prepare_weights(
        self, weights: MXFP4Weights, config: MXFP4Config
    ) -> PreparedFlashInferMXFP4Weights:
        self.validate(config)
        fused_moe = self._get_fused_moe()

        tensors = [
            (field, getattr(weights, field)) for field in self._expected_shapes(config)
        ]
        for field, tensor in tensors:
            self._validate_weight_field(field, tensor, config)
        self._validate_weight_devices(tensors)

        hidden = config.hidden_size
        intermediate = config.intermediate_size
        padded_hidden = _round_up(hidden, _K_ALIGNMENT)
        padded_intermediate = _round_up(intermediate, _K_ALIGNMENT)
        experts = config.num_experts

        # Transform one tensor at a time.  Holding all four padded inputs while
        # an interleaver allocates its output would add several unnecessary
        # gigabytes to GPT-OSS-120B's load-time peak.
        fc1 = _reorder_and_pad_gate_up(
            weights.gate_up,
            (experts, 2 * padded_intermediate, padded_hidden // 2),
            0,
        )
        fc1 = fused_moe.interleave_moe_weights_for_sm90_mixed_gemm(fc1, "fp4")
        fc1_scales = _reorder_and_pad_gate_up(
            weights.gate_up_scales,
            (experts, 2 * padded_intermediate, padded_hidden // _MXFP4_GROUP_SIZE),
            127,
        )
        fc1_scales = fused_moe.interleave_moe_scales_for_sm90_mixed_gemm(
            fc1_scales, _MXFP4_GROUP_SIZE
        )
        fc1_scales = _reinterpret_interleaved_scales(fc1_scales, "gate_up_scales")
        fc2 = _copy_into_padded(
            weights.down,
            (experts, padded_hidden, padded_intermediate // 2),
            0,
        )
        fc2 = fused_moe.interleave_moe_weights_for_sm90_mixed_gemm(fc2, "fp4")
        fc2_scales = _copy_into_padded(
            weights.down_scales,
            (
                experts,
                padded_hidden,
                padded_intermediate // _MXFP4_GROUP_SIZE,
            ),
            127,
        )
        fc2_scales = fused_moe.interleave_moe_scales_for_sm90_mixed_gemm(
            fc2_scales, _MXFP4_GROUP_SIZE
        )
        fc2_scales = _reinterpret_interleaved_scales(fc2_scales, "down_scales")

        device = weights.gate_up.device
        fc1_bias = weights.gate_up_bias.to(
            device=device, dtype=torch.bfloat16
        ).contiguous()
        fc1_bias = _reorder_and_pad_gate_up(
            fc1_bias, (experts, 2 * padded_intermediate), 0
        )
        fc2_bias = weights.down_bias.to(
            device=device, dtype=torch.bfloat16
        ).contiguous()
        fc2_bias = _copy_into_padded(fc2_bias, (experts, padded_hidden), 0)

        activation_shape = (experts,)
        alpha = torch.full(
            activation_shape,
            fill_value=1.702,
            dtype=torch.float32,
            device=device,
        )
        beta = torch.full(
            activation_shape,
            fill_value=1.0,
            dtype=torch.float32,
            device=device,
        )
        limit = torch.full(
            activation_shape,
            fill_value=float(config.swiglu_limit),
            dtype=torch.float32,
            device=device,
        )
        return PreparedFlashInferMXFP4Weights(
            fc1_weights=fc1,
            fc2_weights=fc2,
            fc1_scales=fc1_scales,
            fc2_scales=fc2_scales,
            fc1_bias=fc1_bias,
            fc2_bias=fc2_bias,
            swiglu_alpha=alpha,
            swiglu_beta=beta,
            swiglu_limit=limit,
            padded_hidden_size=padded_hidden,
            padded_intermediate_size=padded_intermediate,
            config=config,
            fc2_zero_bias=torch.full(
                (experts, padded_hidden),
                fill_value=0,
                dtype=torch.bfloat16,
                device=device,
            ),
        )

    def forward_partial(
        self,
        hidden_states: Tensor,
        selected_experts: Tensor,
        expert_weights: Tensor,
        weights: PreparedFlashInferMXFP4Weights,
    ):
        """Keep weighted expert sums in FP32 until EP has combined all owners.

        The pinned W4 kernel returns BF16. Give each assignment its own row
        with unit weight and zero output bias, preserving its BF16 GEMM
        result. Apply output bias and routing weights in FP32 afterward,
        matching CUTLASS's finalizer order without rounding an owner sum.

        Expansion and chunk sizes depend only on the input shape. Inactive
        assignments keep the sentinel expert ID, so padded EP exchanges can
        replay this operation in a CUDA graph without reading routing counts
        back to the host or allocating tensors with data-dependent shapes.
        """

        config = weights.config
        self.validate(config)
        result = hidden_states.new_zeros(hidden_states.shape, dtype=torch.float32)
        active = (
            (selected_experts >= 0)
            & (selected_experts < config.num_experts)
            & (expert_weights != 0)
        )
        # Both bias arguments must exist when FC1 has bias in the pinned ABI.
        # This view shares all packed weights; only the small zero bias is new.
        unbiased = replace(weights, fc2_bias=weights.fc2_zero_bias)
        assignments = hidden_states.shape[0] * config.top_k
        limit = self.max_tokens
        flat_experts = selected_experts.reshape(-1)
        flat_weights = expert_weights.reshape(-1)
        flat_active = active.reshape(-1)
        for start in range(0, assignments, limit):
            stop = min(start + limit, assignments)
            count = stop - start
            token_rows = (
                torch.arange(
                    start, stop, device=hidden_states.device, dtype=torch.int64
                )
                // config.top_k
            )
            selected_active = flat_active[start:stop]
            experts = torch.where(
                selected_active, flat_experts[start:stop], config.num_experts
            )
            indices = torch.full(
                (count, config.top_k),
                config.num_experts,
                device=hidden_states.device,
                dtype=torch.int32,
            )
            indices[:, 0] = experts.int()
            scales = torch.zeros(
                indices.shape, device=hidden_states.device, dtype=torch.float32
            )
            scales[:, 0] = selected_active
            output = self.forward(
                hidden_states[token_rows], indices, scales, unbiased
            ).float()
            safe_experts = experts.clamp(0, config.num_experts - 1).long()
            output += weights.fc2_bias[safe_experts, : config.hidden_size].float()
            output *= flat_weights[start:stop, None].float()
            output.masked_fill_(~selected_active[:, None], 0)
            # Assignments are contiguous in token/top-k order. Reduce each
            # token's segment in that order instead of racing FP32 atomics;
            # a chunk may begin or end partway through its four assignments.
            first, last = start // config.top_k, (stop - 1) // config.top_k + 1
            if start % config.top_k == 0 and count % config.top_k == 0:
                torch.sum(
                    output.view(-1, config.top_k, config.hidden_size),
                    dim=1,
                    out=result[first:last],
                )
                continue
            boundaries = (
                torch.arange(first, last + 1, device=hidden_states.device)
                * config.top_k
            ).clamp(start, stop)
            summed = torch.segment_reduce(  # ty: ignore[unresolved-attribute]  # Missing in Torch stubs.
                output, "sum", lengths=boundaries.diff(), unsafe=True
            )
            result[first:last].add_(summed)
        return result

    def forward(
        self,
        hidden_states: Tensor,
        selected_experts: Tensor,
        expert_weights: Tensor,
        weights: PreparedFlashInferMXFP4Weights,
    ):
        config = weights.config
        tokens_shape = _shape(hidden_states)
        if len(tokens_shape) != 2:
            raise ValueError(
                f"hidden_states must have shape [tokens, hidden], got {tokens_shape}"
            )
        num_tokens, hidden_size = tokens_shape
        self.validate(config, num_tokens=num_tokens)
        if hidden_size != config.hidden_size:
            raise ValueError(
                f"hidden_states width must be {config.hidden_size}, got {hidden_size}"
            )

        fused_moe = self._get_fused_moe()
        self._validate_sm90_tensor(hidden_states, "hidden_states")
        if hidden_states.dtype != torch.bfloat16:
            raise ValueError(
                f"hidden_states must have dtype bfloat16, got {hidden_states.dtype}"
            )
        if hidden_states.device != weights.fc1_weights.device:
            raise ValueError(
                "hidden_states and prepared expert weights must share a device"
            )

        expected_routing = {
            "selected_experts": (num_tokens, config.top_k),
            "expert_weights": (num_tokens, config.top_k),
        }
        routing_values = {
            "selected_experts": selected_experts,
            "expert_weights": expert_weights,
        }
        for field, expected in expected_routing.items():
            tensor = routing_values[field]
            actual = _shape(tensor)
            if actual != expected:
                raise ValueError(f"{field} must have shape {expected}, got {actual}")
            if tensor.device != hidden_states.device:
                raise ValueError(f"{field} must be on {hidden_states.device}")
        if selected_experts.dtype not in {torch.int32, torch.int64}:
            raise ValueError("selected_experts must have an integer dtype")
        floating = {torch.float16, torch.bfloat16, torch.float32}
        if expert_weights.dtype not in floating:
            raise ValueError("expert_weights must have a floating dtype")

        # Preserve the inactive ID config.num_experts for CUTLASS's range filter.
        # Its W4 prologue stores one row per (token, expert), stopping at the
        # first matching slot. A zero-weight duplicate can hide a real route.
        selected_experts = selected_experts.to(
            device=hidden_states.device, dtype=torch.int32
        ).contiguous()
        expert_weights = expert_weights.to(
            device=hidden_states.device, dtype=torch.float32
        ).contiguous()

        kernel_input = _copy_into_padded(
            hidden_states,
            (num_tokens, weights.padded_hidden_size),
            0,
        )
        # The pinned FlashInfer wrapper returns a list of auxiliary tensors in
        # normal-latency mode even though its public annotation says Tensor.
        # Supplying our own output buffer gives this adapter a stable contract
        # and avoids depending on that return-value discrepancy.
        output = torch.empty(
            (num_tokens, weights.padded_hidden_size),
            dtype=torch.bfloat16,
            device=hidden_states.device,
        )
        fused_moe.cutlass_fused_moe(
            input=kernel_input,
            token_selected_experts=selected_experts,
            token_final_scales=expert_weights,
            fc1_expert_weights=weights.fc1_weights,
            fc2_expert_weights=weights.fc2_weights,
            output_dtype=torch.bfloat16,
            output=output,
            quant_scales=[weights.fc1_scales, weights.fc2_scales],
            fc1_expert_biases=weights.fc1_bias,
            fc2_expert_biases=weights.fc2_bias,
            swiglu_alpha=weights.swiglu_alpha,
            swiglu_beta=weights.swiglu_beta,
            swiglu_limit=weights.swiglu_limit,
            use_w4_group_scaling=True,
            use_packed_weights=False,
            tune_max_num_tokens=_MAX_TOKENS,
            activation_type=fused_moe.ActivationType.Swiglu,
            use_fused_finalize=True,
        )
        output_shape = _shape(output)
        expected_kernel_output = (num_tokens, weights.padded_hidden_size)
        if output_shape != expected_kernel_output:
            raise RuntimeError(
                "FlashInfer returned shape "
                f"{output_shape}, expected {expected_kernel_output}"
            )
        if output.dtype != torch.bfloat16:
            raise RuntimeError(
                f"FlashInfer returned dtype {output.dtype}, expected bfloat16"
            )
        if weights.padded_hidden_size != config.hidden_size:
            output = output[:, : config.hidden_size].contiguous()
        return output


__all__ = [
    "KERNEL_ID",
    "MXFP4Config",
    "MXFP4Weights",
    "FlashInferSM90MXFP4Kernel",
    "PreparedFlashInferMXFP4Weights",
]
