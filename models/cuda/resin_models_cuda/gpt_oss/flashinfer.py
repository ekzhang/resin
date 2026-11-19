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
from functools import cached_property

import torch
from torch import Tensor

KERNEL_ID = "flashinfer_cutlass_sm90_mxfp4"
_MXFP4_GROUP_SIZE = 32
_SM90_SCALE_ALIGNMENT = 4
_K_ALIGNMENT = _MXFP4_GROUP_SIZE * _SM90_SCALE_ALIGNMENT
_MAX_TOKENS = 8192
_TOP_K = 4
_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


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


def _pad(
    tensor: Tensor, shape: tuple[int, ...], fill: int = 0, *, gate_up: bool = False
) -> Tensor:
    if not gate_up and tensor.shape == shape:
        return tensor.contiguous()

    padded = tensor.new_full(shape, fill)
    cols = tuple(slice(size) for size in tensor.shape[2:])
    rows = tensor.shape[1]
    if gate_up:
        # Checkpoint rows alternate gate/up. CUTLASS stores all up rows first.
        half, padded_half = rows // 2, shape[1] // 2
        padded[:, :half, *cols].copy_(tensor[:, 1::2])
        padded[:, padded_half : padded_half + half, *cols].copy_(tensor[:, ::2])
    else:
        padded[:, :rows, *cols].copy_(tensor)
    return padded


class FlashInferSM90MXFP4Kernel:
    """Adapter for ``flashinfer.fused_moe.cutlass_fused_moe`` on Hopper."""

    max_tokens = _MAX_TOKENS

    def validate(self, config: MXFP4Config) -> None:
        if config.top_k != _TOP_K:
            raise ValueError("GPT-OSS FlashInfer requires top_k=4")
        if not 4 <= config.num_experts <= 128:
            raise ValueError("GPT-OSS FlashInfer requires 4 to 128 local experts")

    def validate_device(self, device: torch.device) -> None:
        if device.type != "cuda":
            raise ValueError("packed GPT-OSS MXFP4 execution requires a CUDA device")
        major, minor = torch.cuda.get_device_capability(device)
        if (major, minor) != (9, 0):
            raise ValueError(
                f"kernel {KERNEL_ID!r} requires SM90, got sm{major}{minor}"
            )

    @cached_property
    def _fused_moe(self):
        from flashinfer import fused_moe

        return fused_moe

    def _validate_weights(self, weights: MXFP4Weights, config: MXFP4Config) -> None:
        experts, hidden, intermediate = (
            config.num_experts,
            config.hidden_size,
            config.intermediate_size,
        )
        group = _MXFP4_GROUP_SIZE
        shapes = {
            "gate_up": (experts, 2 * intermediate, hidden // 2),
            "down": (experts, hidden, intermediate // 2),
            "gate_up_scales": (experts, 2 * intermediate, hidden // group),
            "down_scales": (experts, hidden, intermediate // group),
            "gate_up_bias": (experts, 2 * intermediate),
            "down_bias": (experts, hidden),
        }
        device = weights.gate_up.device
        self.validate_device(device)
        for name, expected in shapes.items():
            tensor = getattr(weights, name)
            if tensor.shape != expected:
                raise ValueError(
                    f"{name} must have shape {expected}, got {tensor.shape}"
                )
            if tensor.device != device:
                raise ValueError(f"{name} must be on {device}, got {tensor.device}")
            dtypes = _FLOAT_DTYPES if name.endswith("_bias") else (torch.uint8,)
            if tensor.dtype not in dtypes:
                raise ValueError(
                    f"{name} must have dtype in {dtypes}, got {tensor.dtype}"
                )

    def _pack_projection(
        self,
        weight: Tensor,
        scales: Tensor,
        bias: Tensor,
        rows: int,
        width: int,
        *,
        gate_up: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        fused_moe = self._fused_moe
        shape = (weight.shape[0], rows)
        group = _MXFP4_GROUP_SIZE
        # Interleave one padded tensor at a time to bound checkpoint load memory.
        packed = fused_moe.interleave_moe_weights_for_sm90_mixed_gemm(
            _pad(weight, (*shape, width // 2), gate_up=gate_up), "fp4"
        )
        interleaved = fused_moe.interleave_moe_scales_for_sm90_mixed_gemm(
            _pad(scales, (*shape, width // group), 127, gate_up=gate_up), group
        )
        # CUTLASS reads four E8M0 scale bytes at a time. Keep the folded layout.
        packed_scales = interleaved.view(torch.int32)
        packed_bias = _pad(bias.to(dtype=torch.bfloat16), shape, gate_up=gate_up)
        return packed, packed_scales, packed_bias

    def prepare_weights(
        self, weights: MXFP4Weights, config: MXFP4Config
    ) -> PreparedFlashInferMXFP4Weights:
        self.validate(config)
        self._validate_weights(weights, config)
        hidden = _round_up(config.hidden_size, _K_ALIGNMENT)
        intermediate = _round_up(config.intermediate_size, _K_ALIGNMENT)
        fc1, fc1_scales, fc1_bias = self._pack_projection(
            weights.gate_up,
            weights.gate_up_scales,
            weights.gate_up_bias,
            2 * intermediate,
            hidden,
            gate_up=True,
        )
        fc2, fc2_scales, fc2_bias = self._pack_projection(
            weights.down, weights.down_scales, weights.down_bias, hidden, intermediate
        )
        activation = torch.empty(
            config.num_experts, dtype=torch.float32, device=fc1.device
        )
        return PreparedFlashInferMXFP4Weights(
            fc1_weights=fc1,
            fc2_weights=fc2,
            fc1_scales=fc1_scales,
            fc2_scales=fc2_scales,
            fc1_bias=fc1_bias,
            fc2_bias=fc2_bias,
            swiglu_alpha=torch.full_like(activation, 1.702),
            swiglu_beta=torch.ones_like(activation),
            swiglu_limit=activation.fill_(config.swiglu_limit),
            padded_hidden_size=hidden,
            padded_intermediate_size=intermediate,
            config=config,
            fc2_zero_bias=torch.zeros_like(fc2_bias),
        )

    def forward_partial(
        self,
        hidden_states: Tensor,
        selected_experts: Tensor,
        expert_weights: Tensor,
        weights: PreparedFlashInferMXFP4Weights,
    ) -> Tensor:
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
        self._validate_inputs(hidden_states, selected_experts, expert_weights, weights)
        result = hidden_states.new_zeros(hidden_states.shape, dtype=torch.float32)
        active = (
            (selected_experts >= 0)
            & (selected_experts < config.num_experts)
            & (expert_weights != 0)
        )
        # Both bias arguments must exist when FC1 has bias in the pinned ABI.
        # This view shares all packed weights; only the small zero bias is new.
        unbiased = replace(weights, fc2_bias=weights.fc2_zero_bias)
        chunk_rows = _MAX_TOKENS // _TOP_K
        flat_experts = selected_experts.reshape(-1)
        flat_weights = expert_weights.reshape(-1)
        flat_active = active.reshape(-1)
        for first in range(0, hidden_states.shape[0], chunk_rows):
            last = min(first + chunk_rows, hidden_states.shape[0])
            start, stop = first * _TOP_K, last * _TOP_K
            count = stop - start
            token_rows = (
                torch.arange(
                    start, stop, device=hidden_states.device, dtype=torch.int64
                )
                // _TOP_K
            )
            selected_active = flat_active[start:stop]
            experts = torch.where(
                selected_active, flat_experts[start:stop], config.num_experts
            )
            indices = torch.full(
                (count, _TOP_K),
                config.num_experts,
                device=hidden_states.device,
                dtype=torch.int32,
            )
            indices[:, 0] = experts.int()
            scales = torch.zeros(
                indices.shape, device=hidden_states.device, dtype=torch.float32
            )
            scales[:, 0] = selected_active
            output = self._launch(
                hidden_states[token_rows], indices, scales, unbiased
            ).float()
            safe_experts = experts.clamp(0, config.num_experts - 1).long()
            output += weights.fc2_bias[safe_experts, : config.hidden_size].float()
            output *= flat_weights[start:stop, None].float()
            output.masked_fill_(~selected_active[:, None], 0)
            # Keep each token's four contributions together and in route order.
            torch.sum(
                output.view(-1, _TOP_K, config.hidden_size),
                dim=1,
                out=result[first:last],
            )
        return result

    def forward(
        self,
        hidden_states: Tensor,
        selected_experts: Tensor,
        expert_weights: Tensor,
        weights: PreparedFlashInferMXFP4Weights,
    ) -> Tensor:
        self._validate_inputs(hidden_states, selected_experts, expert_weights, weights)
        if not 1 <= hidden_states.shape[0] <= _MAX_TOKENS:
            raise ValueError(f"num_tokens must be between 1 and {_MAX_TOKENS}")
        return self._launch(hidden_states, selected_experts, expert_weights, weights)

    @staticmethod
    def _validate_inputs(
        hidden_states: Tensor,
        selected_experts: Tensor,
        expert_weights: Tensor,
        weights: PreparedFlashInferMXFP4Weights,
    ) -> None:
        hidden = weights.config.hidden_size
        if hidden_states.ndim != 2 or hidden_states.shape[1] != hidden:
            raise ValueError(f"hidden_states must have shape [tokens, {hidden}]")
        if hidden_states.dtype != torch.bfloat16:
            raise ValueError("hidden_states must have dtype bfloat16")
        routing_shape = (hidden_states.shape[0], _TOP_K)
        if (
            selected_experts.shape != routing_shape
            or expert_weights.shape != routing_shape
        ):
            raise ValueError(f"routing tensors must have shape {routing_shape}")
        device = hidden_states.device
        if device != weights.fc1_weights.device or any(
            tensor.device != device for tensor in (selected_experts, expert_weights)
        ):
            raise ValueError(
                "hidden states, routing and expert weights must share a device"
            )
        if selected_experts.dtype not in (torch.int32, torch.int64):
            raise ValueError("selected_experts must have an integer dtype")
        if expert_weights.dtype not in _FLOAT_DTYPES:
            raise ValueError("expert_weights must have a floating dtype")

    def _launch(
        self,
        hidden_states: Tensor,
        selected_experts: Tensor,
        expert_weights: Tensor,
        weights: PreparedFlashInferMXFP4Weights,
    ) -> Tensor:
        config = weights.config
        fused_moe = self._fused_moe
        # Preserve the inactive ID config.num_experts for CUTLASS's range filter.
        # Its W4 prologue stores one row per (token, expert), stopping at the
        # first matching slot. A zero-weight duplicate can hide a real route.
        selected_experts = selected_experts.to(torch.int32).contiguous()
        expert_weights = expert_weights.to(torch.float32).contiguous()
        shape = (hidden_states.shape[0], weights.padded_hidden_size)
        kernel_input = _pad(hidden_states, shape)
        # The pinned FlashInfer wrapper returns a list of auxiliary tensors in
        # normal-latency mode even though its public annotation says Tensor.
        # Supplying our own output buffer gives this adapter a stable contract
        # and avoids depending on that return-value discrepancy.
        output = torch.empty(
            shape,
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
