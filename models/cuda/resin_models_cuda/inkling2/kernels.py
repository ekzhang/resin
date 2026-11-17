"""Inkling 2 tensor operations, with independent CPU references for verification."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def rms_norm(x: Tensor, weight: Tensor | None, eps: float, *, dtype=None) -> Tensor:
    dtype = dtype or x.dtype
    if x.is_cuda:
        from resin_models_cuda.inkling2.triton import rms_norm as kernel

        return kernel(x, weight, eps, dtype)
    result = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
    if weight is not None:
        result = result * weight.float()
    return result.to(dtype)


def route_experts(logits: Tensor, bias: Tensor, top_k: int, scale: float):
    """Select with biased sigmoid scores; normalize unbiased probabilities stably."""
    if logits.is_cuda:
        from resin_models_cuda.inkling2.triton import route_experts as kernel

        return kernel(logits, bias, top_k, scale)
    ids = (logits.sigmoid() + bias).argsort(descending=True, stable=True)[..., :top_k]
    probabilities = torch.nn.functional.logsigmoid(logits.gather(-1, ids))
    maximum = probabilities.amax(-1, keepdim=True)
    shifted = (probabilities - maximum).exp()
    if top_k == 8:
        p = shifted.unbind(-1)
        total = ((p[0] + p[2]) + (p[1] + p[3])) + ((p[4] + p[6]) + (p[5] + p[7]))
        total = total[..., None]
    else:
        total = shifted.sum(-1, keepdim=True)
    return ids, (probabilities - (maximum + total.log())).exp() * scale


def l2_clip(
    x: Tensor, scale: Tensor | None, threshold: float, tau: Tensor | None = None
) -> Tensor:
    if x.is_cuda:
        from resin_models_cuda.inkling2.triton import l2_clip as kernel

        return kernel(x, scale, threshold, tau)
    value = x.float() if scale is None else x.float() * scale.float()
    norm = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
    value = value * (threshold / (norm + 1e-6)).clamp(max=1.0)
    # Clip rounds before the optional long-context scale, matching the prologue.
    value = value.to(x.dtype)
    if tau is not None:
        value = (value.float() * tau[..., None, None]).to(x.dtype)
    return value


def short_convolution(
    x: Tensor,
    state: Tensor,
    weight: Tensor,
    dynamic: Tensor | None = None,
    *,
    state_out: Tensor | None = None,
    inputs_out: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Four residual taps; state stores the three oldest-to-newest raw inputs.

    Static weights are [channels, taps] in chronological order, rounded to the
    input dtype. Dynamic weights remain FP32, in [taps, channels] order with tap
    zero the current token, plus [B,T,heads,taps]. Optional destinations receive
    raw history and raw inputs; they must not overlap the input history.
    """
    if x.is_cuda:
        from resin_models_cuda.inkling2.triton import short_convolution as kernel

        return kernel(
            x, state, weight, dynamic, state_out=state_out, inputs_out=inputs_out
        )
    batch, tokens, width = x.shape
    history = torch.cat((state, x), dim=1)
    output = torch.zeros_like(x, dtype=torch.float32)
    for tap in range(4):
        lag = tap if dynamic is not None else 3 - tap
        values = history[:, 3 - lag : 3 - lag + tokens].float()
        if dynamic is None:
            coefficient = weight[:, tap].to(x.dtype).float()
        else:
            heads = dynamic.shape[-2]
            coefficient = weight[tap].float() + dynamic[
                ..., tap
            ].float().repeat_interleave(width // heads, dim=-1)
        # FP64 evaluates the product and sum before one FP32 rounding, matching
        # the CUDA kernel's explicit fma rather than two rounded operations.
        output = (values.double() * coefficient.double() + output.double()).float()
    output = (output + x.float()) * (1.0 / math.sqrt(9 if dynamic is not None else 5))
    tail = history[:, -3:].contiguous()
    if state_out is not None:
        state_out.copy_(tail)
        tail = state_out
    if inputs_out is not None:
        inputs_out.copy_(x)
    return output.to(x.dtype), tail


def block_attention_residual(
    sources: list[Tensor],
    query: Tensor,
    *,
    normalize_values: bool,
    partial_addend: Tensor | None = None,
) -> tuple[Tensor, ...]:
    """Attend to completed residual blocks and the current partial block in FP32."""
    if sources[0].is_cuda:
        from resin_models_cuda.inkling2.triton import block_attention_residual as kernel

        return kernel(sources, query, normalize_values, partial_addend)
    values = torch.stack([x.float() for x in sources], dim=-2)
    if partial_addend is not None:
        values[..., -1, :] += partial_addend.float()
    normalized = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + 1e-6)
    queries = query.float().view(-1, values.shape[-1])
    scores = normalized @ queries.t() * (values.shape[-1] ** -0.5)
    probabilities = scores.softmax(dim=-2).transpose(-1, -2)
    output = probabilities @ (normalized if normalize_values else values)
    return output.unbind(-2)


def residual_add(*inputs: Tensor) -> Tensor:
    if inputs[0].is_cuda:
        from resin_models_cuda.inkling2.triton import residual_add as kernel

        return kernel(inputs)
    result = inputs[0].float()
    for value in inputs[1:]:
        result = result + value.float()
    return result.to(inputs[0].dtype)


def convolution_residual(
    x: Tensor,
    state: Tensor,
    weight: Tensor,
    attention: Tensor,
    partial: Tensor | None = None,
    *,
    state_out: Tensor | None = None,
    inputs_out: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Round expert output, convolve raw history, and update the residual block."""
    if x.is_cuda and attention.dtype == state.dtype == torch.bfloat16:
        from resin_models_cuda.inkling2.triton import short_convolution as kernel

        return kernel(
            x,
            state,
            weight,
            None,
            attention=attention,
            partial=partial,
            state_out=state_out,
            inputs_out=inputs_out,
        )
    convolved, next_state = short_convolution(
        x.to(attention.dtype), state, weight, state_out=state_out, inputs_out=inputs_out
    )
    inputs = (
        (attention, convolved) if partial is None else (partial, attention, convolved)
    )
    return residual_add(*inputs), next_state


def gate_sink(output: Tensor, gate: Tensor, lse: Tensor, sinks: Tensor) -> Tensor:
    """Add the zero-valued sink to the softmax denominator, then gate channels."""
    rows, heads, dim = output.shape
    # FA4 returns [heads, packed queries]; keep its strided transpose as a view.
    statistics = lse.reshape(heads, rows).t()
    if output.is_cuda:
        from resin_models_cuda.inkling2.triton import gate_sink as kernel

        return kernel(output, gate, statistics, sinks)
    result = output.float() * torch.sigmoid(gate.float().view(rows, heads, dim))
    return (result * torch.sigmoid(statistics.float() - sinks.float())[..., None]).to(
        gate.dtype
    )


def select_convolution_state(
    initial: Tensor, inputs: Tensor, accepted: Tensor
) -> Tensor:
    """Recover each request's state at its accepted speculative boundary."""
    if initial.is_cuda:
        from resin_models_cuda.inkling2.triton import select_convolution_state as kernel

        return kernel(initial, inputs, accepted)
    # initial [B,L,3,D], inputs [B,T,L,D]; accepted counts input positions.
    history = torch.cat((initial.transpose(1, 2), inputs), dim=1)
    rows = accepted[:, None] + torch.arange(3, device=accepted.device)[None, :]
    gather = rows[:, :, None, None].expand(-1, -1, inputs.shape[2], inputs.shape[3])
    return history.gather(1, gather).transpose(1, 2).contiguous()


def copy_state_checkpoints(
    storage, addresses, candidates, ends, initial, inputs, features, positions
):
    """Copy exact accepted prefixes from verification history, including DSpark."""
    if storage.is_cuda:
        from resin_models_cuda.inkling2.triton import copy_state_checkpoints as kernel

        return kernel(
            storage, addresses, candidates, ends, initial, inputs, features, positions
        )
    flags = torch.zeros(len(addresses), dtype=torch.bool, device=storage.device)
    history = torch.cat((initial.transpose(1, 2), inputs), dim=1)
    for item, ((row, prefix), (_, _, slab, slot)) in enumerate(
        zip(candidates.tolist(), addresses.tolist(), strict=True)
    ):
        count = prefix - int(positions[row])
        if 0 < count <= inputs.shape[1] and prefix <= int(ends[row]):
            state = history[row, count : count + 3].transpose(0, 1).flatten()
            storage[slab, slot].copy_(torch.cat((state, features[row, count - 1])))
            flags[item] = True
    return flags


__all__ = [
    "rms_norm",
    "route_experts",
    "l2_clip",
    "short_convolution",
    "block_attention_residual",
    "residual_add",
    "convolution_residual",
    "gate_sink",
    "select_convolution_state",
    "copy_state_checkpoints",
]
