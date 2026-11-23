"""Inkling 2 Metal kernels at Mini's real widths against the CUDA CPU references.

The tiny model in test_inkling2_model.py runs every kernel, but at widths that
skip multi-simdgroup reductions, the BF16 GEMV and multi-value-per-thread rows.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "cuda" / "tests"))
from _inkling2_expert_reference import dequant, nvfp4_reference  # noqa: E402

from resin_models_cuda.inkling2 import kernels as reference  # noqa: E402
from resin_models_mlx.fused_norm import rms_norm  # noqa: E402
from resin_models_mlx.inkling2 import kernels  # noqa: E402
from resin_models_mlx.inkling2.quantization import quantize_activations  # noqa: E402

HIDDEN, INTERMEDIATE, KV_WIDTH, EXPERTS = 2304, 1152, 1024, 128


def _mx(x: torch.Tensor) -> mx.array:
    if x.dtype == torch.bfloat16:
        return mx.array(x.view(torch.int16).numpy()).view(mx.bfloat16)
    return mx.array(x.numpy())


def _torch(x: mx.array) -> torch.Tensor:
    if x.dtype == mx.bfloat16:
        return torch.from_numpy(np.array(x.view(mx.int16))).view(torch.bfloat16)
    return torch.from_numpy(np.array(x))


def _within_bf16_step(actual: torch.Tensor, expected: torch.Tensor) -> None:
    # Reductions sum in a different order; allow one BF16 rounding step.
    actual, expected = actual.float(), expected.float()
    step = expected.abs().clamp(min=1e-30) * 2.0**-7
    assert torch.all((actual - expected).abs() <= step)


@pytest.fixture
def generator():
    return torch.Generator().manual_seed(0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rms_norm(generator, dtype):
    x = (torch.randn(7, HIDDEN, generator=generator) * 3).to(dtype)
    weight = 1 + 0.2 * torch.randn(HIDDEN, generator=generator)
    for scale in (weight, None):
        actual = rms_norm(_mx(x), None if scale is None else _mx(scale), 1e-6)
        expected = reference.rms_norm(x, scale, 1e-6, dtype=torch.bfloat16)
        _within_bf16_step(_torch(actual), expected)


@pytest.mark.parametrize("tokens", [1, 2, 5, 16, 17, 35])
def test_short_convolution(generator, tokens):
    x = (torch.randn(3, tokens, HIDDEN, generator=generator) * 2).bfloat16()
    state = torch.randn(3, 3, HIDDEN, generator=generator).bfloat16()
    weight = (torch.randn(HIDDEN, 4, generator=generator) * 0.3).bfloat16()
    actual, history = kernels.short_convolution(_mx(x), _mx(state), _mx(weight))
    expected, expected_history = reference.short_convolution(x, state, weight.float())
    assert torch.equal(_torch(actual), expected)
    assert torch.equal(_torch(history), expected_history)
    # Dynamic: FP32 static taps (tap zero current) plus per-head BF16 taps.
    x = x[..., :KV_WIDTH].contiguous()
    state = state[..., :KV_WIDTH].contiguous()
    static = torch.randn(4, KV_WIDTH, generator=generator) * 0.3
    dynamic = (torch.randn(3, tokens, 8, 4, generator=generator) * 0.3).bfloat16()
    actual, history = kernels.short_convolution(
        _mx(x), _mx(state), _mx(static), _mx(dynamic)
    )
    expected, expected_history = reference.short_convolution(x, state, static, dynamic)
    assert torch.equal(_torch(actual), expected)
    assert torch.equal(_torch(history), expected_history)


@pytest.mark.parametrize(
    "sources,queries,addend", [(1, 1, False), (4, 3, False), (6, 1, True)]
)
def test_depth_mix(generator, sources, queries, addend):
    values = [
        (torch.randn(5, HIDDEN, generator=generator) * (index + 1)).bfloat16()
        for index in range(sources)
    ]
    query = torch.randn(queries, HIDDEN, generator=generator)
    extra = torch.randn(5, HIDDEN, generator=generator).bfloat16() if addend else None
    actual = kernels.depth_mix(
        [_mx(x) for x in values], _mx(query), None if extra is None else _mx(extra)
    )
    expected = torch.stack(
        reference.block_attention_residual(
            values, query, normalize_values=True, partial_addend=extra
        ),
        dim=1,
    )
    torch.testing.assert_close(_torch(actual), expected, rtol=1e-5, atol=1e-5)


def test_route_experts_breaks_ties_by_lower_id(generator):
    logits = torch.randn(9, EXPERTS, generator=generator) * 2
    bias = torch.randn(EXPERTS, generator=generator) * 0.1
    # Exact ties across simdgroups must choose the lower expert ID.
    logits[0] = 0.0
    bias_tied = bias.clone()
    bias_tied[:] = 0.0
    for scores in (bias, bias_tied):
        ids, weights = kernels.route_experts(_mx(logits), _mx(scores), 8, 8.0)
        expected_ids, expected_weights = reference.route_experts(logits, scores, 8, 8.0)
        assert torch.equal(_torch(ids).long(), expected_ids)
        torch.testing.assert_close(
            _torch(weights), expected_weights, rtol=1e-6, atol=1e-7
        )


@pytest.mark.parametrize("width", [HIDDEN, INTERMEDIATE])
def test_activation_quantization(generator, width):
    x = (
        torch.randn(17, width, generator=generator)
        * 2.0 ** torch.randint(-6, 6, (17, 1))
    ).bfloat16()
    x[3] = 0
    packed, scales, amax = nvfp4_reference(x)
    values, row_amax = quantize_activations(_mx(x), "nvfp4")
    assert values.dtype == mx.float32 and row_amax is not None
    assert row_amax.dtype == mx.bfloat16
    assert torch.equal(
        _torch(values).abs(), dequant(packed, scales, torch.ones_like(amax), 1.0).abs()
    )
    assert torch.equal(_torch(row_amax), amax)
    grouped = x.float().reshape(17, -1, 32)
    raw = grouped.abs().amax(-1) * (1.0 / 448.0)
    exponent = ((raw.view(torch.int32) + 0x007FFFFF) & 0x7F800000) >> 23
    exponent = torch.where(raw <= 2.0**-127, 0, exponent.clamp(max=254))
    scale = (exponent << 23).view(torch.float32)
    inverse = torch.where(exponent == 0, torch.zeros_like(scale), 1.0 / scale)
    codes = (
        (grouped * inverse[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn).float()
    )
    values, row_amax = quantize_activations(_mx(x), "mxfp8")
    assert row_amax is None
    assert torch.equal(_torch(values), (codes * scale[..., None]).flatten(1))


def test_nvfp4_row_max_precision():
    # BF16 zero, subnormals, normal exponent boundaries and largest finite value.
    bits = np.array([0, 1, 0x7F, 0x80, 0x81, 0x3F80, 0x7F7F], np.uint32)
    signed_bits = np.broadcast_to(bits[:, None] | 0x8000, (len(bits), 16))
    # Construct both inputs as bits so a dtype conversion cannot flush subnormals.
    x = mx.array(signed_bits.astype(np.uint16)).view(mx.bfloat16)
    x_wide = mx.array(signed_bits << 16).view(mx.float32)
    _, compact = quantize_activations(x, "nvfp4")
    _, wide = quantize_activations(x_wide, "nvfp4")
    assert compact is not None and wide is not None
    assert compact.dtype == mx.bfloat16 and wide.dtype == mx.float32
    compact_bits = np.array(compact.view(mx.uint16)).astype(np.uint32) << 16
    wide_bits = np.array(wide.view(mx.uint32))
    expected = bits << 16
    subnormal = (bits > 0) & (bits < 0x80)
    # Metal can flush subnormal row maxima to positive zero in either dtype.
    for actual in (compact_bits, wide_bits):
        flushed = subnormal & (actual == 0)
        np.testing.assert_array_equal(actual, np.where(flushed, 0, expected))


@pytest.mark.parametrize("rows", [1, 3, 4, 8, 9])
def test_gemv_f32(generator, rows):
    weight = (torch.randn(1000, HIDDEN, generator=generator) * 0.02).bfloat16()
    x = torch.randn(rows, HIDDEN, generator=generator).bfloat16()
    actual = kernels.gemv_f32(_mx(x), _mx(weight))
    torch.testing.assert_close(
        _torch(actual), x.float() @ weight.float().t(), rtol=1e-5, atol=1e-5
    )


def test_expert_activation_and_combine(generator):
    routes, rows, top_k = 24, 3, 8
    gate_up = torch.randn(routes, 2 * INTERMEDIATE, generator=generator) * 3
    gamma = torch.rand(routes, generator=generator) * 2
    activated = kernels.expert_activation(_mx(gate_up), _mx(gamma), 4.0, 25.0)
    rounded = gate_up.bfloat16().float()
    gate, up = rounded[:, 0::2], rounded[:, 1::2]
    expected = (
        ((4 * (gate / 4).tanh()) * gate.sigmoid()) * (25 * (up / 25).tanh())
    ) * gamma[:, None]
    _within_bf16_step(_torch(activated), expected.bfloat16())
    down = torch.randn(routes, HIDDEN, generator=generator)
    inverse = torch.randperm(routes, generator=generator).int()
    combined = kernels.expert_combine(_mx(down), _mx(inverse), top_k)
    routed = down[inverse.long()].bfloat16().float().view(rows, top_k, HIDDEN)
    total = routed[:, 0]
    for slot in range(1, top_k):
        total = total + routed[:, slot]
    assert torch.equal(_torch(combined), total.bfloat16())
