"""Fixed target projection shapes, original layouts and owned graph views."""

import gc
import weakref
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from resin_models_cuda.glm5_3_flash.model import DenseMLP, Router, SparseMLP
from resin_models_cuda.glm5_3_flash.projections import bmm, fixed_rows, linear


def test_original_policy_preserves_exact_object_and_layout():
    value = torch.randn(3, 16)[:, ::2]
    seen = []

    def operation(x):
        seen.append(x)
        return x

    assert fixed_rows(value, operation) is value
    assert seen == [value] and seen[0].stride() == (16, 2)


@pytest.mark.parametrize("rows", [0, 1, 7, 64])
def test_fixed_rows_pads_bounds_and_retains_output_owner(rows):
    value = torch.arange(rows * 24).reshape(rows, 24).float()[:, ::2]
    calls = []

    def operation(x):
        result = x + 3
        calls.append((x.clone(), result))
        return result

    actual = fixed_rows(value, operation, projection_rows=64)
    assert torch.equal(actual, value + 3)
    assert len(calls) == 1
    if rows:
        padded, full = calls[0]
        assert padded.shape == (64, 12)
        assert torch.equal(padded[:rows], value)
        assert not padded[rows:].count_nonzero()
        assert actual.untyped_storage().data_ptr() == full.untyped_storage().data_ptr()
        owner = weakref.ref(full)
        calls.clear()
        del full
        gc.collect()
        assert owner() is not None and torch.equal(actual, value + 3)
    else:
        assert calls[0][0].shape == (0, 12)


@pytest.mark.parametrize("rows", [-1, True, 1.0])
def test_bad_projection_capacity_rejected_before_operation(rows):
    with pytest.raises(ValueError, match="projection_rows"):
        fixed_rows(torch.ones(3, 8), lambda _: None, projection_rows=rows)


def test_capacity_and_layout_errors_fail_closed():
    value = torch.ones(7, 8)
    with pytest.raises(ValueError, match="capacity"):
        fixed_rows(value, lambda _: None, projection_rows=6)
    with pytest.raises(ValueError, match="overlaps"):
        fixed_rows(value, lambda _: None, projection_rows=64, input_strides=(1, 1))
    with pytest.raises(ValueError, match="offset"):
        fixed_rows(value, lambda _: None, projection_rows=64, input_storage_offset=-1)
    with pytest.raises(ValueError, match="axis"):
        fixed_rows(value, lambda _: None, projection_rows=64, token_axis=2)


def _bmm_parent(kind, rows, device, dtype):
    if kind == "kda":
        packed = torch.empty(rows, 24896, device=device, dtype=dtype)
        return (
            packed,
            packed[:, -256:].reshape(rows, 2, 128).transpose(0, 1),
            (128, 24896, 1),
            24640,
        )
    groups, inner = (64, 256) if kind == "mla_key" else (64, 512)
    packed = torch.empty(rows, groups, inner, device=device, dtype=dtype)
    return packed, packed.transpose(0, 1), (inner, groups * inner, 1), 0


def _consumer(kind, value):
    if kind == "kda":
        result = value.unbind(0)
        assert all(x.is_contiguous() for x in result)
        return result
    if kind == "mla_key":
        return value.transpose(0, 1).contiguous()
    flat = value.transpose(0, 1).reshape(value.shape[1], -1)
    assert flat.is_contiguous()
    return flat


def _equal(actual, expected):
    if isinstance(actual, tuple):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected, strict=True):
            _equal(a, b)
    else:
        assert torch.equal(actual, expected)


@pytest.mark.parametrize("kind", ["kda", "mla_key", "mla_value"])
@pytest.mark.parametrize("rows", [1, 7, 64])
def test_bmm_offset_stride_and_original_consumers(kind, rows, monkeypatch):
    parent, value, strides, offset = _bmm_parent(kind, rows, "cpu", torch.float32)
    parent.copy_(torch.arange(parent.numel()).reshape(parent.shape) % 5)
    weight = torch.ones(value.shape[0], value.shape[-1], 7)
    observed = []
    original = torch.bmm

    def observe(a, b):
        observed.append((a.shape, a.stride(), a.storage_offset()))
        return original(a, b)

    monkeypatch.setattr(torch, "bmm", observe)
    actual = bmm(
        value,
        weight,
        projection_rows=64,
        input_strides=strides,
        input_storage_offset=offset,
    )
    assert observed == [
        (torch.Size((value.shape[0], 64, value.shape[2])), strides, offset)
    ]
    _, reference, _, _ = _bmm_parent(kind, 64, "cpu", torch.float32)
    reference.zero_()
    reference[:, :rows].copy_(value)
    expected = original(reference, weight)[:, :rows]
    _equal(actual, expected)
    _equal(_consumer(kind, actual), _consumer(kind, expected))


def test_router_padding_crops_before_one_expert_exchange():
    config = SimpleNamespace(
        num_experts_per_tok=3, norm_topk_prob=True, routed_scaling_factor=1.25
    )
    module = SparseMLP.__new__(SparseMLP)
    nn.Module.__init__(module)
    module.router = Router(config, torch.randn(11, 16), torch.randn(11))
    calls = []

    class Experts(nn.Module):
        def forward(self, hidden, ids, weights):
            calls.append((hidden, ids, weights))
            return torch.zeros_like(hidden)

    module.experts = Experts()
    module.dispatch = None
    module.shared = DenseMLP(
        {
            name + ".weight": torch.randn(shape)
            for name, shape in [
                ("gate_proj", (12, 16)),
                ("up_proj", (12, 16)),
                ("down_proj", (16, 12)),
            ]
        },
        10,
    )
    hidden = torch.randn(12, 16)
    result = module(hidden, projection_rows=64)
    assert len(calls) == 1 and calls[0][0] is hidden
    padded = F.pad(hidden, (0, 0, 0, 64 - hidden.shape[0]))
    for actual, expected in zip(calls[0][1:], module.router(padded), strict=True):
        assert torch.equal(actual, expected[: len(hidden)])
    assert torch.equal(result, module.shared(padded)[: len(hidden)])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("projection_rows", [64, 96])
@torch.inference_mode()
def test_gpu_fixed_rows_changed_input_graph(dtype, projection_rows):
    torch.manual_seed(2743)
    weight = torch.randn(1536, 4096, device="cuda", dtype=dtype) * 0.02
    full = torch.randn(projection_rows, 4096, device="cuda", dtype=dtype)
    expected = F.linear(full, weight)
    for rows in (1, 7, projection_rows):
        actual = linear(full[:rows], weight, projection_rows=projection_rows)
        assert torch.equal(actual, expected[:rows])
    live_rows = projection_rows // 8 * 6  # MTP5 verification for this batch bucket.
    static = full[:live_rows].clone()
    for _ in range(3):
        linear(static, weight, projection_rows=projection_rows)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = linear(static, weight, projection_rows=projection_rows)
    gc.collect()
    pressure = torch.empty(16 * 1024**2, device="cuda", dtype=torch.uint8).fill_(17)
    del pressure
    for scale in (0.5, -0.75):
        static.copy_(static.roll(1, 0) * scale)
        graph.replay()
        torch.cuda.synchronize()
        reference = torch.zeros_like(full)
        reference[:live_rows].copy_(static)
        assert torch.equal(output, F.linear(reference, weight)[:live_rows])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("kind", ["kda", "mla_key", "mla_value"])
@pytest.mark.parametrize("projection_rows", [64, 96])
@torch.inference_mode()
def test_gpu_bmm_full_parent_offset_and_changed_input_graph(kind, projection_rows):
    live_rows = projection_rows // 8 * 6
    parent, value, strides, offset = _bmm_parent(
        kind, live_rows, "cuda", torch.bfloat16
    )
    parent.normal_()
    outer = 8192 if kind == "kda" else 512 if kind == "mla_key" else 256
    weight = (
        torch.randn(
            value.shape[0], outer, value.shape[-1], device="cuda", dtype=torch.bfloat16
        ).transpose(1, 2)
        * 0.02
    )
    for _ in range(3):
        bmm(
            value,
            weight,
            projection_rows=projection_rows,
            input_strides=strides,
            input_storage_offset=offset,
        )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = bmm(
            value,
            weight,
            projection_rows=projection_rows,
            input_strides=strides,
            input_storage_offset=offset,
        )
    gc.collect()
    pressure = torch.empty(16 * 1024**2, device="cuda", dtype=torch.uint8).fill_(9)
    del pressure
    for scale in (0.5, -0.75):
        parent.copy_(parent.roll(1, 0) * scale)
        graph.replay()
        torch.cuda.synchronize()
        _, reference, _, _ = _bmm_parent(kind, projection_rows, "cuda", torch.bfloat16)
        reference.zero_()
        reference[:, :live_rows].copy_(value)
        expected = torch.bmm(reference, weight)[:, :live_rows]
        _equal(output, expected)
        _equal(_consumer(kind, output), _consumer(kind, expected))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.inference_mode()
def test_gpu_qkv_tape_view_retains_full_projection_owner_after_later_projection():
    torch.manual_seed(624)
    weight = torch.randn(24896, 4096, device="cuda", dtype=torch.bfloat16) * 0.01
    first = torch.randn(4, 4096, device="cuda", dtype=torch.bfloat16)
    later = torch.randn_like(first)

    def project_twice():
        projection = linear(first, weight, projection_rows=64)
        # This is the QKV view retained by SpeculativeTape, including row stride.
        raw_qkv = projection[:, :24576].reshape(4, 3, 8192)
        other = linear(later, weight, projection_rows=64)
        return raw_qkv, other

    for _ in range(3):
        project_twice()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        qkv, other = project_twice()
    assert qkv.stride() == (24896, 8192, 1)
    gc.collect()
    pressure = torch.empty(32 * 1024**2, device="cuda", dtype=torch.uint8).fill_(71)
    del pressure
    for scale in (0.5, -0.75):
        first.copy_(first.roll(1, 0) * scale)
        later.mul_(-0.25)
        graph.replay()
        torch.cuda.synchronize()
        reference = torch.zeros(64, 4096, device="cuda", dtype=torch.bfloat16)
        reference[:4].copy_(first)
        expected = F.linear(reference, weight)[:4]
        assert torch.equal(qkv, expected[:, :24576].reshape(4, 3, 8192))
        reference[:4].copy_(later)
        assert torch.equal(other, F.linear(reference, weight)[:4])
