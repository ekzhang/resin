"""Numerical and replay contracts for Qwen fused normalization and attention."""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import pytest
import torch
from torch.nn import functional as F

from resin.backend.cuda.paged_kv import SlabPagedKvCache
from resin.backend.cuda.slab import DevicePageGroup
from resin.models.qwen3_5.config import Qwen35Config
from resin_models_cuda.qwen3_5 import model as qwen
from resin_models_cuda.qwen3_5.attention import (
    GraphAttentionSpec,
    GraphPagedAttentionRuntime,
    PreparedPagedAttention,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _norm(hidden, weight, eps, gate=None):
    normalized = hidden.float() * torch.rsqrt(
        hidden.float().square().mean(-1, keepdim=True) + eps
    )
    output = normalized * weight.float()
    if gate is not None:
        output = output * F.silu(gate.float())
    return output.to(hidden.dtype)


class _ReferenceGatedNorm(qwen.Qwen35GatedRMSNorm):
    def __init__(self, source):
        super().__init__(source.weight.numel(), source.eps)
        self.weight = source.weight
        self.eps = source.eps

    def forward(self, hidden, gate):
        return _norm(hidden, self.weight, self.eps, gate)


def _check(actual, expected):
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
    difference = actual.float() - expected.float()
    error = (difference.norm() / expected.float().norm().clamp_min(1e-20)).item()
    if error > 0.003:
        raise AssertionError(f"Qwen relative L2 error {error}")
    return {"relative_l2": error, "max_abs": difference.abs().max().item()}


def _capture(call):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            call()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = call()
    torch.cuda.current_stream().wait_stream(stream)
    return graph, output


def _norms():
    results = []
    for rows, width, strided in [
        (8, 5120, False),
        (1024, 5120, False),
        (68, 128, True),
    ]:
        storage = torch.randn(
            rows, width * (2 if strided else 1), device="cuda", dtype=torch.bfloat16
        )
        hidden = storage[:, :width]
        gate = torch.randn_like(hidden)
        for gated in (False, True):
            norm = (
                (qwen.Qwen35GatedRMSNorm if gated else qwen.Qwen35RMSNorm)(width, 1e-6)
                .cuda()
                .bfloat16()
            )
            norm.weight.normal_(mean=0.0, std=0.3)
            optimized = (
                (lambda: norm(hidden, gate)) if gated else (lambda: norm(hidden))
            )

            def reference():
                return _norm(
                    hidden,
                    norm.weight if gated else (1 + norm.weight.float()),
                    norm.eps,
                    gate if gated else None,
                )

            graph, output = _capture(optimized)
            errors = []
            for _ in range(3):
                storage.normal_()
                gate.normal_()
                expected = reference()
                graph.replay()
                errors.append(_check(output, expected))
            results.append(
                dict(
                    rows=rows,
                    width=width,
                    strided=strided,
                    gated=gated,
                    errors=errors,
                )
            )
    return results


def _config():
    return Qwen35Config(
        vocab_size=32,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        max_position_embeddings=131072,
        rms_norm_eps=1e-6,
        rope_theta=10000,
        partial_rotary_factor=1.0,
        mrope_section=None,
        layer_types=("linear_attention",),
        mtp_num_hidden_layers=0,
        mtp_use_dedicated_embeddings=False,
        eos_token_id=1,
        bos_token_id=2,
    )


def _delta():
    module = qwen.Qwen35GatedDeltaNet(_config()).cuda().bfloat16().eval()
    reference = copy.deepcopy(module)
    reference.norm = _ReferenceGatedNorm(reference.norm)
    kernels = qwen._fla_kernels()
    assert kernels is not None

    def repeat_heads(kernel):
        def run(query, key, value, **kwargs):
            repeats = value.shape[2] // query.shape[2]
            return kernel(
                query.repeat_interleave(repeats, 2),
                key.repeat_interleave(repeats, 2),
                value,
                **kwargs,
            )

        return run

    old_kernels = tuple(repeat_heads(kernel) for kernel in kernels)
    results = []
    for batch, tokens in ((8, 1), (2, 4), (1, 129), (1, 1024)):
        hidden = torch.randn(batch, tokens, 256, device="cuda", dtype=torch.bfloat16)
        recurrent = torch.randn(batch, 8, 128, 128, device="cuda") * 0.01
        convolution = torch.randn(
            batch, module.conv_dim, 3, device="cuda", dtype=torch.bfloat16
        )

        def old():
            with patch.object(qwen, "_fla_kernels", return_value=old_kernels):
                return reference(hidden, recurrent, convolution)

        def new():
            return module(hidden, recurrent, convolution)

        graph, output = _capture(new)
        errors = []
        for _ in range(3):
            hidden.normal_()
            recurrent.normal_(std=0.01)
            convolution.normal_()
            expected = old()
            graph.replay()
            errors.append([_check(a, b) for a, b in zip(output, expected, strict=True)])
        # Chunk continuation must preserve both recurrent and convolution caches.
        if tokens > 4:
            first = module(hidden[:, : tokens // 2], recurrent, convolution)
            second = module(hidden[:, tokens // 2 :], first[1], first[2])
            whole = new()
            _check(torch.cat((first[0], second[0]), dim=1), whole[0])
            _check(second[1], whole[1])
            torch.testing.assert_close(second[2], whole[2], rtol=0, atol=0)
        results.append(
            dict(
                batch=batch,
                tokens=tokens,
                errors=errors,
            )
        )
    return results


def _paged_attention():
    from flashinfer import BatchPrefillWithPagedKVCacheWrapper

    results = []
    # Three physical layers interleave each logical page. The two requests use
    # disjoint, nonmonotonic page lists, including partial final pages.
    keys = torch.randn(18, 16, 2, 64, device="cuda", dtype=torch.bfloat16)
    values = torch.randn_like(keys)
    cache = cast(SlabPagedKvCache, SimpleNamespace(keys=keys, values=values))
    base = torch.tensor([9, 3, 15], device="cuda", dtype=torch.int32)
    physical = base.clone()
    indptr = torch.tensor([0, 2, 3], device="cuda", dtype=torch.int32)
    last = torch.tensor([5, 9], device="cuda", dtype=torch.int32)
    for tokens in (1, 4):
        queries = torch.randn(2, tokens, 4, 64, device="cuda", dtype=torch.bfloat16)
        qo = torch.tensor([0, tokens, 2 * tokens], device="cuda", dtype=torch.int32)
        wrapper = BatchPrefillWithPagedKVCacheWrapper(
            torch.empty(32 * 1024**2, device="cuda", dtype=torch.uint8),
            "NHD",
            use_cuda_graph=True,
            qo_indptr_buf=qo,
            paged_kv_indptr_buf=indptr,
            paged_kv_indices_buf=physical,
            paged_kv_last_page_len_buf=last,
            backend="fa2",
        )
        wrapper.plan(
            qo,
            indptr,
            physical,
            last,
            4,
            2,
            64,
            16,
            causal=True,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
            disable_split_kv=True,
        )
        # PreparedPagedAttention flattens prefill tokens but uses [B,H,D] for
        # single-token decode; both are accepted by this prefill wrapper.
        attention = PreparedPagedAttention(wrapper, cache, base, physical, 3, tokens)
        graph, output = _capture(
            lambda: tuple(attention.run(queries, layer) for layer in range(3))
        )
        errors = []
        for page_ids in ([3, 1, 5], [2, 4, 1]):
            base.copy_(torch.tensor(page_ids, device="cuda", dtype=torch.int32) * 3)
            queries.normal_()
            graph.replay()
            for layer in range(3):
                expected = []
                for row, (pages, length) in enumerate(
                    ((page_ids[:2], 21), (page_ids[2:], 9))
                ):
                    indices = torch.tensor(pages, device="cuda") * 3 + layer
                    key = keys[indices].flatten(0, 1)[:length].repeat_interleave(2, 1)
                    value = (
                        values[indices].flatten(0, 1)[:length].repeat_interleave(2, 1)
                    )
                    mask = torch.arange(length, device="cuda")[None] <= (
                        length - tokens + torch.arange(tokens, device="cuda")[:, None]
                    )
                    expected.append(
                        F.scaled_dot_product_attention(
                            queries[row].transpose(0, 1).float(),
                            key.transpose(0, 1).float(),
                            value.transpose(0, 1).float(),
                            attn_mask=mask,
                        )
                        .transpose(0, 1)
                        .bfloat16()
                    )
                errors.append(_check(output[layer], torch.stack(expected)))
        results.append({"tokens": tokens, "errors": errors})
    return results


@pytest.mark.parametrize("check", [_norms, _delta, _paged_attention])
@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 9,
    reason="model normalization kernels require Hopper or newer CUDA",
)
@torch.inference_mode()
def test_qwen_native_kernels(check):
    torch.manual_seed(93)
    check()


def _dense_graph_attention(query, cache, call, lengths):
    """Independent FP32 causal reference from the exact dense lease prefix."""
    spec = call.spec
    outputs = []
    group = cache.groups[spec.group_index]
    for row in range(len(query)):
        if group.slab_ids[row, 0].item() == 0:
            outputs.append(torch.zeros_like(query[row]))
            continue
        length = int(lengths[row].item())
        physical = (
            group.slab_ids[row] * cache.slab_units
            + group.slotnums[row] * spec.layer_count
            + spec.layer_count
            - 1
        ).long()
        key = cache.keys[physical].flatten(0, 1)[:length].repeat_interleave(2, 1)
        value = cache.values[physical].flatten(0, 1)[:length].repeat_interleave(2, 1)
        causal = torch.arange(length, device="cuda")[None] <= (
            length - spec.tokens + torch.arange(spec.tokens, device="cuda")[:, None]
        )
        outputs.append(
            F.scaled_dot_product_attention(
                query[row].transpose(0, 1).float(),
                key.transpose(0, 1).float(),
                value.transpose(0, 1).float(),
                attn_mask=causal,
            )
            .transpose(0, 1)
            .bfloat16()
        )
    return torch.stack(outputs)


@pytest.mark.parametrize(
    "rows,tokens",
    [(rows, tokens) for rows in (4, 32) for tokens in (1, 2, 4)] + [(4, 129)],
)
@torch.inference_mode()
def test_graph_attention_refreshes_exact_csr_across_reserved_pages(tokens, rows):
    torch.manual_seed(391 + rows + tokens)
    width, page, heads, kv_heads, dimension, slab_units = 8, 128, 4, 2, 256, 6
    first = torch.zeros(rows, dtype=torch.int32, device="cuda")
    first[1::3] = 2
    slabs = (
        torch.arange(rows * width, device="cuda", dtype=torch.int32).view(rows, width)
        // 2
        + 1
    )
    slots = (
        torch.arange(rows * width, device="cuda", dtype=torch.int32).view(rows, width)
        % 2
    )
    slabs[-1].zero_()
    slots[-1].zero_()
    groups = tuple(
        DevicePageGroup(slabs.clone(), slots.clone(), first.clone()) for _ in range(3)
    )
    keys = (
        torch.randn(
            (rows * width // 2 + 2) * slab_units,
            page,
            kv_heads,
            dimension,
            device="cuda",
            dtype=torch.bfloat16,
        )
        * 0.2
    )
    values = torch.randn_like(keys)
    keys[:slab_units].zero_()
    values[:slab_units].zero_()
    cache = cast(
        SlabPagedKvCache,
        SimpleNamespace(
            keys=keys,
            values=values,
            groups=groups,
            page_tokens=page,
            slab_units=slab_units,
        ),
    )
    runtime = GraphPagedAttentionRuntime(
        torch.device("cuda"),
        cache,
        (
            GraphAttentionSpec(0, 3, tokens, tokens),
            GraphAttentionSpec(2, 1, tokens, tokens + 1),
        ),
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        head_dim=dimension,
        workspace=torch.empty(64 * 1024**2, dtype=torch.uint8, device="cuda"),
    )
    queries = (
        torch.randn(rows, tokens, heads, dimension, device="cuda", dtype=torch.bfloat16)
        * 0.2
    )
    positions = first.long() * page + 125

    def run():
        runtime.refresh(positions)
        return tuple(
            call.prepared.run(queries, call.spec.layer_count - 1)
            for call in runtime.calls
        )

    graph, output = _capture(run)
    pointers = [
        (
            call.kv_indptr.data_ptr(),
            call.prepared.base_pages.data_ptr(),
            call.last_page_lengths.data_ptr(),
        )
        for call in runtime.calls
    ]
    for iteration, position in enumerate(
        (125, 126, 127, 128, 129, 255, 256, 257, 511, 512, 513)
    ):
        positions.copy_(
            first.long() * page + position + torch.arange(rows, device="cuda") % 3
        )
        positions[-1] = 100000  # A padded row must never expose a future page.
        if iteration == 5:
            for group in groups:
                group.slab_ids[:-1].copy_(group.slab_ids[:-1].flip(1))
                group.slotnums[:-1].copy_(group.slotnums[:-1].flip(1))
        queries.normal_(std=0.2)
        graph.replay()
        for index, call in enumerate(runtime.calls):
            expected_lengths = positions + call.spec.sequence_offset - first * page
            expected_lengths[-1] = tokens
            counts = (expected_lengths + page - 1) // page
            expected_ptr = torch.zeros(rows + 1, device="cuda", dtype=torch.int32)
            expected_ptr[1:] = counts.cumsum(0)
            torch.testing.assert_close(call.kv_indptr, expected_ptr, rtol=0, atol=0)
            torch.testing.assert_close(
                call.last_page_lengths,
                ((expected_lengths - 1) % page + 1).int(),
                rtol=0,
                atol=0,
            )
            for row in range(rows):
                start, end = expected_ptr[row : row + 2].cpu().tolist()
                expected_pages = (
                    groups[call.spec.group_index].slab_ids[row, : end - start]
                    * slab_units
                    + groups[call.spec.group_index].slotnums[row, : end - start]
                    * call.spec.layer_count
                )
                torch.testing.assert_close(
                    call.prepared.base_pages[start:end], expected_pages, rtol=0, atol=0
                )
            expected = _dense_graph_attention(queries, cache, call, expected_lengths)
            torch.testing.assert_close(output[index], expected, rtol=0.015, atol=0.003)
            assert pointers[index] == (
                call.kv_indptr.data_ptr(),
                call.prepared.base_pages.data_ptr(),
                call.last_page_lengths.data_ptr(),
            )
