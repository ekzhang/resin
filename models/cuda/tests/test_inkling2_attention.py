"""Representative attention numerics, graph replay and cache/state ownership."""

import os
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from resin.backend.cuda.slab import DevicePageGroup
from resin_models_cuda.inkling2.cache import DeviceCache

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.fixture(autouse=True)
def cuda_device():
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    torch.manual_seed(281)


def cache_fixture(lengths, kv_heads):
    columns = (int(lengths.max()) + 127) // 128
    pages = (
        torch.randperm(len(lengths) * columns, device="cuda").reshape(-1, columns) + 1
    ).to(torch.int32)
    storage = torch.randn(
        pages.numel() + 1, 2, 128, kv_heads, 128, device="cuda"
    ).bfloat16()
    return DeviceCache(
        geometry=cast(Any, SimpleNamespace(slab_units=1)),
        groups=(
            DevicePageGroup(pages, torch.zeros_like(pages), torch.zeros_like(lengths)),
        ),
        keys=storage[:, 0],
        values=storage[:, 1],
        states=storage.new_empty(0),
        draft_features=None,
    )


def dense_attention(q, relative, cache, lengths, window, *, query_indices=None):
    tokens = q.shape[1]
    if query_indices is None:
        query_indices = torch.arange(tokens, device=q.device)
    else:
        q, relative = q[:, query_indices], relative[:, query_indices]
    query_heads_per_kv_head = q.shape[2] // cache.keys.shape[-2]
    output, lse = [], []
    for row, end in enumerate(lengths.tolist()):
        logical = torch.arange(end, device=q.device)
        page = cache.groups[0].slab_ids[row, logical // 128].long()
        k, v = (
            x[page, logical % 128].float().repeat_interleave(query_heads_per_kv_head, 1)
            for x in (cache.keys, cache.values)
        )
        distance = end - tokens + query_indices[:, None] - logical
        scores = torch.einsum("qhd,khd->qhk", q[row].float(), k) / 128**0.5
        bias = (
            relative[row]
            .float()
            .gather(-1, distance.clamp(0, 1023)[:, None].expand(-1, q.shape[2], -1))
        )
        scores += bias * ((distance >= 0) & (distance < 1024))[:, None]
        visible = distance >= 0
        if window is not None:
            visible &= distance <= window
        scores.masked_fill_(~visible[:, None], -torch.inf)
        output.append(torch.einsum("qhk,khd->qhd", scores.softmax(-1), v))
        lse.append(scores.logsumexp(-1))
    return torch.stack(output), torch.stack(lse)


def gated_reference(output, lse, gate, sinks):
    return (
        output * gate.float().sigmoid() * (lse - sinks.float()).sigmoid()[..., None]
    ).bfloat16()


def capture(call):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        call()
        call()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = call()
    return graph, result


def test_store_kv_unseen_chunks_reuse_compilation(monkeypatch):
    from resin_models_cuda.inkling2.triton import _store_kv

    device = torch.cuda.current_device()
    saved = _store_kv.device_caches[device]
    monkeypatch.setitem(_store_kv.device_caches, device, ({}, *saved[1:]))
    variants = None
    for tokens, start in ((128, 0), (1, 127), (129, 128), (4096, 257), (3073, 8193)):
        positions = (
            torch.arange(start, start + tokens, device="cuda")[None]
            .expand(2, -1)
            .clone()
        )
        cache = cache_fixture(positions[:, -1] + 1, 2)
        keys, values = (
            torch.randn(2, tokens, 2, 256, device="cuda").bfloat16()[..., ::2]
            for _ in range(2)
        )
        for row in (None, 1):
            active = cache if row is None else cache.row(row)
            if row is not None:
                # Row metadata and an odd packed-token offset lose alignment.
                keys, values, positions = (
                    keys[1:, 1:],
                    values[1:, 1:],
                    positions[:1, 1:],
                )
                keys.neg_()
                values.neg_()
            pages = active.groups[0].slab_ids.gather(1, positions // 128).long()
            expected = [cache.keys.clone(), cache.values.clone()]
            for output, source in zip(expected, (keys, values), strict=True):
                output[pages, positions % 128] = source
            active.store_kv(
                keys, values, positions, group_index=0, layer=0, layer_count=1
            )
            for actual, wanted in zip(
                (cache.keys, cache.values), expected, strict=True
            ):
                torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
            compiled = set(_store_kv.device_caches[device][0])
            if variants is None:
                variants = compiled
            else:
                assert compiled == variants


@pytest.mark.parametrize("window", [None, 1023])
def test_paged_attention_and_sink_match_dense(window):
    from resin_models_cuda.inkling2.attention import paged_attention
    from resin_models_cuda.inkling2.kernels import gate_sink

    lengths = torch.tensor([127, 129], device="cuda")
    cache = cache_fixture(lengths, 1)
    q = torch.randn(2, 3, 6, 128, device="cuda").bfloat16() * 0.2
    relative = torch.randn(2, 3, 6, 1024, device="cuda").bfloat16() * 0.05
    expected, expected_lse = dense_attention(q, relative, cache, lengths, window)
    actual, lse = paged_attention(
        q,
        cache=cache,
        group_index=0,
        layer=0,
        layer_count=1,
        positions=lengths[:, None] - 3 + torch.arange(3, device="cuda"),
        sequence_lengths=lengths,
        max_sequence_length=256,
        softmax_scale=128**-0.5,
        window_left=window,
        relative_logits=relative.flatten(0, 1),
        return_lse=True,
        output_dtype=torch.float32,
    )
    torch.testing.assert_close(actual, expected, rtol=0.015, atol=0.001)
    torch.testing.assert_close(
        lse, expected_lse.flatten(0, 1).t(), rtol=0.0005, atol=0.0005
    )
    gate, sinks = torch.randn_like(q), torch.randn_like(q[..., 0])
    gated = gate_sink(
        actual.flatten(0, 1), gate.reshape(6, -1), lse, sinks.flatten(0, 1)
    )
    torch.testing.assert_close(
        gated.view_as(q),
        gated_reference(expected, expected_lse, gate, sinks),
        rtol=0.02,
        atol=0.001,
    )


def test_generic_fp32_split_kv_reuses_compilation():
    from resin_models_cuda.inkling2 import fa4
    from resin_models_cuda.inkling2.decode_attention import gated_decode

    lengths = torch.tensor([727, 1185], dtype=torch.int32, device="cuda")
    cache = cache_fixture(lengths, 2)
    batch, tokens, heads = 2, 3, 12
    q = torch.randn(batch, tokens, heads, 128, device="cuda").bfloat16() * 0.2
    relative = torch.randn(batch, tokens, heads, 1024, device="cuda").bfloat16() * 0.05
    gate, sinks = torch.randn_like(q), torch.randn_like(q[..., 0])
    window = 1023
    dense, lse = dense_attention(q, relative, cache, lengths, window)
    expected = gated_reference(dense, lse, gate, sinks).flatten(0, 1)
    cu_q = torch.arange(batch + 1, dtype=torch.int32, device=q.device) * tokens

    def attend(splits):
        return gated_decode(
            q.flatten(0, 1),
            cache.keys,
            cache.values,
            relative.flatten(0, 1),
            gate.flatten(0, 1),
            sinks.flatten(0, 1),
            page_table=cache.groups[0].slab_ids,
            seqused_k=lengths,
            cu_seqlens_q=cu_q,
            tokens=tokens,
            num_splits=splits,
            window_left=window,
        )

    torch.testing.assert_close(attend(2), expected, rtol=0.02, atol=0.001)
    compiled = fa4.compile_count
    torch.testing.assert_close(attend(4), expected, rtol=0.02, atol=0.001)
    assert fa4.compile_count == compiled


@pytest.mark.parametrize("heads", [6, 8])
def test_global_prefill_split_bias_boundary_and_compilation(heads):
    from resin_models_cuda.inkling2 import fa4
    from resin_models_cuda.inkling2.attention import (
        gated_attention,
        prepare_paged_attention_batch,
    )
    from resin_models_cuda.inkling2.prefill_attention import (
        _merge_gate,
        warm_prefill_split,
    )

    device = torch.device("cuda", torch.cuda.current_device())
    if torch.cuda.get_device_capability(device) != (10, 3):
        pytest.skip("requires the SM103 global prefill kernel")  # ty: ignore[too-many-positional-arguments]
    warm_prefill_split(
        heads, 1, 1024, device, prefill_chunk_size=4096, max_sequence_length=32768
    )
    fa4_compiles = fa4.compile_count
    merge_compiles = set(_merge_gate.device_caches[device.index][0])

    def check(tokens, history, pages, *, replay=False):
        lengths = torch.tensor([history + tokens], dtype=torch.int32, device=device)
        cache = cache_fixture(lengths, 1)
        q = (
            torch.randn((1, tokens, heads, 128), dtype=torch.bfloat16, device=device)
            * 0.2
        )
        relative = torch.zeros((*q.shape[:3], 1024), dtype=q.dtype, device=device)
        relative[..., 0], relative[..., 1023] = 4, 5
        packed = torch.randn(
            (1, tokens, heads * 128 + heads + 4), dtype=q.dtype, device=device
        )
        gate = packed[..., 2 : 2 + heads * 128].view_as(q)
        sinks = packed[..., -heads:]
        positions = history + torch.arange(tokens, device=device)[None]
        options: dict[str, Any] = dict(
            cache=cache,
            group_index=0,
            layer=0,
            layer_count=1,
            positions=positions,
            sequence_lengths=lengths,
            max_sequence_length=history + tokens,
            softmax_scale=128**-0.5,
            relative_logits=relative.flatten(0, 1),
        )
        batch_args: dict[str, Any] = dict(batch=1, tokens=tokens, layer_counts={0: 1})
        prepared = prepare_paged_attention_batch(
            cache,
            lengths,
            history + tokens,
            global_prefill_relative_extent=1024,
            **batch_args,
        )
        stock = prepare_paged_attention_batch(
            cache,
            lengths,
            history + tokens,
            **batch_args,
        )
        split = prepared.global_prefill_split
        assert stock.global_prefill_split is None
        if pages:
            assert split is not None and split.pages == pages
            assert split.far_lengths.item() == pages * 128
            assert split.near_lengths.item() == history + tokens - pages * 128
        else:
            assert split is None

        def attend(batch=prepared):
            return gated_attention(q, gate, sinks, prepared_batch=batch, **options)

        indices = torch.tensor([0, 1, 127, 128, tokens - 1], device=device)

        def assert_dense(actual):
            out, lse = dense_attention(
                q, relative, cache, lengths, None, query_indices=indices
            )
            expected = gated_reference(out, lse, gate[:, indices], sinks[:, indices])
            torch.testing.assert_close(
                actual[:, indices], expected, rtol=0.014, atol=0.0005
            )

        actual = attend()
        torch.testing.assert_close(
            actual,
            attend(stock),
            rtol=0.014 if pages else 0,
            atol=0.0005 if pages else 0,
        )
        assert_dense(actual)
        if replay:
            graph, actual = capture(attend)
            q.mul_(-0.875)
            sinks.add_(0.25)
            reordered = cache.groups[0].slab_ids.flip(1)
            cache.groups[0].slab_ids.copy_(reordered)
            prepared.page_tables[0][0].copy_(reordered)
            graph.replay()
            torch.testing.assert_close(actual, attend(), rtol=0, atol=0)
            assert_dense(actual)

    check(1024, 16384 + 1022, 0)
    check(1024, 16384 + 1023, 128, replay=True)
    check(1153, 16384 + 1024 + 128, 129)
    assert fa4.compile_count == fa4_compiles
    assert set(_merge_gate.device_caches[device.index][0]) == merge_compiles


def pad_and_shear(q, relative, used, local):
    """Independent Torch layout for the native consumer's last ten KV tiles."""
    padded_q, biases = [], []
    for query, rel, end in zip(q, relative, used, strict=True):
        tokens, heads, dim = query.shape
        kv_heads = heads // 6
        pq = query.new_zeros(tokens, kv_heads, 8, dim)
        pr = rel.new_zeros(tokens, kv_heads, 8, 1024)
        pq[:, :, :6], pr[:, :, :6] = (
            query.view(tokens, kv_heads, 6, dim),
            rel.view(tokens, kv_heads, 6, 1024),
        )
        last_tile = (torch.minimum(end, end - tokens + 16) + 127) // 128
        keys = (last_tile - 10) * 128 + torch.arange(1280, device=q.device)
        distance = end - tokens + torch.arange(tokens, device=q.device)[:, None] - keys
        bias = pr.gather(
            -1,
            distance.clamp(0, 1023)[:, None, None]
            .expand(tokens, kv_heads, 8, 1280)
            .long(),
        )
        old = (distance > 1023) | (keys < 0)[None]
        bias = torch.where(old[:, None, None], 0.0, bias)
        bias.masked_fill_(
            ((distance < 0) | (old if local else False))[:, None, None], -torch.inf
        )
        padded_q.append(pq.flatten(1, 2))
        biases.append(bias.flatten(1, 2))
    return torch.cat(padded_q), torch.cat(biases)


@pytest.mark.parametrize("local", [False, True])
def test_native_warmup_and_changed_graph(local, monkeypatch):
    """Can run first in a fresh process: startup covers this decode geometry."""
    from triton import knobs

    from resin_models_cuda.inkling2 import fa4_bias

    fa4_bias.warm_native_bias(4, torch.device("cuda"))
    compiled = fa4_bias.compile_count

    def no_compile(**event):
        raise AssertionError(f"request compiled {event['fn'].name}")

    monkeypatch.setattr(knobs.runtime, "jit_post_compile_hook", no_compile)
    used = torch.tensor([1151, 1153], device="cuda", dtype=torch.int32)
    cache = cache_fixture(used, 4)
    q = torch.randn(2, 8, 24, 128, device="cuda").bfloat16() * 0.2
    relative = torch.randn(2, 8, 24, 1024, device="cuda").bfloat16() * 0.05
    relative[..., 1023] = 6
    gate, sinks = torch.randn_like(q), torch.randn_like(q[..., 0])
    padded, bias = pad_and_shear(q, relative, used, local)
    cu_q = torch.tensor([0, 8, 16], device="cuda", dtype=torch.int32)

    def call():
        return fa4_bias.native_bias_attention(
            padded,
            cache.keys,
            cache.values,
            bias,
            gate.flatten(0, 1),
            sinks.flatten(0, 1),
            page_table=cache.groups[0].slab_ids,
            seqused_k=used,
            cu_seqlens_q=cu_q,
            tokens=8,
            num_splits=fa4_bias.decode_splits(2, 4, 1023 if local else None),
            window_left=1023 if local else None,
        )

    graph, actual = capture(call)
    for _ in range(2):
        used.sub_(1)
        q.mul_(-0.75)
        relative.mul_(-0.5)
        gate.add_(0.125)
        cache.groups[0].slab_ids.copy_(cache.groups[0].slab_ids.flip(1))
        pq, pr = pad_and_shear(q, relative, used, local)
        padded.copy_(pq)
        bias.copy_(pr)
        graph.replay()
        output, lse = dense_attention(q, relative, cache, used, 1023 if local else None)
        torch.testing.assert_close(actual, call(), rtol=0, atol=0)
        torch.testing.assert_close(
            actual.view_as(q),
            gated_reference(output, lse, gate, sinks),
            rtol=0.02,
            atol=0.001,
        )
    assert fa4_bias.compile_count == compiled


@pytest.mark.parametrize("dynamic", [False, True])
def test_fused_prologue_preserves_cache_state_and_replay(dynamic):
    from resin_models_cuda.inkling2.attention_prologue import attention_prologue
    from resin_models_cuda.inkling2.kernels import l2_clip, rms_norm, short_convolution

    lengths = torch.tensor([134, 136], device="cuda")
    cache = cache_fixture(lengths, 2)
    positions = lengths[:, None] - 8 + torch.arange(8, device="cuda")
    packed = torch.randn(2, 8, 1536 + 256 + 256 + 192, device="cuda").bfloat16()
    q, k, v, relative = packed.split([1536, 256, 256, 192], -1)
    args: dict[str, Any] = dict(
        q=q,
        k_input=k,
        v_input=v,
        relative=relative,
        cache=cache,
        positions=positions,
        group_index=0,
        layer=0,
        layer_count=1,
        qk_threshold=3.0,
        rel_threshold=1.7,
        rms_eps=1e-6,
        q_scale=torch.randn(12, 128, device="cuda").bfloat16(),
        rel_scale=torch.randn(12, 16, device="cuda").bfloat16(),
    )
    for name in ("k", "v"):
        args[f"{name}_state"] = torch.randn(2, 3, 512, device="cuda").bfloat16()[
            ..., ::2
        ]
        args[f"{name}_weight"] = torch.randn(
            (4, 256) if dynamic else (256, 4), device="cuda"
        )
        args[f"{name}_state_out"] = torch.empty_like(args[f"{name}_state"])
        args[f"{name}_inputs_out"] = torch.empty_like(args[f"{name}_input"])
        if dynamic:
            args[f"{name}_dynamic"] = torch.randn(2, 8, 2, 4, device="cuda").bfloat16()
    graph, actual = capture(lambda: attention_prologue(**args))
    for _ in range(2):
        packed.mul_(-0.75)
        cache.groups[0].slab_ids.copy_(cache.groups[0].slab_ids.flip(1))
        states = [args[f"{name}_state"].clone() for name in ("k", "v")]
        convolved, tails = zip(
            *(
                short_convolution(
                    args[f"{name}_input"],
                    state,
                    args[f"{name}_weight"],
                    args.get(f"{name}_dynamic"),
                )
                for name, state in zip(("k", "v"), states, strict=True)
            ),
            strict=True,
        )
        expected = (
            l2_clip(q.view(2, 8, 12, 128), args["q_scale"], 3.0),
            l2_clip(relative.view(2, 8, 12, 16), args["rel_scale"], 1.7),
            *tails,
        )
        kv = (
            l2_clip(convolved[0].view(2, 8, 2, 128), None, 3.0),
            rms_norm(convolved[1].view(2, 8, 2, 128), None, 1e-6),
        )
        cache.keys.fill_(-777)
        cache.values.fill_(-777)
        stored = [cache.keys.clone(), cache.values.clone()]
        for row in range(2):
            for token, position in enumerate(positions[row].tolist()):
                page = int(cache.groups[0].slab_ids[row, position // 128])
                for target, values in zip(stored, kv, strict=True):
                    target[page, position % 128] = values[row, token]
        graph.replay()
        for got, want in zip(actual, expected, strict=True):
            torch.testing.assert_close(got, want, rtol=0, atol=0)
        for got, want in zip((cache.keys, cache.values), stored, strict=True):
            torch.testing.assert_close(got, want, rtol=0, atol=0)
        for name, state, tail in zip(("k", "v"), states, tails, strict=True):
            torch.testing.assert_close(args[f"{name}_state"], state, rtol=0, atol=0)
            torch.testing.assert_close(args[f"{name}_state_out"], tail, rtol=0, atol=0)
            torch.testing.assert_close(
                args[f"{name}_inputs_out"], args[f"{name}_input"], rtol=0, atol=0
            )
