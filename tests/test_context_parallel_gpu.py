"""Real NCCL token exchange and FA4 causal/window attention agree with CP=1."""

from __future__ import annotations

import faulthandler
from datetime import timedelta
from itertools import accumulate

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from resin import BatchPhase
from resin.backend.cuda.context_parallel import (
    ContextParallelTokens,
    context_parallel_query_batch,
)
from resin.backend.cuda.paged_kv import DevicePagedAttentionBatch, PagedKvTable
from resin.backend.topology import RankGroup

pytestmark = pytest.mark.skipif(
    not dist.is_nccl_available() or torch.cuda.device_count() < 2,
    reason="requires two CUDA GPUs, NCCL, and the pinned FlashAttention 4 build",
)


def _attention(fa4, query, keys, values, batch, *, sliding):
    if query.shape[0] == 0:
        return torch.empty_like(query)
    group = batch.group("full")
    result = fa4(
        query,
        keys,
        values,
        cu_seqlens_q=batch.cu_seqlens_q,
        cu_seqlens_k=None,
        max_seqlen_q=batch.max_seqlen_q,
        max_seqlen_k=group.max_length,
        min_seqlen_k=group.min_length,
        seqused_q=None,
        seqused_k=group.lengths,
        page_table=group.page_table,
        softmax_scale=query.shape[-1] ** -0.5,
        causal=True,
        window_size=(2, 0) if sliding else (None, None),
        softcap=0.0,
        num_splits=1,
        pack_gqa=True,
        deterministic=False,
        return_lse=False,
        learnable_sink=torch.zeros(
            query.shape[1], dtype=query.dtype, device=query.device
        ),
    )
    return result[0] if isinstance(result, tuple) else result


def _rank(rank, rendezvous):
    from flash_attn.cute import flash_attn_varlen_func

    from resin_models_cuda.gpt_oss.cuda import _register_empty_fa4_aux_data

    _register_empty_fa4_aux_data()
    faulthandler.dump_traceback_later(120, repeat=True)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    dist.init_process_group(
        "nccl",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        device_id=device,
        timeout=timedelta(minutes=3),
    )
    graph = None
    try:
        group = RankGroup((0, 1), dist.group.WORLD, rank)
        for lengths, prefixes in (((5, 3, 1), (4, 1, 0)), ((1,), (0,))):
            tokens = ContextParallelTokens(lengths, group, device=device)
            generator = torch.Generator(device=device).manual_seed(1729)

            def random(shape):
                return torch.randn(
                    shape, device=device, dtype=torch.bfloat16, generator=generator
                )

            total, rows = sum(lengths), len(lengths)
            query = random((total, 4, 64))
            current_keys, current_values = (
                random((total, 2, 64)),
                random((total, 2, 64)),
            )
            keys, values = (
                random((rows + 1, 128, 2, 64)),
                random((rows + 1, 128, 2, 64)),
            )
            gathered_keys = tokens.gather(tokens.select(current_keys))
            gathered_values = tokens.gather(tokens.select(current_values))
            torch.testing.assert_close(gathered_keys, current_keys, rtol=0, atol=0)
            torch.testing.assert_close(gathered_values, current_values, rtol=0, atol=0)
            offsets = tuple(accumulate(lengths, initial=0))
            append_slots = torch.tensor(
                [
                    (row + 1) * 128 + prefix + index
                    for row, (length, prefix) in enumerate(
                        zip(lengths, prefixes, strict=True)
                    )
                    for index in range(length)
                ],
                device=device,
                dtype=torch.long,
            )
            keys.view(-1, 2, 64).index_copy_(0, append_slots, gathered_keys)
            values.view(-1, 2, 64).index_copy_(0, append_slots, gathered_values)
            cache_lengths = tuple(a + b for a, b in zip(lengths, prefixes, strict=True))
            batch = DevicePagedAttentionBatch(
                BatchPhase.PREFILL,
                torch.arange(total, device=device),
                torch.arange(rows, device=device),
                torch.tensor([end - 1 for end in offsets[1:]], device=device),
                torch.tensor(offsets, device=device, dtype=torch.int32),
                max(lengths),
                (
                    (
                        "full",
                        PagedKvTable(
                            torch.arange(1, rows + 1, device=device, dtype=torch.int32)[
                                :, None
                            ],
                            torch.tensor(
                                cache_lengths, device=device, dtype=torch.int32
                            ),
                            append_slots,
                            min(cache_lengths),
                            max(cache_lengths),
                        ),
                    ),
                ),
            )
            local_batch = context_parallel_query_batch(batch, tokens)
            for sliding in (False, True):
                reference = _attention(
                    flash_attn_varlen_func, query, keys, values, batch, sliding=sliding
                )
                local = _attention(
                    flash_attn_varlen_func,
                    tokens.select(query),
                    keys,
                    values,
                    local_batch,
                    sliding=sliding,
                )
                actual = tokens.gather(local)
                torch.testing.assert_close(actual, reference, atol=0.02, rtol=0.02)

            # The very same prepared exchange captures and replays with new
            # values, including the rank with no local token in the N=1 case.
            local = tokens.select(query).clone()
            output = torch.empty_like(query)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    output.copy_(tokens.gather(local))
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output.copy_(tokens.gather(local))
            local.add_(1)
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(output, query + 1, rtol=0, atol=0)
            graph.reset()
            graph = None
    finally:
        torch.cuda.synchronize()
        if graph is not None:
            graph.reset()
        dist.destroy_process_group()


def test_context_parallel_matches_full_fa4_attention_and_replays_token_exchange(
    tmp_path,
):
    mp.spawn(_rank, args=(str(tmp_path / "cp-nccl"),), nprocs=2, join=True)
