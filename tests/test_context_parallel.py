"""Context-parallel ownership preserves packed rows and global causal positions."""

from __future__ import annotations

import multiprocessing
import os
import socket
import time
from datetime import timedelta
from itertools import accumulate

import pytest
import torch
import torch.distributed as dist

from resin import BatchPhase
from resin.backend.cuda.context_parallel import (
    ContextParallelTokens,
    _ContextParallelPlan,
)
from resin.backend.topology import RankGroup
from resin_models_cuda.gpt_oss.paged_kv import (
    DeviceAttentionBatch,
    PagedKvTable,
    context_parallel_query_batch,
)


@pytest.mark.parametrize("lengths", [(5, 3, 1), (1,), (), (0, 2, 0, 1)])
@pytest.mark.parametrize("size", [1, 2, 4])
def test_balanced_shards_cover_global_tokens_and_preserve_each_sequence(lengths, size):
    plans = [
        _ContextParallelPlan.build(lengths, rank=rank, size=size)
        for rank in range(size)
    ]
    total = sum(lengths)
    assert [index for plan in plans for index in plan.token_indices] == list(
        range(total)
    )
    assert (
        max(plan.local_tokens for plan in plans)
        - min(plan.local_tokens for plan in plans)
        <= 1
    )
    assert [
        sum(plan.query_lengths[row] for plan in plans) for row in range(len(lengths))
    ] == list(lengths)
    padded = [
        index
        for plan in plans
        for index in (
            *plan.token_indices,
            *([-1] * (plan.capacity - plan.local_tokens)),
        )
    ]
    assert [padded[index] for index in plans[0].gather_indices] == list(range(total))


def _batch(lengths, prefixes):
    offsets = tuple(accumulate(lengths, initial=0))
    group = PagedKvTable(
        page_table=torch.arange(len(lengths), dtype=torch.int32)[:, None],
        lengths=torch.tensor(
            [a + b for a, b in zip(lengths, prefixes, strict=True)], dtype=torch.int32
        ),
        append_slots=torch.arange(sum(lengths)),
        min_length=min(a + b for a, b in zip(lengths, prefixes, strict=True)),
        max_length=max(a + b for a, b in zip(lengths, prefixes, strict=True)),
    )
    return DeviceAttentionBatch(
        BatchPhase.PREFILL,
        torch.arange(sum(lengths)),
        torch.arange(len(lengths)),
        torch.tensor([end - 1 for end in offsets[1:]]),
        torch.tensor(offsets, dtype=torch.int32),
        max(lengths),
        (("full", group),),
    )


def test_local_query_metadata_preserves_full_and_sliding_causal_coordinates():
    lengths, prefixes = (5, 3, 1), (4, 1, 0)
    batch = _batch(lengths, prefixes)
    offsets = tuple(accumulate(lengths, initial=0))
    for rank in range(4):
        tokens = ContextParallelTokens(
            lengths, RankGroup(tuple(range(4)), None, rank), device="cpu"
        )
        local = context_parallel_query_batch(batch, tokens)
        assert local.group("full").append_slots is batch.group("full").append_slots
        assert local.last_token_indices is batch.last_token_indices
        assert local.cu_seqlens_q[-1] == tokens.plan.local_tokens
        assert torch.diff(local.cu_seqlens_q).gt(0).all()
        for active, sequence in enumerate(local.sequence_indices.tolist()):
            count = int(local.cu_seqlens_q[active + 1] - local.cu_seqlens_q[active])
            key_length = int(local.group("full").lengths[active])
            global_start = max(tokens.plan.start, offsets[sequence]) - offsets[sequence]
            for window in (None, 3):
                keys = torch.arange(prefixes[sequence] + lengths[sequence])
                global_positions = (
                    prefixes[sequence] + global_start + torch.arange(count)
                )
                expected = keys[None, :] <= global_positions[:, None]
                actual_positions = key_length - count + torch.arange(count)
                actual = (keys[None, :] < key_length) & (
                    keys[None, :] <= actual_positions[:, None]
                )
                if window is not None:
                    expected &= keys[None, :] >= global_positions[:, None] - window + 1
                    actual &= keys[None, :] >= actual_positions[:, None] - window + 1
                assert torch.equal(actual, expected)


def _gather_worker(rank, rendezvous):
    torch.set_num_threads(1)
    interfaces = {name for _, name in socket.if_nameindex()}
    os.environ["GLOO_SOCKET_IFNAME"] = "lo0" if "lo0" in interfaces else "lo"
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=25),
    )
    try:
        group = RankGroup((0, 1), dist.group.WORLD, rank)
        for lengths in ((5, 3, 1), (1,), (0,), ()):
            tokens = ContextParallelTokens(lengths, group, device="cpu")
            reference = torch.arange(sum(lengths) * 6, dtype=torch.float32).view(
                -1, 2, 3
            )
            local = tokens.select(reference)
            torch.testing.assert_close(tokens.gather(local), reference)
            torch.testing.assert_close(tokens.gather(local + 1), reference + 1)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo is unavailable")
def test_context_token_exchange_handles_ragged_and_empty_shards(tmp_path):
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(target=_gather_worker, args=(rank, str(tmp_path / "cp")))
        for rank in range(2)
    ]
    try:
        for process in processes:
            process.start()
        deadline = time.monotonic() + 45
        for process in processes:
            process.join(max(0, deadline - time.monotonic()))
        assert [process.exitcode for process in processes] == [0, 0]
    finally:
        for process in processes:
            if process.is_alive():
                process.kill()
            process.join()
