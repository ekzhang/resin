"""World-owned embeddings compose with the independent attention/expert meshes."""

import multiprocessing
import os
import socket
import time
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
from torch import nn

from resin import Parallelism
from resin.backend.collectives import close_layer_collectives, layer_collectives
from resin.backend.topology import ParallelGroups
from resin_models_cuda.deepseek_v41.embedding import ShardedEmbeddingLookup
from resin_models_cuda.deepseek_v41.engram import ParallelEngramEmbedding


def table_values(rows, dimension, device):
    values = (
        (
            torch.arange(rows * dimension, device=device).reshape(rows, dimension) % 19
            - 9
        )
        / 8
    ).to(torch.float8_e4m3fn)
    exponents = (
        torch.arange(rows, device=device)[:, None]
        + torch.arange(dimension // 32, device=device)
    ) % 7 - 3
    scales = (2.0 ** exponents.float()).to(torch.float8_e8m0fnu)
    reference = (values.float() * scales.float().repeat_interleave(32, -1)).bfloat16()
    return values, scales, reference


@torch.inference_mode()
def _worker(rank, rendezvous):
    torch.set_num_threads(1)
    interfaces = {name for _, name in socket.if_nameindex()}
    os.environ["GLOO_SOCKET_IFNAME"] = "lo0" if "lo0" in interfaces else "lo"
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=30),
    )
    try:
        for topology in (
            Parallelism(world_size=4, attn_dp=4, ep=4),
            Parallelism(world_size=4, attn_dp=2, cp=2, ep=2),
            Parallelism(world_size=4, ep=1),
        ):
            groups = ParallelGroups.create(topology, device_backend="gloo")
            lookup = ShardedEmbeddingLookup(groups.device_domain, 32)
            shards = [ParallelEngramEmbedding(17, 32, lookup) for _ in range(2)]
            tables = nn.ModuleList(shards)
            values, scales, reference = table_values(17, 32, "cpu")
            for table in shards:
                assert table.weight.shape == (5, 32)
                count = min(17, table.stop) - table.start
                table.weight.zero_()
                table.scale.fill_(1)
                table.weight[:count].copy_(values[table.start : table.start + count])
                table.scale[:count].copy_(scales[table.start : table.start + count])
            assert list(layer_collectives(tables)) == [lookup]
            ids = (torch.arange(12).reshape(4, 3) + rank * 5) % 17
            count = (3, 2, 0, 1)[rank]
            for table in shards:
                assert torch.equal(table(ids[:count]), reference[ids[:count]])
                assert table.forward_empty(3).shape == (0, 3, 32)
            valid = torch.tensor(count, dtype=torch.int32)
            pointers = None
            for count in ((3, 2, 0, 1)[rank], (0, 1, 4, 2)[rank]):
                valid.fill_(count)
                with lookup.fixed_capacity(4, valid_rows=valid):
                    for table in shards:
                        expected = reference[ids].clone()
                        expected[count:] = 0
                        assert torch.equal(table(ids), expected)
                current = tuple(t.data_ptr() for t in lookup._buffers[4, 3, ids.device])
                assert pointers is None or current == pointers
                pointers = current
                with lookup.fixed_capacity(4, valid_rows=valid):
                    hooks = lookup.prefetch(
                        [(ids, shards[0].local), (ids.flip(-1), shards[1].local)]
                    )
                    second, first = hooks[1](), hooks[0]()
                    assert first.data_ptr() != second.data_ptr()
                    for actual, tokens in ((first, ids), (second, ids.flip(-1))):
                        expected = reference[tokens].clone()
                        expected[count:] = 0
                        assert torch.equal(actual, expected)
            hooks = lookup.prefetch(
                [
                    (ids[:count], shards[0].local),
                    (ids[:count].flip(-1), shards[1].local),
                ]
            )
            assert torch.equal(hooks[0](), reference[ids[:count]])
            assert torch.equal(hooks[1](), reference[ids[:count].flip(-1)])
            close_layer_collectives(tables)
            close_layer_collectives(tables)
            assert not lookup.allocated
            groups.close()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo is unavailable")
def test_world_lookup_handles_unequal_idle_and_replicated_attention_rows(tmp_path):
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(target=_worker, args=(rank, str(tmp_path / "embedding-meshes")))
        for rank in range(4)
    ]
    try:
        for process in processes:
            process.start()
        deadline = time.monotonic() + 60
        for process in processes:
            process.join(max(0, deadline - time.monotonic()))
        assert [process.exitcode for process in processes] == [0] * 4
    finally:
        for process in processes:
            if process.is_alive():
                process.kill()
            process.join()
