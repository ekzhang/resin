"""Storage transports preserve opaque bytes and complete native checkpoints."""

import hashlib
import random
import struct
from typing import Any

import numpy as np
import pytest
from output_helpers import forward_output

from resin import (
    CacheGroupLayout,
    CacheLayout,
    CacheRetention,
    Engine,
    RunnerConfig,
    SamplingParams,
    SchedulerConfig,
)
from resin.cache_storage import MemoryCacheStorage


class ByteRunner:
    def __init__(self):
        self.layout = CacheLayout(
            parent_capacity=40,
            groups=[
                CacheGroupLayout(4, blocks_per_slab=2),
                CacheGroupLayout(
                    2,
                    blocks_per_slab=4,
                    retention=CacheRetention.WINDOW,
                    window_tokens=4,
                ),
            ],
        )
        self.runner_config = RunnerConfig(
            cache_layout=self.layout, continuation_prefill=True
        )
        self.arena = np.zeros((41, 64), np.uint8)
        self.histories = {}

    def bind_resources(self, binding):
        assert binding.cache_layout == self.layout

    def region(self, slab, slot):
        return memoryview(self.arena[int(slab), int(slot) * 16 : (int(slot) + 1) * 16])

    def cache_storage_buffers(self, groups):
        return tuple(
            (self.region(slab, slot)[:5], self.region(slab, slot)[5:])
            for _, rows in groups
            for slab, slot in rows
        )

    def payload(self, history, group, block):
        end = (block + 1) * self.layout.groups[group].block_tokens
        tokens = list(history)
        while len(tokens) < end:
            tokens.append(tokens[-1] + 1)
        return hashlib.blake2b(
            struct.pack("<II", group, block)
            + np.asarray(tokens[:end], "<u4").tobytes(),
            digest_size=16,
        ).digest()

    def forward(self, plan):
        output = []
        for row, request_id in enumerate(plan.sequence_ids):
            history = self.histories[int(request_id)]
            start = int(plan.context_lengths[row])
            begin, end = map(int, plan.token_indptr[row : row + 2])
            for group in plan.cache_groups:
                index = group.group_index
                width = self.layout.groups[index].block_tokens
                first = int(group.block_table_start_indices[row])
                left, right = map(int, group.block_indptr[row : row + 2])
                for block, slab, slot in zip(
                    range(first, first + right - left),
                    group.block_slab_ids[left:right],
                    group.block_slotnums[left:right],
                    strict=True,
                ):
                    if (block + 1) * width <= start:
                        assert bytes(self.region(slab, slot)) == self.payload(
                            history, index, block
                        )
                for position in range(begin, end):
                    self.region(
                        group.append_slab_ids[position], group.append_slotnums[position]
                    )[:] = self.payload(
                        history, index, int(plan.positions[position]) // width
                    )
            token = int(plan.token_ids[end - 1]) + 1
            if plan.sampling_mask[row]:
                history.append(token)
            output.append(token)
        return forward_output(output)


def run(store, prompt, *, namespace="weights/layout-v1", **options):
    runner = ByteRunner()
    with Engine(
        runner,
        prefix_cache=True,
        prefix_cache_storage=store,
        prefix_cache_namespace=namespace,
        enable_overlap=False,
        scheduler_config=SchedulerConfig(prefill_chunk_size=4),
        **options,
    ) as engine:
        request = engine.submit(
            prompt, sampling_params=SamplingParams(max_output_tokens=3)
        )
        runner.histories[request.id] = list(prompt)
        while engine.has_work():
            engine.step()
        encoded = engine._metrics_prometheus_protobuf(model_name="test")
        assert (
            encoded is not None and b"resin:cache_storage_load_bytes_total" in encoded
        )
        return request.cached_tokens, request.take_result(), engine.metrics_snapshot()


def test_memory_storage_immutable_scatter_gather_and_capacity():
    store = MemoryCacheStorage(max_bytes=16)
    source = bytearray(b"abcdefgh")
    assert store.store(
        (b"a",), ((memoryview(source)[:3], memoryview(source)[3:]),)
    ) == (True,)
    source[:] = b"XXXXXXXX"
    store.store((b"a",), ((memoryview(source),),))
    target = bytearray(8)
    assert store.load((b"a",), ((memoryview(target)[:5], memoryview(target)[5:]),)) == (
        True,
    )
    assert target == b"abcdefgh"
    for key in (b"b", b"c"):
        store.store((key,), ((memoryview(source),),))
    assert sum(store.lookup((b"a", b"b", b"c"))) == 2
    assert store.lookup((b"c", b"absent")) == (True, False)
    assert store.load((b"c",), ((memoryview(bytearray(7)),),)) == (False,)


def test_randomized_memory_consistency():
    rng = random.Random(6324)
    store = MemoryCacheStorage(max_bytes=4096)
    oracle = {}
    for _ in range(400):
        key = bytes([rng.randrange(32)])
        payload = rng.randbytes(16)
        if rng.randrange(2):
            source = bytearray(payload)
            store.store((key,), ((memoryview(source),),))
            oracle.setdefault(key, payload)
            source[:] = bytes(16)
        else:
            target = bytearray(16)
            assert store.load((key,), ((memoryview(target),),)) == (key in oracle,)
            assert bytes(target) == oracle.get(key, bytes(16))


def test_randomized_grouped_restoration_with_storage_eviction():
    rng = random.Random(933)
    store = MemoryCacheStorage(max_bytes=512)
    for _ in range(32):
        prompt = list(range(1, rng.choice((9, 13, 17))))
        prompt += [rng.randrange(30, 40) for _ in range(rng.randrange(1, 7))]
        cached, output, _ = run(store, prompt)
        assert 0 <= cached < len(prompt) and cached % 4 == 0
        assert output == [prompt[-1] + step for step in (1, 2, 3)]


@pytest.mark.parametrize("namespace", ["weights/layout-v1", "other-model", None])
def test_cross_engine_prefix_bytes_and_namespaces(namespace):
    store = MemoryCacheStorage()
    prompt = list(range(1, 17))
    cached, _, metrics = run(store, prompt)
    assert cached == 0
    assert metrics.storage["store_bytes"] > 0
    cached, output, metrics = run(store, prompt + [99], namespace=namespace)
    assert output == [100, 101, 102]
    assert cached == (16 if namespace == "weights/layout-v1" else 0)
    assert bool(metrics.storage["load_bytes"]) == bool(cached)
    # A successful restore remains valid after the provider's creating Engine closes.
    assert metrics.storage["lookups"] > 0
    cached, output, _ = run(store, prompt)
    assert cached == 12 and output == [17, 18, 19]  # Replay the prompt's final token.


@pytest.mark.parametrize("failure", ["stale", "raise", "tail"])
def test_partial_load_never_publishes_a_checkpoint(failure):
    class FailingStorage(MemoryCacheStorage):
        failing = False

        def lookup(self, keys):
            result = super().lookup(keys)
            return (
                (*result[:-1], False) if self.failing and failure == "tail" else result
            )

        def load(self, keys, buffers):
            result = super().load(keys, buffers)
            if not self.failing or failure == "tail":
                return result
            if failure == "raise":
                raise OSError("retired transport failure")
            return (*result[:-1], False)

    store = FailingStorage()
    prompt = list(range(1, 17))
    run(store, prompt)
    store.failing = True
    cached, output, metrics = run(store, prompt + [99])
    assert cached == (12 if failure == "tail" else 0)
    assert output == [100, 101, 102]
    assert bool(metrics.storage["errors"]) == (failure == "raise")
    store.failing = False
    cached, output, _ = run(store, prompt + [99])
    assert cached == 16 and output == [100, 101, 102]


def test_failed_construction_does_not_bind_runner():
    class OnceRunner(ByteRunner):
        binds = 0

        def bind_resources(self, binding):
            self.binds += 1
            assert self.binds == 1
            super().bind_resources(binding)

    runner = OnceRunner()
    options: dict[str, Any] = dict(
        prefix_cache=True, prefix_cache_storage=MemoryCacheStorage()
    )
    with pytest.raises(ValueError, match="namespace"):
        Engine(runner, prefix_cache_namespace=b"", **options)
    assert runner.binds == 0
    Engine(runner, **options).close()
    assert runner.binds == 1


def test_cuda_storage_retires_producer_stream_and_multidimensional_copies():
    import torch

    if not torch.cuda.is_available():
        pytest.skip()
    store = MemoryCacheStorage()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        source = torch.full((32, 128), 17, dtype=torch.uint8, device="cuda")
        assert store.store((b"key",), ((source,),)) == (True,)
        source.zero_()
        target = torch.empty_like(source)
        assert store.load((b"key",), ((target,),)) == (True,)
        assert stream.query()
    torch.testing.assert_close(target, torch.full_like(target, 17))
