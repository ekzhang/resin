from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from random import Random
from threading import Event
from time import sleep
from typing import Any

import pytest
import torch
from test_flat_cache import RecordingCachedRunner

from resin import CacheGroupLayout, CacheLayout, Engine, SamplingParams
from resin.cache_storage import MemoryCacheStorage


def make_engine(store, fill=0, limit=2):
    runner = RecordingCachedRunner(
        CacheLayout(parent_capacity=32, groups=[CacheGroupLayout(2)])
    )
    arena = torch.full((33,), fill, dtype=torch.uint8)

    def buffers(groups, *, prefix_length):
        return tuple(
            {"value": arena[slab : slab + 1]} for _, rows in groups for slab, _ in rows
        )

    setattr(runner, "cache_storage_buffers", buffers)
    options: dict[str, Any] = dict(prefix_cache=True, prefix_cache_max_inflight=limit)
    engine = Engine(runner, prefix_cache_storage=store, **options)
    return engine, runner, arena


def test_memory_bytes_capacity_and_scatter_gather():
    store, known = MemoryCacheStorage(max_bytes=10), {}
    for i, key in enumerate(Random(7).choices((b"a", b"b", b"c"), k=400)):
        source = torch.tensor([i % 256, 255 - i % 256], dtype=torch.uint8)
        expected = known[key] if store.lookup((key,)).result()[0] else bytes(source)
        known[key] = expected
        assert store.store((key,), ({"x": source[:1], "y": source[1:]},)).result()
        source.fill_(122)
        target = torch.empty(2, dtype=torch.uint8)
        assert store.load((key,), ({"y": target[1:], "x": target[:1]},)).result()
        assert bytes(target) == expected and store.nbytes <= 10
        assert sum(store.lookup((b"a", b"b", b"c")).result()) <= 2
    assert not store.load((key,), ({"different": target},)).result()
    store.close()
    store, other = MemoryCacheStorage(), MemoryCacheStorage()
    for provider, cached in ((store, 0), (store, 2), (other, 0)):
        engine, _, arena = make_engine(provider, 73 if not cached else 0)
        with engine:
            request = engine.submit(
                [1, 2, 3], sampling_params=SamplingParams(max_output_tokens=2)
            )
            while engine.has_work():
                engine.step()
                sleep(0.001)
            assert request.cached_tokens == cached
            assert not cached or bool((arena == 73).any())
    store.close()
    other.close()


@pytest.mark.parametrize("limit", [1, 4])
def test_storage_inflight_limit(limit):
    store, pending = MemoryCacheStorage(), []

    def unavailable(*args, **kwargs):
        raise NotImplementedError("cache_storage_buffers unavailable")

    def empty(*args, **kwargs):
        return ()

    for missing in ("hook", "implementation"):
        unsupported = RecordingCachedRunner(
            CacheLayout(parent_capacity=8, groups=[CacheGroupLayout(2)])
        )
        if missing != "hook":
            setattr(unsupported, "cache_storage_buffers", unavailable)
        with pytest.raises(NotImplementedError, match="cache_storage"):
            Engine(unsupported, prefix_cache=True, prefix_cache_storage=store)
        assert unsupported.bindings == []
        setattr(unsupported, "cache_storage_buffers", empty)
        Engine(unsupported, prefix_cache=True, prefix_cache_storage=store).close()

    def lookup(keys):
        future = Future()
        pending.append((future, keys))
        return future

    setattr(store, "lookup", lookup)
    with pytest.raises(ValueError, match="prefix_cache_max_inflight"):
        make_engine(store, limit=0)
    engine, _, _ = make_engine(store, limit=limit)
    with engine:
        try:
            requests = [
                engine.submit(
                    [1, 2, 3], sampling_params=SamplingParams(max_output_tokens=1)
                )
                for _ in range(limit + 1)
            ]
            assert engine.metrics_snapshot().cache["storage_pending"] == limit
            assert len(pending) == limit
            for _ in range(3):
                engine.step()
            assert requests[-1].done
        finally:
            for future, keys in pending:
                future.set_result((False,) * len(keys))
    store.close()


@pytest.mark.parametrize("delay", ["lookup", "load", "store"])
def test_async_progress_cancellation_and_retirement(delay):
    store, gate, pending = MemoryCacheStorage(), Event(), []
    params = SamplingParams(max_output_tokens=32)
    warm, _, _ = make_engine(store, 73)
    warm.submit([1, 2], sampling_params=params)
    warm.step()
    warm.close()
    engine, runner, arena = make_engine(store)
    running = engine.submit([9, 9], sampling_params=params)
    engine.step()
    method = getattr(store, delay)
    with ThreadPoolExecutor(3) as workers:

        def copy(*args):
            if not gate.wait(2):
                return False
            value = method(*args).result()
            return False if delay == "load" else value

        def submit(*args):
            future = workers.submit(copy, *args)
            pending.append((future, args[1] if len(args) == 2 else ()))
            return future

        setattr(store, delay, submit)
        waiting = engine.submit([1, 2, 3], sampling_params=params)
        while not pending:
            engine.step()
            sleep(0.001)
        start = len(runner.plans)
        for _ in range(3):
            engine.step()
        assert any(running.id in p.sequence_ids for p in runner.plans[start:])
        held = tuple(
            r for _, batch in pending for regions in batch for r in regions.values()
        )
        before = tuple(bytes(r) for r in held)
        assert engine.cancel(running)
        if delay != "lookup":
            assert engine.cancel(waiting)
        engine.clear_prefix_cache()
        engine.submit([7] * 8, sampling_params=SamplingParams(max_output_tokens=1))
        engine.step()
        for slab in runner.plans[-1].cache_groups[0].append_slab_ids:
            arena[int(slab)] = 7
        assert before == tuple(bytes(r) for r in held)
        closing = workers.submit(engine.close)
        with pytest.raises(TimeoutError):
            closing.result(0.02)
        gate.set()
        closing.result(1)
    assert delay != "load" or waiting.cached_tokens == 0
    if delay == "store":
        assert all(future.result() for future, _ in pending)
    store.close()
