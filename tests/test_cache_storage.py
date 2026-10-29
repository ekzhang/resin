from concurrent.futures import ThreadPoolExecutor, TimeoutError
from random import Random
from threading import Event
from time import sleep
from typing import Any

import pytest
from test_flat_cache import RecordingCachedRunner

from resin import CacheGroupLayout, CacheLayout, Engine, SamplingParams
from resin.cache_storage import MemoryCacheStorage


def make_engine(store, fill=0):
    runner = RecordingCachedRunner(
        CacheLayout(parent_capacity=32, groups=[CacheGroupLayout(2)])
    )
    arena = bytearray([fill] * 33)

    def buffers(groups):
        view = memoryview(arena)
        return tuple((view[slab : slab + 1],) for slab, _ in groups[0][1])

    setattr(runner, "cache_storage_buffers", buffers)
    options: dict[str, Any] = dict(prefix_cache=True, prefix_cache_namespace="m")
    engine = Engine(runner, prefix_cache_storage=store, **options)
    return engine, runner, arena


def test_memory_bytes_capacity_and_scatter_gather():
    store, known = MemoryCacheStorage(max_bytes=6), {}
    for i, key in enumerate(Random(7).choices((b"a", b"b", b"c"), k=400)):
        source = bytearray([i % 256, 255 - i % 256])
        expected = known[key] if store.lookup((key,)).result()[0] else bytes(source)
        known[key], view = expected, memoryview(source)
        assert store.store((key,), ((view[:1], view[1:]),)).result()
        source[:] = b"zz"
        target = bytearray(2)
        assert store.load((key,), ((memoryview(target),),)).result()
        assert target == expected and store.nbytes <= 6
        assert sum(store.lookup((b"a", b"b", b"c")).result()) <= 2
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
        held = tuple(r for _, batch in pending for regions in batch for r in regions)
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
