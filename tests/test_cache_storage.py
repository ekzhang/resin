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


def make_engine(
    store,
    fill=0,
    limit=2,
    *,
    policy="write_through",
    running=32,
    exports=None,
    capacity=32,
):
    runner = RecordingCachedRunner(
        CacheLayout(parent_capacity=capacity, groups=[CacheGroupLayout(2)])
    )
    arena = torch.full((capacity + 1,), fill, dtype=torch.uint8)

    def buffers(groups, *, prefix_length):
        if groups and exports is not None:
            exports.append(prefix_length)
        return tuple(
            {"value": arena[slab : slab + 1]} for _, rows in groups for slab, _ in rows
        )

    setattr(runner, "cache_storage_buffers", buffers)
    options: dict[str, Any] = dict(prefix_cache=True, max_running_requests=running)
    if limit is not None:
        options["prefix_cache_max_inflight"] = limit
    if policy is not None:
        options["prefix_cache_storage_policy"] = policy
    engine = Engine(runner, prefix_cache_storage=store, **options)
    return engine, runner, arena


@pytest.mark.parametrize(("capacity", "count"), [(4, 2), (8, 4)])
def test_completed_imports_leave_room_for_prefill_and_long_generation(capacity, count):
    store = MemoryCacheStorage(max_bytes=4096)
    prefixes = [list(range(i * 10, i * 10 + 4)) for i in range(count)]
    for prefix in prefixes:
        warm, _, _ = make_engine(store, 73, capacity=capacity)
        with warm:
            request = warm.submit(
                prefix, sampling_params=SamplingParams(max_output_tokens=1)
            )
            while not request.done:
                warm.step()
                sleep(0.001)
    engine, runner, _ = make_engine(store, limit=None, capacity=capacity, running=count)
    with engine:
        requests = [
            engine.submit(
                prefix + [99], sampling_params=SamplingParams(max_output_tokens=4)
            )
            for prefix in prefixes
        ]
        for _ in range(1000):
            engine.next_phase()  # Retire every load before any model work.
            if all(request.cached_tokens == 4 for request in requests):
                break
            sleep(0.001)
        assert all(request.cached_tokens == 4 for request in requests)
        assert not runner.plans and engine.metrics_snapshot().cache["free_slabs"] == 0
        engine.step()
        assert runner.plans and not runner.released  # Waiting victims never ran.
        for _ in range(1000):
            if not engine.has_work():
                break
            engine.step()
            sleep(0.001)
        assert all(
            request.done and request.take_result() == [100, 101, 102, 103]
            for request in requests
        )
        assert engine.metrics_snapshot().requests["retracted_requests"] > 0
    store.close()


def test_memory_bytes_capacity_and_scatter_gather():
    store, known = MemoryCacheStorage(max_bytes=10), {}
    for i, key in enumerate(Random(7).choices((b"a", b"b", b"c"), k=400)):
        source = torch.tensor(
            [(i + j) % 256 for j in range(2 + i % 6)], dtype=torch.uint8
        )
        expected = (
            known[key] if store.lookup((key,)).result()[0] else bytes(source.tolist())
        )
        known[key] = expected
        assert store.store((key,), ({"x": source[:1], "y": source[1:]},)).result()
        source.fill_(122)
        target = torch.empty(len(expected), dtype=torch.uint8)
        assert store.load((key,), ({"y": target[1:], "x": target[:1]},)).result()
        assert bytes(target.tolist()) == expected and store.nbytes <= 10
        assert sum(store.lookup((b"a", b"b", b"c")).result()) <= 2
    assert not store.load((key,), ({"different": target},)).result()
    store.close()
    store = MemoryCacheStorage(max_bytes=0)
    assert store.store((b"",), ({},)).result()
    assert store.lookup((b"",)).result() == (True,)
    assert store.load((b"",), ({},)).result() and store.nbytes == 0
    store.close()
    store = MemoryCacheStorage(max_bytes=6)
    regions = ({"x": torch.tensor([1], dtype=torch.uint8)},) * 3
    assert store.store((b"a", b"b"), regions[:2]).result()
    assert store.store((b"a", b"c"), regions[:2]).result()
    assert store.lookup((b"a", b"b", b"c")).result() == (True, False, True)
    assert not store.store((b"a", b"c", b"d"), regions).result()
    assert store.lookup((b"a", b"c", b"d")).result() == (True, True, False)
    store.close()
    store, other = (
        MemoryCacheStorage(max_bytes=4096),
        MemoryCacheStorage(max_bytes=4096),
    )
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


@pytest.mark.parametrize(
    ("running", "limit", "expected"), [(1, None, 2), (4, None, 4), (4, 1, 1)]
)
def test_storage_inflight_limit(running, limit, expected):
    store, pending = MemoryCacheStorage(max_bytes=4096), []

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
        with pytest.raises(ValueError, match="prefix_cache_storage_policy"):
            Engine(
                unsupported,
                prefix_cache=True,
                prefix_cache_storage=store,
                prefix_cache_storage_policy="invalid",  # ty: ignore[invalid-argument-type]
            )
        assert unsupported.bindings == []
        Engine(unsupported, prefix_cache=True, prefix_cache_storage=store).close()

    def lookup(keys):
        future = Future()
        pending.append((future, keys))
        return future

    setattr(store, "lookup", lookup)
    with pytest.raises(ValueError, match="prefix_cache_max_inflight"):
        make_engine(store, limit=0)
    engine, _, _ = make_engine(store, limit=limit, running=running)
    with engine:
        try:
            requests = [
                engine.submit(
                    [1, 2, 3], sampling_params=SamplingParams(max_output_tokens=1)
                )
                for _ in range(expected + 1)
            ]
            assert engine.metrics_snapshot().cache["storage_pending"] == expected
            assert len(pending) == expected
            for _ in range(3):
                engine.step()
            assert requests[-1].done
        finally:
            for future, keys in pending:
                future.set_result((False,) * len(keys))
    store.close()


@pytest.mark.parametrize("policy", [None, "write_through"])
def test_storage_write_policy(policy):
    store, exports = MemoryCacheStorage(max_bytes=4096), []
    engine, _, _ = make_engine(store, policy=policy, exports=exports)
    params = SamplingParams(max_output_tokens=1)
    with engine:
        cold = engine.submit([1, 2, 3], sampling_params=params)
        while engine.has_work():
            engine.step()
            sleep(0.001)
        assert cold.cached_tokens == 0
        assert exports == ([] if policy is None else [2])
        exports.clear()
        reused = engine.submit([1, 2, 7, 8, 9], sampling_params=params)
        assert reused.cached_tokens == 2
        if policy is None:
            assert exports == [2]
        while engine.has_work():
            engine.step()
            sleep(0.001)
        if policy is None:
            assert exports == [2]
    store.close()


@pytest.mark.parametrize("delay", ["lookup", "load", "store"])
def test_async_progress_cancellation_and_retirement(delay):
    store, gate, pending = MemoryCacheStorage(max_bytes=4096), Event(), []
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
