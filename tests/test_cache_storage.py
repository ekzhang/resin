from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from random import Random
from threading import Event
from time import sleep
from typing import Any
from weakref import ref

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
    store, known, pending = MemoryCacheStorage(max_bytes=10), {}, []
    for i, key in enumerate(Random(7).choices((b"a", b"b", b"c"), k=400)):
        source = torch.tensor(
            [(i + j) % 256 for j in range(2 + i % 6)], dtype=torch.uint8
        )
        expected = (
            known[key] if store.lookup((key,)).result()[0] else bytes(source.tolist())
        )
        known[key] = expected
        assert store.store((key,), ({"x": source[:1], "y": source[1:]},)).result()
        for future, target, contents in pending:
            assert future.result() and bytes(target.tolist()) == contents
        pending.clear()
        source.fill_(122)
        for _ in range(2):
            target = torch.empty(len(expected), dtype=torch.uint8)
            pending.append(
                (
                    store.load((key,), ({"y": target[1:], "x": target[:1]},)),
                    target,
                    expected,
                )
            )
        assert store.nbytes <= 10
        assert sum(store.lookup((b"a", b"b", b"c")).result()) <= 2
    for future, target, contents in pending:
        assert future.result() and bytes(target.tolist()) == contents
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


def test_queued_reads_retire_before_eviction_and_close():
    class Fence:
        def __init__(self):
            self.entered, self.ready, self.copies = Event(), Event(), []

        def synchronize(self):
            self.entered.set()
            assert self.ready.wait(2)
            for region, source in self.copies:
                region.copy_(source)
            self.copies.clear()

    class DelayedCopies(MemoryCacheStorage):
        def _copy(self, pairs, load):
            if load:
                fence.copies.extend(
                    (region, self.pool[offset : offset + region.nbytes])
                    for region, offset in pairs
                )
            else:
                super()._copy(pairs, load)

    store, fence, launch = DelayedCopies(max_bytes=4), Fence(), Event()
    original = torch.tensor([31, 47], dtype=torch.uint8)
    assert store.store((b"a",), ({"x": original},)).result()
    store.streams["fake"] = fence
    outputs = [torch.full_like(original, 99) for _ in range(3)]
    blocker = store.executor.submit(launch.wait)
    reads = [store.load((b"a",), ({"x": output},)) for output in outputs]
    assert reads[-1].cancel()
    reserve = store._reserve

    def retire_before_reuse(size, protected):
        assert all(future.done() for future in reads)
        assert all(torch.equal(output, original) for output in outputs[:2])
        return reserve(size, protected)

    setattr(store, "_reserve", retire_before_reuse)
    replacement = store.store((b"b",), ({"x": original + 1},))
    try:
        launch.set()
        assert fence.entered.wait(2)
        assert len(fence.copies) == 2 and not any(f.done() for f in reads[:2])
        assert store.lookup((b"a", b"b")).result() == (True, False)
        assert not replacement.done()
    finally:
        fence.ready.set()
    assert replacement.result(2) and blocker.result(2)
    assert reads[-1].cancelled() and outputs[-1].tolist() == [99, 99]
    assert store.lookup((b"a", b"b")).result() == (False, True)

    launch.clear()
    fence.entered.clear()
    fence.ready.clear()
    store.executor.submit(launch.wait)
    closing_reads = [store.load((b"b",), ({"x": out},)) for out in outputs[:2]]
    with ThreadPoolExecutor(1) as workers:
        closing = workers.submit(store.close)
        try:
            launch.set()
            assert fence.entered.wait(2)
            with pytest.raises(TimeoutError):
                closing.result(0.02)
            assert not any(future.done() for future in closing_reads)
        finally:
            fence.ready.set()
        closing.result(2)
    assert all(future.result() for future in closing_reads)
    assert all(torch.equal(output, original + 1) for output in outputs[:2])
    with pytest.raises(RuntimeError, match="closed"):
        store.load((b"b",), ({"x": outputs[0]},))


def test_failed_read_fence_preserves_unretired_buffers():
    class BrokenFence:
        def synchronize(self):
            raise RuntimeError("retirement is unknown")

    store = MemoryCacheStorage(max_bytes=4)
    value = torch.tensor([31, 47], dtype=torch.uint8)
    assert store.store((b"a",), ({"x": value},)).result()
    store.streams["fake"] = BrokenFence()
    destination = torch.empty_like(value)
    retained = ref(destination)
    assert store.load((b"a",), ({"x": destination},)).result() is None
    assert store.halted
    assert store.store((b"b",), ({"x": value},)).result() is False
    untouched = torch.full_like(value, 99)
    assert store.load((b"a",), ({"x": untouched},)).result() is False
    assert untouched.tolist() == [99, 99] and b"a" in store.objects
    del destination
    store.close()
    assert retained() is not None


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
