"""Storage integration preserves grouped cache bytes and native ownership."""

from __future__ import annotations

import gc
import hashlib
import os
import struct
import weakref

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from output_helpers import forward_output

from resin import (
    CacheGroupFamily,
    CacheGroupLayout,
    CacheLayout,
    CacheRetention,
    Engine,
    RunnerConfig,
    SamplingParams,
    SchedulerConfig,
)
from resin.cache_storage import CacheBuffer, CacheResult, MemoryCacheStorage


class ByteRunner:
    """A complete byte arena whose cache values have an independent oracle."""

    def __init__(self, *, signature=b"test-model/weights-1/cache-v1", capacity=96):
        self.cache_layout = CacheLayout(
            parent_capacity=capacity,
            groups=[
                CacheGroupLayout(4, blocks_per_slab=2),
                CacheGroupLayout(
                    2,
                    blocks_per_slab=4,
                    retention=CacheRetention.WINDOW,
                    window_tokens=8,
                    read_window_tokens=3,
                ),
                CacheGroupLayout(4, blocks_per_slab=2, family=CacheGroupFamily.STATE),
            ],
        )
        self.runner_config = RunnerConfig(
            cache_layout=self.cache_layout,
            continuation_prefill=True,
            prefill_chunk_size=4,
        )
        self.cache_storage_signature = signature
        self.arena = np.zeros((capacity + 1, 64), np.uint8)
        self.histories = {}
        self.calls = []
        self.transfers = []
        self.released = []

    def bind_resources(self, binding):
        assert binding.cache_layout == self.cache_layout

    def region(self, slab, slot):
        return memoryview(self.arena[int(slab), int(slot) * 16 : (int(slot) + 1) * 16])

    def cache_storage_buffers(self, transfer):
        self.transfers.append(transfer)
        return tuple(
            CacheBuffer(
                (self.region(slab, slot)[:5], self.region(slab, slot)[5:]), 16, "cpu"
            )
            for group in transfer.groups
            for slab, slot in zip(group.slab_ids, group.slotnums, strict=True)
        )

    def payload(self, history, group, block):
        end = (block + 1) * self.cache_layout.groups[group].block_tokens
        tokens = list(history)
        while len(tokens) < end:
            tokens.append(tokens[-1] + 1)
        data = (
            struct.pack("<II", group, block) + np.asarray(tokens[:end], "<u4").tobytes()
        )
        return hashlib.blake2b(data, digest_size=16).digest()

    def forward(self, plan):
        self.calls.append(plan)
        output = []
        for row, request_id in enumerate(plan.sequence_ids):
            history = self.histories[int(request_id)]
            start = int(plan.context_lengths[row])
            begin, end = map(int, plan.token_indptr[row : row + 2])
            for group in plan.cache_groups:
                index = group.group_index
                width = self.cache_layout.groups[index].block_tokens
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
                        ), (
                            "readable cache bytes differ from the no-storage prefix oracle",
                            int(request_id),
                            index,
                            block,
                            start,
                        )
                for position in range(begin, end):
                    logical = int(plan.positions[position]) // width
                    destination = self.region(
                        group.append_slab_ids[position], group.append_slotnums[position]
                    )
                    destination[:] = self.payload(history, index, logical)
            token = int(plan.token_ids[end - 1]) + 1
            if plan.sampling_mask[row]:
                history.append(token)
            output.append(token)
        return forward_output(output)

    def release(self, ids):
        self.released.extend(ids)


class Transfer:
    def __init__(self, provider, kind, keys, buffers, delay):
        self.provider, self.kind, self.keys, self.buffers = (
            provider,
            kind,
            keys,
            buffers,
        )
        self.remaining = delay
        self.result = None
        self.polls = self.waits = 0
        self.before = tuple(
            b"".join(bytes(region) for region in buffer.regions) for buffer in buffers
        )

    def finish(self):
        if self.result is not None:
            return self.result
        provider = self.provider
        if self.kind in provider.bad:
            failure = provider.bad[self.kind]
            if failure == "raise":
                raise RuntimeError("transport retirement is unproven")
            if failure == "invalid":
                return CacheResult(())
            self.result = CacheResult(
                (False,) * len(self.keys),
                "" if failure == "empty" else "retired transport failure",
            )
        elif self.kind == "lookup":
            self.result = CacheResult(
                tuple(key in provider.objects for key in self.keys)
            )
        elif self.kind == "store":
            current = tuple(
                b"".join(bytes(region) for region in buffer.regions)
                for buffer in self.buffers
            )
            assert current == self.before, (
                "native source pages changed before transport retirement"
            )
            for key, payload in zip(self.keys, current, strict=True):
                provider.objects.setdefault(key, payload)
            self.result = CacheResult((True,) * len(self.keys))
        else:
            if provider.drop_next_load:
                provider.objects.pop(self.keys[-1], None)
                provider.drop_next_load = False
            success = []
            for key, buffer in zip(self.keys, self.buffers, strict=True):
                payload = provider.objects.get(key)
                hit = payload is not None and len(payload) == buffer.nbytes
                success.append(hit)
                if hit:
                    offset = 0
                    for region in buffer.regions:
                        region[:] = payload[offset : offset + len(region)]
                        offset += len(region)
            self.result = CacheResult(tuple(success))
        return self.result

    def poll(self):
        self.polls += 1
        if self.remaining is None:
            return None
        if self.remaining:
            self.remaining -= 1
            return None
        return self.finish()

    def wait(self):
        self.waits += 1
        self.remaining = 0
        return self.finish()


class Store:
    def __init__(self, **delays):
        self.delays = delays
        self.objects = {}
        self.operations = []
        self.bad = {}
        self.drop_next_load = False
        self.closed = False

    def submit(self, kind, keys, buffers=()):
        operation = Transfer(self, kind, keys, buffers, self.delays.get(kind, 0))
        self.operations.append(operation)
        return operation

    def lookup(self, keys):
        return self.submit("lookup", keys)

    def load(self, keys, buffers):
        return self.submit("load", keys, buffers)

    def store(self, keys, buffers):
        return self.submit("store", keys, buffers)

    def release(self, kind=None):
        for operation in self.operations:
            if operation.result is None and (kind is None or operation.kind == kind):
                operation.remaining = 0

    def close(self):
        self.closed = True


def engine(runner, store=None, *, namespace="test/model-1", **options):
    return Engine(
        runner,
        prefix_cache=True,
        prefix_cache_storage=store,
        prefix_cache_namespace=namespace if store is not None else None,
        scheduler_config=SchedulerConfig(prefill_chunk_size=4),
        enable_overlap=False,
        **options,
    )


def submit(engine, runner, prompt, output=3):
    lease = engine.submit(
        prompt, sampling_params=SamplingParams(max_output_tokens=output)
    )
    runner.histories[lease.id] = list(prompt)
    return lease


def drain(engine, limit=500):
    events = []
    for _ in range(limit):
        if not engine.has_work():
            return events
        events.extend(engine.step())
    raise AssertionError(
        "engine did not retire storage and model work within its bounded schedule"
    )


def warm(store, prompt=None, **options):
    prompt = list(range(1, 18)) if prompt is None else prompt
    runner = ByteRunner()
    source = engine(runner, store, **options)
    lease = submit(source, runner, prompt)
    drain(source)
    assert lease.take_result() == list(range(prompt[-1] + 1, prompt[-1] + 4))
    return source, runner


def test_grouped_prefix_restores_after_trie_clear_and_updates_owned_metrics():
    store = Store()
    cached, runner = warm(store)
    before = cached.metrics_snapshot()
    assert before.storage["stored_objects"] > 0
    assert all(
        {group.group_index for group in transfer.groups} == {0, 1, 2}
        for transfer in runner.transfers
    )
    cached.clear_prefix_cache()
    previous_calls = len(runner.calls)
    restored = submit(cached, runner, list(range(1, 18)))
    drain(cached)
    assert restored.cached_tokens == 16
    assert restored.take_result() == [18, 19, 20]
    assert int(runner.calls[previous_calls].context_lengths[0]) == 16
    snapshot = cached.metrics_snapshot()
    stats = snapshot.storage
    assert stats["restored_tokens"] == 16
    assert stats["load_hits"] > 0 and stats["load_bytes"] == 16 * stats["load_hits"]
    assert (
        stats["pending_lookups"]
        == stats["pending_loads"]
        == stats["pending_stores"]
        == 0
    )
    assert stats["inflight_bytes"] == 0
    for kind in ("lookup", "load", "store"):
        retired = sum(
            operation.kind == kind and operation.result is not None
            for operation in store.operations
        )
        assert stats[f"{kind}_seconds"]["count"] == retired
        assert stats[f"{kind}_seconds"]["sum"] >= 0
    stats["loads"] = 0
    assert snapshot.storage["loads"] > 0
    assert before.storage["restored_tokens"] == 0
    wire = cached._metrics_prometheus_protobuf(
        model_name="test", rank=3, attn_dp_rank=1
    )
    for name in (
        "cache_storage_loads_total",
        "cache_storage_restored_tokens_total",
        "cache_storage_load_seconds",
    ):
        assert f"resin:{name}".encode() in wire
    # Source labels are carried through the actual native Prometheus encoder.
    assert b"cache_source" in wire and b"storage" in wire
    assert snapshot.requests["storage_cached_tokens"] == 16
    assert snapshot.requests["cached_tokens"] == 16
    local = submit(cached, runner, list(range(1, 18)))
    assert local.cached_tokens == 16
    drain(cached)
    assert local.take_result() == [18, 19, 20]
    later = cached.metrics_snapshot()
    assert later.requests["cached_tokens"] == 32
    assert later.requests["storage_cached_tokens"] == 16
    assert later.storage["restored_tokens"] == 16
    cached.close()
    assert not store.closed


@pytest.mark.parametrize(
    "namespace,signature,expected",
    [
        ("test/model-1", b"test-model/weights-1/cache-v1", 16),
        (b"test/model-1", b"test-model/weights-1/cache-v1", 16),
        ("other/model", b"test-model/weights-1/cache-v1", 0),
        ("test/model-1", b"test-model/weights-2/cache-v1", 0),
        ("test/model-1", b"test-model/weights-1/cache-v2", 0),
        (None, b"test-model/weights-1/cache-v1", 0),
    ],
)
def test_shared_provider_requires_explicit_compatible_identity(
    namespace, signature, expected
):
    store = Store()
    source, _ = warm(store)
    source.close()
    runner = ByteRunner(signature=signature, capacity=80)
    with engine(runner, store, namespace=namespace) as target:
        request = submit(target, runner, list(range(1, 18)))
        drain(target)
        assert request.cached_tokens == expected
        assert request.take_result() == [18, 19, 20]
    assert not store.closed and store.objects


def test_default_namespace_is_private_to_each_engine():
    store = Store()
    source, _ = warm(store, namespace=None)
    source.close()
    runner = ByteRunner()
    with engine(runner, store, namespace=None) as target:
        request = submit(target, runner, list(range(1, 18)))
        drain(target)
        assert request.cached_tokens == 0
        assert request.take_result() == [18, 19, 20]


def test_warm_device_hit_republishes_objects_evicted_by_another_store_client():
    store = MemoryCacheStorage(max_bytes=512)
    cached, runner = warm(store)
    before = cached.metrics_snapshot().storage["lookups"]
    # A separate client fills the same bounded provider, evicting every prefix.
    payload = b"x" * store.max_bytes
    assert store.store(
        (b"another-client",), (CacheBuffer((payload,), len(payload), "cpu"),)
    ).wait().success == (True,)
    # Recompute the last chunk of the same aligned prefix from a warm device
    # hit, so normal commit publishes that checkpoint again.
    reused = submit(cached, runner, list(range(1, 17)))
    assert reused.cached_tokens == 12
    drain(cached)
    assert reused.take_result() == [17, 18, 19]
    # Re-publication must not depend on a prior cold lookup discovering eviction.
    assert cached.metrics_snapshot().storage["lookups"] == before
    cached.clear_prefix_cache()
    restored = submit(cached, runner, list(range(1, 18)))
    drain(cached)
    assert restored.cached_tokens == 16
    assert restored.take_result() == [18, 19, 20]
    cached.close()


@pytest.mark.parametrize("boundary", ("submit", "poll", "wait"))
def test_metadata_only_lookup_exceptions_fallback_without_poisoning(boundary):
    class FailingLookup(Store):
        def lookup(self, keys):
            operation = super().lookup(keys)
            if boundary == "submit":
                raise RuntimeError("metadata submission failed")
            return operation

    store = FailingLookup(lookup=None if boundary == "wait" else 0)
    store.bad["lookup"] = "raise"
    runner = ByteRunner()
    target = engine(runner, store)
    request = submit(target, runner, list(range(1, 18)))
    if boundary == "wait":
        target.step()
        target.close()
        assert request.cancelled and request.take_result() == []
    else:
        drain(target)
        assert request.take_result() == [18, 19, 20]
        assert target.metrics_snapshot().storage["errors"] > 0
        assert target.metrics_snapshot().storage["misses"] > 0
        target.close()
    assert not target.poisoned
    assert target.cache_free_parent_count == runner.cache_layout.parent_capacity
    assert not store.closed


def test_cancelled_storage_materialization_is_not_a_completed_request_cache_credit():
    store = Store()
    source, _ = warm(store)
    source.close()
    runner = ByteRunner()
    with engine(runner, store) as target:
        request = submit(target, runner, list(range(1, 18)), output=8)
        events = target.step()
        assert request.cached_tokens == 16 and events[0].token_id == 18
        assert target.cancel(request)
        drain(target)
        snapshot = target.metrics_snapshot()
        assert snapshot.storage["restored_tokens"] == 16
        assert snapshot.requests["storage_cached_tokens"] == 0
        assert snapshot.requests["cached_tokens"] == 0
        assert snapshot.requests["prefill_cache_tokens"] == 16
        assert request.take_result() == [18]


def test_disappearing_object_falls_back_without_publishing_partial_group_state():
    store = Store()
    source, _ = warm(store)
    source.close()
    store.drop_next_load = True
    runner = ByteRunner()
    with engine(runner, store) as target:
        request = submit(target, runner, list(range(1, 18)))
        drain(target)
        assert request.cached_tokens == 0
        assert request.take_result() == [18, 19, 20]
        stats = target.metrics_snapshot().storage
        assert stats["load_hits"] > 0 and stats["load_misses"] == 1
        assert stats["restored_tokens"] == 0
        assert int(runner.calls[0].context_lengths[0]) == 0
        assert not target.poisoned


@pytest.mark.parametrize("kind", ("lookup", "load"))
def test_ready_decoder_advances_while_storage_request_is_pending(kind):
    store = Store()
    cached, runner = warm(store)
    cached.clear_prefix_cache()
    decoder = submit(cached, runner, [91, 92, 93], output=8)
    for _ in range(20):
        if any(decoder.id == event.request_id for event in cached.step()):
            break
    else:
        raise AssertionError("the resident request never entered decode")
    store.delays[kind] = None
    waiting = submit(cached, runner, list(range(1, 18)))
    events = cached.step()
    for _ in range(3):
        events += cached.step()
    assert any(
        operation.kind == kind and operation.result is None
        for operation in store.operations
    )
    assert any(event.request_id == decoder.id for event in events)
    assert not any(event.request_id == waiting.id for event in events)
    store.release()
    store.delays.clear()
    drain(cached)
    assert waiting.take_result() == [18, 19, 20]
    assert decoder.take_result() == list(range(94, 102))
    cached.close()


def test_cancellation_keeps_borrowed_sources_and_load_destinations_pinned():
    store = Store()
    cached, runner = warm(store)
    cached.clear_prefix_cache()
    store.delays["load"] = None
    request = submit(cached, runner, list(range(1, 18)))
    cached.step()
    load = next(operation for operation in store.operations if operation.kind == "load")
    assert load.result is None
    assert cached.cancel(request)
    cached.clear_prefix_cache()
    occupied = cached.cache_occupied_block_count
    assert occupied > 0 and cached.has_work()
    # Other allocation must neither reuse nor expose the pending destinations.
    other = submit(cached, runner, list(range(201, 210)))
    cached.step()
    assert load.result is None and cached.cache_occupied_block_count >= occupied
    assert (
        tuple(
            b"".join(bytes(region) for region in buffer.regions)
            for buffer in load.buffers
        )
        == load.before
    )
    store.release()
    store.delays.clear()
    drain(cached)
    assert request.cancelled and request.take_result() == []
    assert other.take_result() == [210, 211, 212]
    cached.clear_prefix_cache()
    assert cached.cache_free_parent_count == runner.cache_layout.parent_capacity
    cached.close()


@pytest.mark.parametrize("cancel_source", (False, True))
def test_pending_exports_survive_clear_and_close_drains_without_closing_provider(
    cancel_source,
):
    store = Store(store=None)
    runner = ByteRunner()
    cached = engine(runner, store)
    request = submit(cached, runner, list(range(1, 18)))
    for _ in range(20):
        cached.step()
        if request.done or (
            cancel_source
            and any(operation.kind == "store" for operation in store.operations)
        ):
            break
    if cancel_source:
        assert not request.done and cached.cancel(request)
    else:
        assert request.done
    exports = [operation for operation in store.operations if operation.kind == "store"]
    assert exports and all(operation.result is None for operation in exports)
    cached.clear_prefix_cache()
    assert cached.cache_occupied_block_count > 0
    assert cached.metrics_snapshot().storage["pending_stores"] <= 2
    other = submit(cached, runner, list(range(201, 210)))
    for _ in range(20):
        cached.step()
        if other.done:
            break
    assert other.take_result() == [210, 211, 212]
    cached.close()
    assert all(
        operation.result is not None and operation.waits for operation in exports
    )
    assert cached.cache_free_parent_count == runner.cache_layout.parent_capacity
    assert store.objects and not store.closed


def test_submission_exception_retains_accepted_export_owners_in_quarantine():
    class UncertainStore(Store):
        def store(self, keys, buffers):
            super().store(keys, buffers)
            raise RuntimeError("submission raised after accepting native work")

    store = UncertainStore()
    runner = ByteRunner()
    target = engine(runner, store)
    submit(target, runner, list(range(1, 18)))
    with pytest.raises(RuntimeError, match="after accepting"):
        drain(target)
    assert target.poisoned and target.cache_occupied_block_count > 0
    assert any(
        operation.buffers for operation in store.operations if operation.kind == "store"
    )
    reason = target.poison_error
    polls = [operation.polls for operation in store.operations]
    with pytest.raises(RuntimeError) as repeated:
        target.step()
    assert reason in str(repeated.value)
    assert [operation.polls for operation in store.operations] == polls
    with pytest.raises(RuntimeError, match="quarantined|poisoned|storage"):
        target.close()


@pytest.mark.parametrize("unretired", (False, True))
def test_engine_destruction_releases_only_physically_retired_storage(unretired):
    releases = []

    class ObservedRunner(ByteRunner):
        def release(self, ids):
            releases.extend(ids)
            super().release(ids)

    class ImmediateLookup:
        def __init__(self, count):
            self.result = CacheResult((False,) * count)

        def poll(self):
            return self.result

        def wait(self):
            return self.result

    class BorrowedOperation:
        def __init__(self, buffers):
            self.buffers = buffers

        def poll(self):
            if unretired:
                raise RuntimeError("native storage access is still in flight")
            return None

        def wait(self):
            if unretired:
                raise RuntimeError("native storage access is still in flight")
            provider.waits += 1
            # Retirement reads the supplied bytes before releasing every owner.
            provider.saved = tuple(
                b"".join(bytes(region) for region in buffer.regions)
                for buffer in self.buffers
            )
            result = CacheResult((True,) * len(self.buffers))
            self.buffers = ()
            return result

    class WeakProvider:
        waits = 0
        saved: tuple[bytes, ...] = ()

        def lookup(self, keys):
            return ImmediateLookup(len(keys))

        def store(self, keys, buffers):
            operation = BorrowedOperation(buffers)
            # Native transports need not keep a second Python handle alive.
            self.operation = weakref.ref(operation)
            return operation

        def load(self, keys, buffers):
            raise AssertionError("the empty store cannot load objects")

    provider = WeakProvider()
    runner = ObservedRunner()
    allocation, runner_owner = weakref.ref(runner.arena), weakref.ref(runner)
    target = engine(runner, provider)
    request = submit(target, runner, list(range(1, 18)))
    if unretired:
        with pytest.raises(RuntimeError, match="still in flight"):
            target.step()
        assert target.poisoned
    else:
        assert target.step() == []
    del runner
    gc.collect()
    assert provider.operation() is not None and allocation() is not None
    del target
    gc.collect()
    if unretired:
        assert provider.operation() is not None
        assert allocation() is not None and runner_owner() is not None
        assert provider.waits == 0 and releases == []
    else:
        assert provider.operation() is None
        assert allocation() is None and runner_owner() is None
        assert provider.waits == 1 and provider.saved
        assert releases == [request.id] and request.cancelled


def test_storage_pending_limits_do_not_admit_unbounded_transfer_owners():
    store = Store(lookup=None, store=None)
    runner = ByteRunner()
    cached = engine(runner, store, prefix_cache_max_pending=1)
    requests = [
        submit(cached, runner, list(range(start, start + 17)))
        for start in (1, 101, 201)
    ]
    for _ in range(5):
        cached.step()
        stats = cached.metrics_snapshot().storage
        assert stats["pending_lookups"] <= 1
        assert stats["pending_stores"] <= 1
    assert not any(request.done for request in requests)
    store.release()
    store.delays.clear()
    drain(cached)
    assert all(request.done for request in requests)
    cached.close()


def test_imports_obey_pending_and_running_request_limits():
    store = Store()
    source, _ = warm(store)
    source.close()
    store.delays["load"] = None
    runner = ByteRunner()
    target = engine(runner, store, prefix_cache_max_pending=1, max_running_requests=1)
    requests = [submit(target, runner, list(range(1, 18))) for _ in range(3)]
    for _ in range(4):
        target.step()
        snapshot = target.metrics_snapshot()
        assert snapshot.storage["pending_loads"] <= 1
        assert snapshot.requests["running_requests"] <= 1
    assert not runner.calls
    store.release()
    store.delays.clear()
    drain(target)
    assert [request.take_result() for request in requests] == [[18, 19, 20]] * 3
    target.close()


def test_cancelled_lookup_can_retire_without_readmitting_the_request():
    store = Store(lookup=None)
    runner = ByteRunner()
    target = engine(runner, store)
    request = submit(target, runner, list(range(1, 18)))
    assert target.step() == []
    assert target.cancel(request)
    store.release()
    drain(target)
    assert request.cancelled and request.take_result() == []
    assert not runner.calls
    assert target.cache_free_parent_count == runner.cache_layout.parent_capacity
    target.close()


def test_storage_configuration_is_explicit_and_unconfigured_hooks_are_unused(
    monkeypatch,
):
    with pytest.raises(ValueError, match="prefix"):
        Engine(ByteRunner(), prefix_cache_storage=Store())
    with pytest.raises(ValueError, match="pending"):
        engine(ByteRunner(), Store(), prefix_cache_max_pending=0)
    with pytest.raises(ValueError, match="requires"):
        Engine(ByteRunner(), prefix_cache_namespace="shared")
    runner = ByteRunner()

    def forbidden(_):
        raise AssertionError("unconfigured storage invoked a runner buffer callback")

    monkeypatch.setattr(runner, "cache_storage_buffers", forbidden)
    runner.cache_storage_signature = object()
    with engine(runner) as target:
        request = submit(target, runner, list(range(1, 18)))
        drain(target)
        assert request.take_result() == [18, 19, 20]
        assert target.metrics_snapshot().storage is None
    with pytest.raises(TypeError, match="signature"):
        engine(runner, Store())


@pytest.mark.parametrize("failure", ("error", "empty", "invalid", "raise"))
def test_terminal_errors_fallback_but_unproven_or_malformed_completions_quarantine(
    failure,
):
    store = Store()
    source, _ = warm(store)
    source.close()
    store.bad["load"] = failure
    runner = ByteRunner()
    target = engine(runner, store)
    request = submit(target, runner, list(range(1, 18)))
    if failure in ("error", "empty"):
        drain(target)
        assert request.take_result() == [18, 19, 20]
        assert target.metrics_snapshot().storage["errors"] > 0
        assert not target.poisoned
        target.close()
    else:
        with pytest.raises(
            (TypeError, RuntimeError), match="boolean result|retirement"
        ):
            target.step()
        assert target.poisoned and not request.done
        assert target.cache_occupied_block_count > 0
        reason = target.poison_error
        polls = [operation.polls for operation in store.operations]
        with pytest.raises(RuntimeError) as repeated:
            target.step()
        assert reason in str(repeated.value)
        assert [operation.polls for operation in store.operations] == polls
        with pytest.raises(RuntimeError, match="quarantined|poisoned|storage"):
            target.close()
        assert not store.closed


def test_absent_and_disabled_storage_metrics_do_not_poll_provider():
    with engine(ByteRunner()) as target:
        snapshot = target.metrics_snapshot()
        assert snapshot.storage is None
        wire = target._metrics_prometheus_protobuf(model_name="test")
        assert b"resin:cache_storage_" not in wire
    store = Store()
    with engine(ByteRunner(), store, enable_metrics=False) as target:
        assert target.metrics_snapshot() is None
        assert target._metrics_prometheus_protobuf(model_name="test") is None
        assert store.operations == []


@settings(
    max_examples=int(os.environ.get("RESIN_STORAGE_FUZZ_EXAMPLES", "50")),
    deadline=None,
    derandomize=True,
)
@given(
    st.lists(st.tuples(st.integers(0, 3), st.integers(0, 5)), min_size=5, max_size=18)
)
def test_randomized_transport_and_cache_clears_match_no_storage_oracle(actions):
    store = Store(lookup=2, load=2, store=2)
    runner, oracle_runner = ByteRunner(), ByteRunner()
    cached, oracle = engine(runner, store), engine(oracle_runner)
    pairs = []
    try:
        prompt = list(range(1, 18))
        seed, reference = (
            submit(cached, runner, prompt),
            submit(oracle, oracle_runner, prompt),
        )
        drain(cached)
        drain(oracle)
        assert seed.take_result() == reference.take_result()
        cached.clear_prefix_cache()
        oracle.clear_prefix_cache()
        pairs.append(
            (submit(cached, runner, prompt), submit(oracle, oracle_runner, prompt))
        )
        for action, branch in actions:
            store.delays.update(
                lookup=branch % 4,
                load=(branch + action) % 5,
                store=(branch + 2 * action) % 4,
            )
            if action == 0 and len(pairs) < 7:
                start = 100 * branch + 1
                prompt = list(range(start, start + 13 + 4 * (branch % 2)))
                pairs.append(
                    (
                        submit(cached, runner, prompt),
                        submit(oracle, oracle_runner, prompt),
                    )
                )
            elif action == 1:
                cached.clear_prefix_cache()
                oracle.clear_prefix_cache()
            elif action == 2 and store.objects:
                key = next(iter(store.objects))
                store.objects.pop(key)
            for _ in range(1 + branch % 3):
                cached.step()
                oracle.step()
        store.release()
        store.delays.clear()
        drain(cached)
        drain(oracle)
        assert [left.take_result() for left, _ in pairs] == [
            right.take_result() for _, right in pairs
        ]
        assert not cached.poisoned
        cached.clear_prefix_cache()
        oracle.clear_prefix_cache()
        assert cached.cache_free_parent_count == runner.cache_layout.parent_capacity
        assert (
            oracle.cache_free_parent_count == oracle_runner.cache_layout.parent_capacity
        )
    finally:
        cached.close()
        oracle.close()
