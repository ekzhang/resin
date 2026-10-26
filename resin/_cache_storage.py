"""Engine-side coordination for optional immutable prefix storage.

The native Engine owns page leases. This module owns transport operations and
their borrowed buffers until the provider establishes physical completion.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from resin._resin import CacheGroupFamily, _HistogramAccumulator
from resin.cache_storage import CacheResult


@dataclass(frozen=True, slots=True)
class _Group:
    group_index: int
    first_block: int
    slab_ids: tuple[int, ...]
    slotnums: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _Transfer:
    resident_tokens: int
    groups: tuple[_Group, ...]


def _transfer(value):
    length, groups = value
    return _Transfer(
        length,
        tuple(
            _Group(index, first, tuple(slabs), tuple(slots))
            for index, first, slabs, slots in groups
        ),
    )


class _Keys:
    def __init__(self, tokens, maximum, groups, namespace):
        self.maximum = maximum
        self.groups = groups
        self.keys = {}
        ends = sorted(
            {
                end
                for group in groups
                for end in range(group.block_tokens, maximum + 1, group.block_tokens)
            }
        )
        encoded = np.asarray(tokens, dtype="<u4").tobytes()
        digest = hashlib.sha256()
        previous = 0
        prefixes = {}
        for end in ends:
            digest.update(encoded[previous * 4 : end * 4])
            prefixes[end] = digest.digest()
            previous = end
        for index, group in enumerate(groups):
            self.keys[index] = tuple(
                hashlib.sha256(
                    namespace + struct.pack("<II", index, end) + prefixes[end]
                ).digest()
                for end in range(group.block_tokens, maximum + 1, group.block_tokens)
            )

    def at(self, index, first, count):
        return self.keys[index][first : first + count]

    def manifest(self, length):
        keys = []
        for index, group in enumerate(self.groups):
            total = length // group.block_tokens
            if group.family == CacheGroupFamily.STATE:
                count = min(total, 2)
            elif group.window_resident_block_count is not None:
                count = min(total, group.window_resident_block_count)
            else:
                count = total
            keys.extend(self.at(index, total - count, count))
        return tuple(keys)

    def longest(self, present, minimum, alignment):
        maximum = self.maximum
        # Full-history groups need an uninterrupted chain. Bound the search
        # once, rather than rescanning each earlier checkpoint's whole history.
        for index, group in enumerate(self.groups):
            if (
                group.family != CacheGroupFamily.STATE
                and group.window_resident_block_count is None
            ):
                contiguous = 0
                for key in self.keys[index]:
                    if not present.get(key, False):
                        break
                    contiguous += group.block_tokens
                maximum = min(maximum, contiguous)
        for length in range(maximum // alignment * alignment, minimum, -alignment):
            valid = True
            for index, group in enumerate(self.groups):
                if (
                    group.family != CacheGroupFamily.STATE
                    and group.window_resident_block_count is None
                ):
                    continue
                total = length // group.block_tokens
                count = min(
                    total,
                    2
                    if group.family == CacheGroupFamily.STATE
                    else group.window_resident_block_count,
                )
                if not all(
                    present.get(key, False)
                    for key in self.at(index, total - count, count)
                ):
                    valid = False
                    break
            if valid:
                return length
        return 0


@dataclass(slots=True)
class _Pending:
    kind: str
    operation: object
    keys: tuple[bytes, ...]
    buffers: tuple
    context: object
    started: float


class _LookupFailure:
    def __init__(self, count, error):
        self.result = CacheResult((False,) * count, str(error))

    def poll(self):
        return self.result

    def wait(self):
        return self.result


class _StorageRuntime:
    def __init__(self, storage, runner, layout, namespace, max_pending, metrics):
        for method in ("lookup", "load", "store"):
            if not callable(getattr(storage, method, None)):
                raise TypeError(f"prefix cache storage requires callable {method}()")
        buffers = getattr(runner, "cache_storage_buffers", None)
        if not callable(buffers):
            raise TypeError(
                "prefix cache storage requires runner.cache_storage_buffers()"
            )
        self.buffers = buffers
        signature = getattr(runner, "cache_storage_signature", None)
        if callable(signature):
            signature = signature()
        if not isinstance(signature, (str, bytes)) or not signature:
            raise TypeError(
                "prefix cache storage requires a nonempty runner.cache_storage_signature"
            )
        signature = signature.encode() if isinstance(signature, str) else signature
        if namespace is None:
            import os

            namespace = os.urandom(32)
        elif isinstance(namespace, str):
            namespace = namespace.encode()
        if not isinstance(namespace, bytes) or not namespace:
            raise ValueError(
                "prefix_cache_namespace must be nonempty str, bytes, or None"
            )
        self.groups = tuple(layout.groups)
        import math

        self.alignment = math.lcm(*(group.block_tokens for group in self.groups))
        geometry = repr(
            (
                layout.replay_suffix_tokens,
                tuple(
                    (
                        g.block_tokens,
                        g.blocks_per_slab,
                        int(g.family),
                        int(g.retention),
                        g.window_tokens,
                        g.read_window_tokens,
                    )
                    for g in self.groups
                ),
            )
        ).encode()
        self.namespace = hashlib.sha256(
            b"resin-prefix-storage-v1\0"
            + struct.pack("<Q", len(namespace))
            + namespace
            + struct.pack("<Q", len(signature))
            + signature
            + geometry
        ).digest()
        self.storage = storage
        self.max_pending = max_pending
        self.pending = []
        self.ready = {}
        self.metrics = metrics
        self.stats = dict.fromkeys(
            (
                "lookups",
                "hits",
                "misses",
                "loads",
                "stores",
                "load_hits",
                "load_misses",
                "stored_objects",
                "load_bytes",
                "store_bytes",
                "errors",
                "restored_tokens",
            ),
            0,
        )
        self.timings = (
            {kind: _HistogramAccumulator() for kind in ("lookup", "load", "store")}
            if metrics
            else {}
        )

    def can_lookup(self):
        return sum(job.kind == "lookup" for job in self.pending) < self.max_pending

    def _submit(self, kind, keys, buffers, context):
        job = _Pending(
            kind, None, keys, buffers, context, perf_counter() if self.metrics else 0.0
        )
        # Retain this generation even if a provider raises after accepting I/O.
        self.pending.append(job)
        if self.metrics:
            self.stats[
                {"lookup": "lookups", "load": "loads", "store": "stores"}[kind]
            ] += len(keys)
        method = getattr(self.storage, kind)
        try:
            job.operation = method(keys) if kind == "lookup" else method(keys, buffers)
        except Exception as error:
            if kind != "lookup":
                raise
            # Metadata probes borrow no cache memory, so failures can fall back
            # to prefill without a data-transfer retirement fence.
            job.operation = _LookupFailure(len(keys), error)
        if not callable(getattr(job.operation, "poll", None)) or not callable(
            getattr(job.operation, "wait", None)
        ):
            raise TypeError(
                "cache storage submissions must return a pollable, waitable operation"
            )

    def candidates(self, candidates):
        contexts = []
        all_keys = {}
        for request_id, tokens, resident, maximum in candidates:
            keys = _Keys(tokens, maximum, self.groups, self.namespace)
            contexts.append((request_id, resident, keys))
            for values in keys.keys.values():
                all_keys.update(dict.fromkeys(values))
        if contexts:
            self._submit("lookup", tuple(all_keys), (), tuple(contexts))

    def export(self, job_id, tokens, descriptor):
        transfer = _transfer(descriptor)
        keys = _Keys(tokens, transfer.resident_tokens, self.groups, self.namespace)
        ordered = tuple(
            key
            for group in transfer.groups
            for key in keys.at(
                group.group_index, group.first_block, len(group.slab_ids)
            )
        )
        buffers = self._buffers(transfer)
        self._submit("store", ordered, buffers, job_id)

    def _buffers(self, transfer):
        buffers = tuple(self.buffers(transfer))
        expected = sum(len(group.slab_ids) for group in transfer.groups)
        if len(buffers) != expected:
            raise ValueError(
                "cache storage buffers must match the native transfer's block count"
            )
        if any(
            type(buffer.nbytes) is not int or buffer.nbytes < 0 for buffer in buffers
        ):
            raise ValueError(
                "cache storage buffer byte counts must be nonnegative integers"
            )
        return buffers

    def ready_loads(self):
        return tuple((request_id, value[0]) for request_id, value in self.ready.items())

    def resolve(self, request_id):
        self.ready.pop(request_id, None)

    def load(self, request_id, job_id, descriptor):
        length, keys = self.ready[request_id]
        transfer = _transfer(descriptor)
        if transfer.resident_tokens != length:
            raise ValueError("cache storage restore boundary changed after lookup")
        buffers = self._buffers(transfer)
        if len(keys) != len(buffers):
            raise ValueError("cache storage manifest does not match the native restore")
        self._submit("load", keys, buffers, job_id)
        self.resolve(request_id)

    def poll(self, wait=False):
        exports, lookups, loads = [], [], []
        for job in tuple(self.pending):
            try:
                result = job.operation.wait() if wait else job.operation.poll()
            except Exception as error:
                if job.kind != "lookup":
                    raise
                result = CacheResult((False,) * len(job.keys), str(error))
            if result is None and not wait:
                continue
            if (
                not isinstance(result, CacheResult)
                or len(result.success) != len(job.keys)
                or any(type(value) is not bool for value in result.success)
            ):
                raise TypeError(
                    "cache storage completion must return one boolean result per object"
                )
            if result.error is not None and not isinstance(result.error, str):
                raise TypeError(
                    "cache storage completion error must be a string or None"
                )
            if self.metrics:
                self.timings[job.kind].observe(perf_counter() - job.started)
                if result.error is not None:
                    self.stats["errors"] += len(job.keys)
            success = (
                tuple(False for _ in job.keys)
                if result.error is not None
                else result.success
            )
            if job.kind == "lookup":
                present = dict(zip(job.keys, success, strict=True))
                if self.metrics:
                    self.stats["hits"] += sum(success)
                    self.stats["misses"] += len(success) - sum(success)
                for request_id, resident, keys in job.context:
                    length = keys.longest(present, resident, self.alignment)
                    if length:
                        self.ready[request_id] = (length, keys.manifest(length))
                    lookups.append((request_id, length))
            else:
                hits = sum(success)
                if self.metrics:
                    if job.kind == "load":
                        self.stats["load_hits"] += hits
                        self.stats["load_misses"] += len(success) - hits
                    else:
                        self.stats["stored_objects"] += hits
                    self.stats[f"{job.kind}_bytes"] += sum(
                        buffer.nbytes
                        for buffer, hit in zip(job.buffers, success, strict=True)
                        if hit
                    )
                if job.kind == "store":
                    exports.append(job.context)
                else:
                    loads.append((job.context, all(success)))
            self.pending.remove(job)
        return tuple(exports), tuple(lookups), tuple(loads)

    def restored(self, tokens):
        if self.metrics:
            self.stats["restored_tokens"] += tokens

    def metrics_snapshot(self):
        snapshot: dict[str, object] = dict(self.stats.items())
        snapshot.update(
            {
                f"pending_{kind}s": sum(job.kind == kind for job in self.pending)
                for kind in ("lookup", "load", "store")
            }
        )
        snapshot["inflight_bytes"] = sum(
            buffer.nbytes for job in self.pending for buffer in job.buffers
        )
        snapshot.update(
            {
                f"{kind}_seconds": histogram.snapshot()
                for kind, histogram in self.timings.items()
            }
        )
        return snapshot
