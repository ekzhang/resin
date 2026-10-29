"""Synchronous prefix matching over a transport-independent byte store."""

import hashlib
import math
import os
import struct

import numpy as np

from resin import CacheGroupFamily


class _PrefixStorage:
    def __init__(self, storage, runner, layout, namespace):
        if not all(
            callable(getattr(storage, name, None))
            for name in ("lookup", "load", "store")
        ):
            raise TypeError("prefix storage requires lookup, load, and store methods")
        self.storage, self.buffers = storage, runner.cache_storage_buffers
        self.groups = layout.groups
        if namespace is None:
            namespace = os.urandom(32)
        if isinstance(namespace, str):
            namespace = namespace.encode()
        if not isinstance(namespace, bytes) or not namespace:
            raise ValueError("prefix_cache_namespace must be nonempty str or bytes")
        self.namespace = hashlib.sha256(namespace).digest()
        self.stats = dict.fromkeys(
            ("lookups", "hits", "load_bytes", "store_bytes", "errors"), 0
        )

    def _keys(self, tokens):
        encoded = np.asarray(tokens, dtype="<u4").tobytes()
        step = math.gcd(*(g.block_tokens for g in self.groups))
        digest, keys = hashlib.sha256(self.namespace), [[] for _ in self.groups]
        for end in range(step, len(tokens) + 1, step):
            digest.update(encoded[(end - step) * 4 : end * 4])
            for index, group in enumerate(self.groups):
                if end % group.block_tokens == 0:
                    keys[index].append(
                        hashlib.sha256(
                            digest.digest() + struct.pack("<II", index, end)
                        ).digest()
                    )
        return keys

    def _probe(self, keys):
        found = tuple(self.storage.lookup(keys))
        if len(found) != len(keys) or any(type(v) is not bool for v in found):
            raise ValueError("storage must return one boolean per key")
        self.stats["lookups"] += len(keys)
        self.stats["hits"] += sum(found)
        return found

    def lookup(self, tokens, maximum, alignment):
        try:
            keys = self._keys(tokens[:maximum])
            flat = tuple(key for group in keys for key in group)
            present = {key for key, ok in zip(flat, self._probe(flat)) if ok}
            for group, group_keys in zip(self.groups, keys):
                if (
                    group.family != CacheGroupFamily.STATE
                    and group.window_resident_block_count is None
                ):
                    for i, key in enumerate(group_keys):
                        if key not in present:
                            maximum = min(maximum, i * group.block_tokens)
                            break
            for length in range(maximum // alignment * alignment, 0, -alignment):
                if all(key in present for key in self._manifest(keys, length)):
                    return length
        except Exception:
            self.stats["errors"] += 1
        return 0

    def _manifest(self, keys, length):
        for group, group_keys in zip(self.groups, keys):
            end = length // group.block_tokens
            count = (
                1
                if group.family == CacheGroupFamily.STATE
                else group.window_resident_block_count
            )
            if count is not None:
                yield from group_keys[max(0, end - count) : end]

    def transfer(self, tokens, groups, load):
        try:
            keys = self._keys(tokens)
            batch = []
            for layout, group, (first, rows) in zip(
                self.groups, keys, groups, strict=True
            ):
                if layout.family == CacheGroupFamily.STATE:
                    first = len(group) - 1
                batch.extend(group[first : first + len(rows)])
            batch = tuple(batch)
            if not load:
                mask = tuple(not found for found in self._probe(batch))
                pending = iter(mask)
                groups = [
                    (first, [row for row in rows if next(pending)])
                    for first, rows in groups
                ]
                batch = tuple(key for key, missing in zip(batch, mask) if missing)
            buffers = self.buffers(groups) if batch else ()
            result = tuple(
                (self.storage.load if load else self.storage.store)(batch, buffers)
            )
            if len(result) != len(batch) or any(type(v) is not bool for v in result):
                raise ValueError("storage must return one boolean per key")
            self.stats["load_bytes" if load else "store_bytes"] += sum(
                sum(r.nbytes for r in regions)
                for regions, ok in zip(buffers, result)
                if ok
            )
            return all(result)
        except Exception:
            self.stats["errors"] += 1
            return False
