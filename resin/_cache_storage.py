"""Engine-owned prefix matching over asynchronous opaque byte storage."""

import hashlib
import math
import os
from itertools import chain

import numpy as np

from resin._resin import CacheGroupFamily


class _Operation:
    def __init__(self, owner, future, finish=lambda value: value, buffers=()):
        self.owner = owner
        self.future, self.finish, self.buffers = future, finish, buffers

    def poll(self, wait=False):
        if not wait and not self.future.done():
            return None
        try:
            value = self.future.result()
        except Exception:
            return 0 if self.future.done() else None
        return None if value is None else self.finish(value)


class _PrefixStorage:
    def __init__(self, store, runner, layout, namespace):
        self.runner = runner
        self.store, self.buffers = store, runner.cache_storage_buffers
        self.groups = layout.groups
        self.alignment = math.lcm(*(g.block_tokens for g in self.groups))
        self.history = {
            i
            for i, g in enumerate(self.groups)
            if g.family != CacheGroupFamily.STATE
            and g.window_resident_block_count is None
        }
        namespace = os.urandom(32) if namespace is None else namespace
        self.namespace = hashlib.sha256(
            namespace.encode() if isinstance(namespace, str) else namespace
        ).digest()

    def _keys(self, tokens):
        encoded, width = np.asarray(tokens, dtype="<u4").tobytes(), self.alignment * 4
        digest, keys = hashlib.sha256(self.namespace), []
        for start in range(0, len(encoded) - width + 1, width):
            digest.update(encoded[start : start + width])
            keys.append(digest.digest())
        return keys

    def lookup(self, tokens, maximum, alignment):
        bases = self._keys(tokens[:maximum])
        tags = ([b"h"] if self.history else []) + [b"t"]
        keys = tuple(key + tag for tag in tags for key in bases)

        def matched(mask):
            present, length = dict(zip(keys, mask, strict=True)), 0
            for i, key in enumerate(bases):
                if self.history and not present[key + b"h"]:
                    break
                if present[key + b"t"]:
                    length = (i + 1) * alignment
            return length

        return _Operation(self, self.store.lookup(keys), matched)

    def transfer(self, tokens, groups, load):
        bases, regions = self._keys(tokens), iter(self.buffers(groups))
        chunks, tail = [()] * len(bases), ()
        for i, (_, rows) in enumerate(groups):
            blocks = tuple(next(regions) for _ in rows)
            if i not in self.history:
                tail += tuple(chain.from_iterable(blocks))
            else:
                width = self.alignment // self.groups[i].block_tokens
                for j in range(len(bases)):
                    chunks[j] += tuple(
                        chain.from_iterable(blocks[j * width : (j + 1) * width])
                    )
        keys, buffers = ([], [])
        if self.history:
            keys, buffers = [key + b"h" for key in bases], chunks
        # Even an empty tail authenticates this exact checkpoint boundary.
        keys.append(bases[-1] + b"t")
        buffers.append(tail)
        future = (self.store.load if load else self.store.store)(
            tuple(keys), tuple(buffers)
        )
        return _Operation(self, future, buffers=buffers)
