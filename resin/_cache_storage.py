"""Engine-owned prefix matching over asynchronous opaque byte storage."""

import hashlib
import math

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
    def __init__(self, store, runner, layout):
        self.runner = runner
        buffers = getattr(runner, "cache_storage_buffers", None)
        if not callable(buffers):
            raise NotImplementedError("prefix storage requires cache_storage_buffers")
        if buffers((), prefix_length=0):
            raise ValueError("empty cache export must return no blocks")
        self.store, self.buffers = store, buffers
        if layout is None:
            raise ValueError("prefix storage requires a managed cache layout")
        self.groups = layout.groups
        self.alignment = math.lcm(*(g.block_tokens for g in self.groups))
        self.widths = tuple(self.alignment // g.block_tokens for g in self.groups)
        self.history = {
            i
            for i, g in enumerate(self.groups)
            if g.family != CacheGroupFamily.STATE
            and g.window_resident_block_count is None
        }

    def _keys(self, tokens, salt=None):
        encoded, width = np.asarray(tokens, dtype="<u4").tobytes(), self.alignment * 4
        digest, keys = hashlib.sha256(), []
        if salt is not None:
            encoded_salt = salt.encode("utf-8")
            digest.update(b"resin-prefix-cache-salt\x00")
            digest.update(len(encoded_salt).to_bytes(8, "little"))
            digest.update(encoded_salt)
        for start in range(0, len(encoded) - width + 1, width):
            digest.update(encoded[start : start + width])
            # A separate key length keeps salted keys disjoint from legacy hashes.
            keys.append((b"s" if salt is not None else b"") + digest.digest())
        return keys

    def lookup(self, tokens, maximum, alignment, salt=None):
        bases = self._keys(tokens[:maximum], salt)
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

    def transfer(self, tokens, groups, load, salt=None):
        import torch

        tensor_type, byte_dtype = torch.Tensor, torch.uint8
        bases = self._keys(tokens, salt)
        regions = iter(self.buffers(groups, prefix_length=len(tokens)))
        chunks, tail = [{} for _ in bases], {}
        for i, (first, rows) in enumerate(groups):
            width = self.widths[i]
            for j in range(len(rows)):
                target = chunks[j // width] if i in self.history else tail
                name_prefix = f"{i}/{first + j}/"
                for name, region in next(regions).items():
                    if not isinstance(name, str) or not isinstance(region, tensor_type):
                        raise TypeError(
                            "cache regions must map string names to tensors"
                        )
                    if region.dtype != byte_dtype or not region.is_contiguous():
                        raise ValueError("cache regions must be contiguous uint8 views")
                    target[name_prefix + name] = region
        if next(regions, None) is not None:
            raise ValueError("cache export returned too many blocks")
        keys, buffers = ([], [])
        if self.history:
            keys, buffers = [key + b"h" for key in bases], chunks
        # Even an empty tail authenticates this exact checkpoint boundary.
        keys.append(bases[-1] + b"t")
        buffers.append(tail)
        buffers = tuple(buffers)
        future = (self.store.load if load else self.store.store)(tuple(keys), buffers)
        return _Operation(self, future, buffers=buffers)
