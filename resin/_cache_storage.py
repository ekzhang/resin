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
        self.history = {
            i
            for i, g in enumerate(self.groups)
            if g.family != CacheGroupFamily.STATE
            and g.window_resident_block_count is None
        }

    def _keys(self, tokens):
        encoded, width = np.asarray(tokens, dtype="<u4").tobytes(), self.alignment * 4
        digest, keys = hashlib.sha256(), []
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
        import torch

        bases = self._keys(tokens)
        regions = iter(self.buffers(groups, prefix_length=len(tokens)))
        chunks, tail = [{} for _ in bases], {}
        for i, (first, rows) in enumerate(groups):
            width = self.alignment // self.groups[i].block_tokens
            for j in range(len(rows)):
                target = chunks[j // width] if i in self.history else tail
                for name, region in next(regions).items():
                    if not isinstance(name, str) or not isinstance(
                        region, torch.Tensor
                    ):
                        raise TypeError(
                            "cache regions must map string names to tensors"
                        )
                    if region.dtype != torch.uint8 or not region.is_contiguous():
                        raise ValueError("cache regions must be contiguous uint8 views")
                    target[f"{i}/{first + j}/{name}"] = region
        if next(regions, None) is not None:
            raise ValueError("cache export returned too many blocks")
        keys, buffers = ([], [])
        if self.history:
            keys, buffers = [key + b"h" for key in bases], chunks
        # Even an empty tail authenticates this exact checkpoint boundary.
        keys.append(bases[-1] + b"t")
        buffers.append(tail)
        buffers = tuple(dict(sorted(regions.items())) for regions in buffers)
        future = (self.store.load if load else self.store.store)(tuple(keys), buffers)
        return _Operation(self, future, buffers=buffers)
